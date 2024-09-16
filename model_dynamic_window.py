"""
Full definition of a GPT Language Model with Flexible Layer Configuration and Optimized TokenMaskProcessor.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from local_attention import LocalAttention
from dynamic_window_attention import DynamicWindowAttention
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class TokenMaskProcessor:
    def __init__(self, vocab_size: int, mask_tokens: List[str], device: torch.device):
        assert mask_tokens is not None, "mask_tokens must be provided"
        self.vocab_size = vocab_size
        self.mask_tokens = mask_tokens
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.device = device
        self.mask_token_ids = self.get_mask_token_ids()

    def get_mask_token_ids(self):
        return torch.tensor([self.tokenizer.encode(token)[0] for token in self.mask_tokens], device=self.device)

    def create_token_mask(self, idx, include_mask_tokens=True):
        mask = torch.isin(idx, self.mask_token_ids)
        if not include_mask_tokens:
            mask = ~mask
        return mask

    def process_masked_tokens(self, x, idx, block, include_mask_tokens=True):
        B, T, C = x.size()
        x_flat = x.view(B*T, C)
        idx_flat = idx.view(B*T)
        mask_flat = self.create_token_mask(idx_flat, include_mask_tokens=include_mask_tokens)
        if mask_flat.sum() == 0:
            return x  # Nothing to process

        # Extract masked tokens
        x_masked = x_flat[mask_flat].unsqueeze(0)  # Shape: (1, num_masked_tokens, C)
        char_ids_masked = idx_flat[mask_flat].unsqueeze(0)  # Shape: (1, num_masked_tokens)

        # Pass both x and masked char_ids to the block
        x_masked = block(x_masked, char_ids=char_ids_masked)

        # Replace the masked tokens in the original tensor
        x_flat[mask_flat] = x_masked.squeeze(0)
        x = x_flat.view(B, T, C)
        return x


class DynamicWindowSelfAttention(nn.Module):
    def __init__(self, config, delimiter_chars: List[str]):
        """
        Initializes the DynamicWindowSelfAttention module.

        Args:
            config (GPTConfig): Configuration object.
            delimiter_chars (List[str]): List of delimiter characters for dynamic window attention.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Verify delimiter_chars is not empty and contains characters
        if not delimiter_chars or not any(delimiter_chars):
            raise ValueError("delimiter_chars must be provided and contain characters")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.dynamic_window_attn = DynamicWindowAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            delimiter_chars=delimiter_chars,
            normalize_v=True
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DynamicWindowSelfAttention.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C).
            char_ids (torch.Tensor): Character IDs tensor with shape (B, T).

        Returns:
            torch.Tensor: Output tensor after applying dynamic window attention.
        """
        # Early return if tensor 'x' has any empty dimensions
        if x.numel() == 0:
            return x

        y = self.dynamic_window_attn(x, char_ids)
        y = self.resid_dropout(y)
        return y


class CausalSelfAttention(nn.Module):
    def __init__(self, config, use_local_attention: bool = False):
        """
        Initializes the CausalSelfAttention module.

        Args:
            config (GPTConfig): Configuration object.
            use_local_attention (bool, optional): Whether to use local attention. Defaults to False.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.use_local_attention = use_local_attention
        self.window_size = 8  # Fixed window size for local attention
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        if self.use_local_attention:
            self.local_attn = LocalAttention(
                dim=config.n_embd // config.n_head,
                window_size=self.window_size,
                causal=True,
                dropout=config.dropout
            )
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CausalSelfAttention.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying causal self-attention.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if self.use_local_attention:
            # Pad the input to make it a multiple of the window size
            original_T = T
            if T % self.window_size != 0:
                padding = self.window_size - (T % self.window_size)
                x = F.pad(x, (0, 0, 0, padding))
                B, T, C = x.size()  # Update T after padding

            q = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = q
            v = q
            mask = torch.ones(B, T, device=x.device).bool()  # (B, T)
            y = self.local_attn(q, k, v, mask=mask)  # (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

            # Remove padding if we added it
            if T > original_T:
                y = y[:, :original_T, :]
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                )
            else:
                # Create a mask for non-zero rows
                non_zero_rows = (x.abs().sum(dim=-1) != 0)  # Shape: (B, T)

                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

                # Combine causal mask with non-zero rows mask
                causal_mask = self.bias[:, :, :T, :T] == 1
                combined_mask = causal_mask & non_zero_rows.unsqueeze(1).unsqueeze(2)
                att = att.masked_fill(~combined_mask, float('-inf'))

                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

            y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        if not self.use_local_attention:
            y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        """
        Initializes the MLP module.

        Args:
            config (GPTConfig): Configuration object.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after MLP.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, attention_type: str = 'causal', delimiter_chars: List[str] = None):
        """
        Initializes the Transformer Block.

        Args:
            config (GPTConfig): Configuration object.
            attention_type (str, optional): Type of attention ('dynamic_window' or 'causal'). Defaults to 'causal'.
            delimiter_chars (List[str], optional): Delimiter characters for dynamic window attention. Defaults to None.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        if attention_type == 'dynamic_window':
            assert delimiter_chars is not None, "delimiter_chars must be provided for dynamic_window attention"
            self.attn = DynamicWindowSelfAttention(config, delimiter_chars=delimiter_chars)
        elif attention_type == 'causal':
            self.attn = CausalSelfAttention(config)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)
        self.attention_type = attention_type

    def forward(self, x: torch.Tensor, char_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Transformer Block.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C).
            char_ids (torch.Tensor, optional): Character IDs tensor with shape (B, T). Required for dynamic window attention.

        Returns:
            torch.Tensor: Output tensor after applying attention and MLP.
        """
        if self.attention_type == 'dynamic_window':
            assert char_ids is not None, "char_ids must be provided for dynamic window attention"
            x = x + self.attn(self.ln_1(x), char_ids=char_ids)
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_tokens: List[str] = field(default_factory=lambda: ['.', '?', '!', '\n'])

    # Layer configuration: list of dictionaries specifying layer types and their counts
    layer_types: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "dynamic_window_word", "count": 2, "delimiters": [' ', '.', '?', '!', '\n']},
        {"type": "dynamic_window_sentence", "count": 2, "delimiters": ['.', '?', '!', '\n']},
        {"type": "causal_eos", "count": 7},
        {"type": "dynamic_window_sentence_char", "count": 1, "delimiters": ['.', '?', '!', '\n']}
    ])


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        """
        Initializes the GPT model with a flexible layer configuration.

        Args:
            config (GPTConfig): Configuration object.
        """
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        assert config.mask_tokens is not None, "mask_tokens must be provided in config"
        self.config = config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(),  # To be filled based on layer_types
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Build the layers based on layer_types
        for layer_spec in config.layer_types:
            layer_type = layer_spec["type"]
            count = layer_spec["count"]
            delimiters = layer_spec.get("delimiters", None)
            for _ in range(count):
                if layer_type == "dynamic_window_word":
                    block = Block(
                        config,
                        attention_type='dynamic_window',
                        delimiter_chars=delimiters
                    )
                elif layer_type == "dynamic_window_sentence":
                    block = Block(
                        config,
                        attention_type='dynamic_window',
                        delimiter_chars=delimiters
                    )
                elif layer_type == "causal_eos":
                    block = Block(
                        config,
                        attention_type='causal'
                    )
                elif layer_type == "dynamic_window_sentence_char":
                    block = Block(
                        config,
                        attention_type='dynamic_window',
                        delimiter_chars=delimiters
                    )
                else:
                    raise ValueError(f"Unknown layer type: {layer_type}")
                self.transformer.h.append(block)

        # Language Model Head with weight tying
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize TokenMaskProcessor
        self.token_mask_processor = TokenMaskProcessor(
            config.vocab_size,
            config.mask_tokens,
            device=device
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(config.layer_types)))

        # Move to device
        self.to(device)

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Returns the number of parameters in the model.

        Args:
            non_embedding (bool, optional): If True, excludes position embeddings. Defaults to True.

        Returns:
            int: Number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        """
        Initializes the weights of the model.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_block(self, layer_index: int) -> Block:
        """
        Retrieves a specific block based on layer index.

        Args:
            layer_index (int): Index of the layer.

        Returns:
            Block: Transformer block.
        """
        # Not used anymore as layers are built based on layer_types
        pass

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass for the GPT model.

        Args:
            idx (torch.Tensor): Input token indices tensor with shape (B, T).
            targets (torch.Tensor, optional): Target token indices tensor with shape (B, T). Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and loss (if targets provided).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # Logging (can be adjusted based on logging level)
        logging.debug(f"Input idx shape: {idx.shape}")
        logging.debug(f"Embedding weight shape: {self.transformer.wte.weight.shape}")

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # shape (b, t, n_embd)

        logging.debug(f"Shape after embedding: {x.shape}")

        for i, block in enumerate(self.transformer.h):
            layer_type = self._get_layer_type(i)
            if layer_type == "dynamic_window_word":
                # Layers 1 and 2: Within a word
                x = block(x, char_ids=idx)
            elif layer_type == "dynamic_window_sentence":
                # Layers 3 and 4: Within a sentence
                x = self.token_mask_processor.process_masked_tokens(
                    x, idx, block, include_mask_tokens=True
                )
            elif layer_type == "causal_eos":
                # Layers 5 to 11: Only end-of-sentence tokens
                x = self.token_mask_processor.process_masked_tokens(
                    x, idx, block, include_mask_tokens=True
                )
            elif layer_type == "dynamic_window_sentence_char":
                # Layer 12: Per character within a sentence
                x = block(x, char_ids=idx)
            else:
                raise ValueError(f"Unknown layer type encountered during forward: {layer_type}")

            logging.debug(f"Shape after block {i}: {x.shape}")

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Return logits without loss
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def _get_layer_type(self, layer_index: int) -> str:
        """
        Retrieves the layer type based on the layer index.

        Args:
            layer_index (int): Index of the layer.

        Returns:
            str: Type of the layer.
        """
        cumulative = 0
        for layer_spec in self.config.layer_types:
            count = layer_spec["count"]
            if layer_index < cumulative + count:
                return layer_spec["type"]
            cumulative += count
        raise ValueError(f"Layer index {layer_index} out of range based on layer_types configuration.")

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Dict[str, Any] = None):
        """
        Loads a pretrained GPT model and initializes the custom GPT model with the pretrained weights.

        Args:
            model_type (str): Type of the pretrained model ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl').
            override_args (Dict[str, Any], optional): Arguments to override in the configuration. Defaults to None.

        Returns:
            GPT: Initialized GPT model with pretrained weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unsupported model_type"
        override_args = override_args or {}  # default to empty dict
        # Only dropout can be overridden
        assert all(k == 'dropout' for k in override_args), "Only 'dropout' can be overridden"

        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

        # Define base configurations based on model_type
        config_args = {
            'gpt2':         {"n_layer": 12, "n_head": 12, "n_embd": 768},    # 124M params
            'gpt2-medium':  {"n_layer": 24, "n_head": 16, "n_embd": 1024},   # 350M params
            'gpt2-large':   {"n_layer": 36, "n_head": 20, "n_embd": 1280},   # 774M params
            'gpt2-xl':      {"n_layer": 48, "n_head": 25, "n_embd": 1600},   # 1558M params
        }[model_type]

        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # Always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # Always 1024 for GPT model checkpoints
        config_args['bias'] = True        # Always True for GPT model checkpoints

        # Override dropout if specified
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        else:
            config_args['dropout'] = 0.0  # Default dropout

        # Create GPTConfig
        config = GPTConfig(**config_args)

        # Adjust layer_types based on n_layer
        total_layers = config_args["n_layer"]
        current_layer = 0
        layer_types = []

        # Example configuration based on original specifications:
        # Layers 1-2: dynamic_window_word
        # Layers 3-4: dynamic_window_sentence
        # Layers 5-11: causal_eos
        # Layer 12: dynamic_window_sentence_char
        # Repeat or adjust based on total_layers

        # Here, for flexibility, we ensure the total layers match
        for spec in config.layer_types:
            layer_types.append(spec)

        # Adjust if the total layers do not match
        specified_layers = sum(spec["count"] for spec in config.layer_types)
        if specified_layers > total_layers:
            raise ValueError("Specified layer counts exceed total number of layers.")
        elif specified_layers < total_layers:
            # Add remaining layers as causal_eos by default
            remaining = total_layers - specified_layers
            layer_types.append({"type": "causal_eos", "count": remaining})

        config.layer_types = layer_types

        # Initialize the custom GPT model
        model = cls(config)

        # Load pretrained weights from HuggingFace's GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Initialize state_dict for the custom model
        sd_custom = model.state_dict()

        # Exclude attention bias and masked bias
        sd_keys_hf = [k for k in sd_hf.keys() if not (k.endswith('.attn.masked_bias') or k.endswith('.attn.bias'))]
        sd_keys_custom = [k for k in sd_custom.keys() if not (k.endswith('.attn.bias'))]

        # Define which weights need to be transposed
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        # Ensure the number of keys match
        assert len(sd_keys_hf) == len(sd_keys_custom), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys_custom)}"

        # Copy weights from HuggingFace model to custom model
        for k_hf, k_custom in zip(sd_keys_hf, sd_keys_custom):
            if any(k_hf.endswith(w) for w in transposed):
                # Transpose weights for certain layers
                assert sd_hf[k_hf].shape[::-1] == sd_custom[k_custom].shape, f"Shape mismatch for {k_hf}"
                sd_custom[k_custom].copy_(sd_hf[k_hf].t())
            else:
                # Direct copy for other layers
                assert sd_hf[k_hf].shape == sd_custom[k_custom].shape, f"Shape mismatch for {k_hf}"
                sd_custom[k_custom].copy_(sd_hf[k_hf])

        print("Pretrained weights loaded successfully.")

        return model

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str) -> torch.optim.Optimizer:
        """
        Configures the optimizer with weight decay settings.

        Args:
            weight_decay (float): Weight decay factor.
            learning_rate (float): Learning rate.
            betas (tuple): Betas for the AdamW optimizer.
            device_type (str): Type of device ('cuda' or 'cpu').

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create parameter groups based on dimensionality
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimates the model's FLOPs utilization (MFU).

        Args:
            fwdbwd_per_iter (int): Number of forward-backward passes per iteration.
            dt (float): Time taken per iteration in seconds.

        Returns:
            float: Model FLOPs utilization.
        """
        """ 
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        """
        N = self.get_num_params()
        cfg = self.config
        L = sum(spec["count"] for spec in cfg.layer_types)
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Generates new tokens by autoregressively sampling from the model.

        Args:
            idx (torch.Tensor): Input token indices tensor with shape (B, T).
            max_new_tokens (int): Number of tokens to generate.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing the generated token indices with shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx