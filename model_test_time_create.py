"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

This code has been modified to generate and use a small single-layer transformer network
after a certain cutoff token count during training. The larger model first processes
the initial tokens up to a cutoff, then derives the parameters of a smaller model that
processes the remaining tokens. The small model's parameters are generated on-the-fly
from the hidden states of the larger model.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.

    This is identical to PyTorch's nn.LayerNorm except that it allows to disable bias.
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    Optionally uses PyTorch's built-in scaled_dot_product_attention if available.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention requires PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # lower-triangular causal mask
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, seq length, embedding dimension
        # calculate Q,K,V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        if self.flash:
            # use PyTorch 2.0 flash attention if available
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # standard attention (less efficient)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ The feed-forward block of a transformer: a simple 2-layer MLP with GELU. """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ An unassuming Transformer block. """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # attention + residual
        x = x + self.attn(self.ln_1(x))
        # feedforward + residual
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """ GPT model configuration """
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size: 50257, rounded up for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # New parameter for cutoff:
    generated_small_cutoff: int = 32  # after this token count, switch to generated small model

class GPT(nn.Module):
    """
    A GPT Language Model. This class also has been extended to:
    - Use a large transformer for the first 'cutoff' tokens.
    - Generate a small single-layer transformer model from the large model's last hidden state at the cutoff.
    - The small model processes all remaining tokens beyond the cutoff.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer main body
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying between wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights
        self.apply(self._init_weights)
        # special scaled init for residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # Define parameters for the small "generated" model
        # This small model is just a single-layer transformer with much smaller embedding dimension.
        self.small_n_embd = 32
        self.small_n_head = 1
        # define shapes of all parameters in small model
        self._small_param_shapes = self._get_small_model_param_shapes(config)
        self._small_total_params = sum(p.numel() for p in self._small_param_shapes)
        # a linear layer to produce small model parameters from the big model's last hidden state at cutoff
        self.small_param_proj = nn.Linear(config.n_embd, self._small_total_params, bias=False)

    def _get_small_model_param_shapes(self, config):
        """
        Return a list of Tensors (empty in content, just shape) that define
        the parameterization of the small generated transformer layer:
        - Includes embeddings, layer norms, attention, MLP, and final lm_head.
        """
        vs = config.vocab_size
        bs = config.block_size
        E = self.small_n_embd
        shapes = [
            torch.empty(vs, E),     # wte_small.weight
            torch.empty(bs, E),     # wpe_small.weight
            torch.empty(E),         # ln_1.weight
            torch.empty(E),         # ln_1.bias
            torch.empty(E, 3*E),    # attn.c_attn.weight
            torch.empty(3*E),       # attn.c_attn.bias
            torch.empty(E, E),      # attn.c_proj.weight
            torch.empty(E),         # attn.c_proj.bias
            torch.empty(E),         # ln_2.weight
            torch.empty(E),         # ln_2.bias
            torch.empty(4*E, E),    # mlp.c_fc.weight
            torch.empty(4*E),       # mlp.c_fc.bias
            torch.empty(E, 4*E),    # mlp.c_proj.weight
            torch.empty(E),         # mlp.c_proj.bias
            torch.empty(vs, E),     # lm_head.weight
        ]
        return shapes

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count, the position embeddings are subtracted out.
        The token embeddings are used also as weights in the final layer (weight tying),
        so they are counted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights in a manner similar to the GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
        """
        Model surgery to reduce the block_size if necessary.
        Useful if loading GPT-2 pretrained (block_size=1024) but wanting smaller context.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a GPT model from a pretrained Hugging Face GPT-2 model.
        Available model_type: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        """
        from transformers import GPT2LMHeadModel
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        # only dropout can be overridden
        assert all(k == 'dropout' for k in override_args)
        print("loading weights from pretrained gpt: %s" % model_type)

        # resolve model config from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # GPT-2 uses a "Conv1D" module that maps to Linear + transpose
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} vs {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose these weights to match our Linear layer shapes
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # direct copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters into those that will experience weight decay and those that won't.
        This is following the GPT-2 paper and general practice:
        - All weights of matmuls are decayed
        - Biases and LayerNorm/embedding weights are not decayed
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def _build_small_model(self, param_vector, idx, targets, device):
        """
        Given the param_vector (produced by small_param_proj), construct the small model's weights,
        and run a forward pass on the remainder of the sequence (idx_small).

        The small model is a single-layer transformer with a tiny embedding dimension.
        This function manually performs embedding lookup, layer norm, attention, MLP,
        and final projection using the extracted parameters.

        param_vector: (b, small_total_params)
        idx: the input token indices for the remainder of the sequence
        targets: the corresponding targets (not directly used here, only logits returned)
        device: device on which computations are performed
        """
        p = param_vector[0]  # we assume a single small model per batch for simplicity
        shapes = self._small_param_shapes
        flat = p
        offset = 0
        params = []
        for s in shapes:
            n = s.numel()
            params.append(flat[offset:offset+n].view(s.shape))
            offset += n

        # Unpack params for readability
        wte_small_weight = params[0]
        wpe_small_weight = params[1]
        ln_1_weight = params[2]
        ln_1_bias   = params[3]
        c_attn_weight = params[4]
        c_attn_bias = params[5]
        c_proj_weight = params[6]
        c_proj_bias = params[7]
        ln_2_weight = params[8]
        ln_2_bias   = params[9]
        c_fc_weight = params[10]
        c_fc_bias = params[11]
        c_proj2_weight = params[12]
        c_proj2_bias = params[13]
        lm_head_weight  = params[14]

        b, t = idx.size()
        vs = self.config.vocab_size
        E = self.small_n_embd

        # token and position embedding
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = F.embedding(idx, wte_small_weight) # (b,t,E)
        pos_emb = F.embedding(pos, wpe_small_weight) # (t,E)
        x = tok_emb + pos_emb  # (b,t,E)

        # layer norm 1 (similar to LN_1 in main model)
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1,keepdim=True,unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var+1e-5)
        x_norm = x_norm * ln_1_weight + ln_1_bias

        # single-head self-attention
        attn_in = x_norm @ c_attn_weight + c_attn_bias
        q, k, v = attn_in.split(E, dim=2)
        q = q.view(b, t, self.small_n_head, E//self.small_n_head).transpose(1,2)
        k = k.view(b, t, self.small_n_head, E//self.small_n_head).transpose(1,2)
        v = v.view(b, t, self.small_n_head, E//self.small_n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(t,t, device=device, dtype=torch.bool))
        att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(b,t,E)
        y = y @ c_proj_weight + c_proj_bias
        x = x + y  # residual connection

        # layer norm 2
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1,keepdim=True,unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var+1e-5)
        x_norm = x_norm * ln_2_weight + ln_2_bias

        # MLP
        hidden = x_norm @ c_fc_weight.T + c_fc_bias
        hidden = F.gelu(hidden)
        hidden = hidden @ c_proj2_weight.T + c_proj2_bias
        x = x + hidden  # residual connection

        # final logits
        logits = x @ lm_head_weight.T  # (b,t,vs)
        return logits

    def forward(self, idx, targets=None):
        """
        Forward pass of the model. If sequence length <= cutoff, run the standard big model.
        If sequence length > cutoff, split input into two parts:
        - first part handled by big model
        - second part handled by generated small model
        Combine their outputs and compute a joint loss if targets are provided.
        """
        device = idx.device
        b, t = idx.size()

        cutoff = self.config.generated_small_cutoff
        if t <= cutoff:
            # normal big model path
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)   # (b,t,n_embd)
            pos_emb = self.transformer.wpe(pos)   # (t,n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            # run the transformer layers
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1
                )
            else:
                # inference-time optimization: only forward last token through lm_head
                logits = logits[:,[-1],:]
                loss = None
            return logits, loss
        else:
            # sequence is longer than cutoff
            cutoff_idx = cutoff
            idx_big = idx[:, :cutoff_idx]
            idx_small = idx[:, cutoff_idx:]
            targets_big = targets[:, :cutoff_idx] if targets is not None else None
            targets_small = targets[:, cutoff_idx:] if targets is not None else None

            # Run big model on first cutoff tokens
            pos = torch.arange(0, cutoff_idx, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx_big)
            pos_emb = self.transformer.wpe(pos)
            x_big = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x_big = block(x_big)
            x_big = self.transformer.ln_f(x_big)
            logits_big = self.lm_head(x_big)  # (b,cutoff_idx,vocab_size)

            # Generate small model params from the final hidden state of big model at cutoff-1 token
            last_h = x_big[:, -1, :]  # (b,n_embd)
            small_params = self.small_param_proj(last_h)  # (b, small_total_params)

            # Run small model on remainder tokens
            logits_small = self._build_small_model(small_params, idx_small, targets_small, device=device)

            # Combine big and small model logits
            full_logits = torch.zeros(b, t, self.config.vocab_size, device=device, dtype=logits_big.dtype)
            full_logits[:, :cutoff_idx] = logits_big
            full_logits[:, cutoff_idx:] = logits_small

            if targets is not None:
                loss_big = F.cross_entropy(
                    logits_big.reshape(-1, logits_big.size(-1)),
                    targets_big.reshape(-1),
                    ignore_index=-1
                )
                loss_small = F.cross_entropy(
                    logits_small.reshape(-1, logits_small.size(-1)),
                    targets_small.reshape(-1),
                    ignore_index=-1
                )
                # Weighted average of the losses by the number of tokens each handles
                loss = (loss_big * cutoff_idx + loss_small * (t - cutoff_idx)) / t
            else:
                loss = None
                # inference time: only last token
                full_logits = full_logits[:, [-1], :]

            return full_logits, loss

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        Based on PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak FLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor (b,t)) and complete
        the sequence max_new_tokens times. The model operates in autoregressive mode.
        """
        for _ in range(max_new_tokens):
            # crop the context if it gets too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # pluck logits at final step
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
