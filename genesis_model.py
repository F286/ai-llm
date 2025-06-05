"""Experimental GPT model with Genesis Attention mechanism.

This model mirrors the original GPT architecture but replaces the
standard scaled dot product attention with *Genesis Attention*.
Genesis Attention divides token representations into a number of
low-dimensional "concept" subspaces. Attention weights between two
positions are obtained by computing similarity within each concept
subspace and then taking the minimum across all concepts. No softmax
normalisation is applied. Two auxiliary losses are used during
training:

1. **Upper bound loss** – penalises values in the similarity matrix
   that exceed 1. This prevents weights from growing without bound.
2. **Sum-to-one loss** – encourages the weights from each query
   position to sum to one, which stabilises learning in the absence of
   softmax normalisation.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConceptMatcher(nn.Module):
    """Project tokens into concept spaces and compute similarities."""

    def __init__(self, embed_dim: int, num_heads: int,
                 num_concepts: int = 8, concept_dim: int = 16):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.num_heads = num_heads
        out_dim = num_heads * num_concepts * concept_dim
        self.q_proj = nn.Linear(embed_dim, out_dim)
        self.k_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per concept similarity matrices.

        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, H, T, T, num_concepts)
        """
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        q = q.view(B, T, self.num_heads, self.num_concepts, self.concept_dim)
        k = k.view(B, T, self.num_heads, self.num_concepts, self.concept_dim)
        # (B, H, T, C, D)
        q = q.permute(0, 2, 1, 3, 4)
        k = k.permute(0, 2, 1, 3, 4)
        # similarity per concept -> (B, H, C, T, T)
        scores = torch.einsum('bhtcd,bhscd->bhtsc', q, k)
        scores = scores.view(B, self.num_heads, T, T, self.num_concepts)
        scores = scores / math.sqrt(self.concept_dim)
        return scores


class GenesisAttention(nn.Module):
    """Multi-head attention based on minimum similarity across concepts."""

    def __init__(self, config, num_concepts: int = 8, concept_dim: int = 16):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.matcher = ConceptMatcher(config.n_embd, config.n_head,
                                      num_concepts, concept_dim)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # concept based similarity
        sim = self.matcher(x)  # (B, H, T, T, C)
        weights = sim.min(dim=-1).values  # (B, H, T, T)

        # causal masking
        causal_mask = self.bias[:, :, :T, :T] == 1
        weights = weights.masked_fill(~causal_mask, 0)

        out = torch.matmul(weights, v)  # (B, H, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        loss_upper = F.relu(weights - 1).pow(2).mean()
        loss_sum = (weights.sum(dim=-1) - 1).pow(2).mean()
        extra_loss = loss_upper + loss_sum
        return out, extra_loss


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = GenesisAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, loss = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, loss


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    sentence_end_tokens: List[str] = field(default_factory=lambda: ['.', '?', '!', '\n'])


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        total_extra_loss = 0.0
        for block in self.transformer.h:
            x, loss = block(x)
            total_extra_loss = total_extra_loss + loss
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss + total_extra_loss
        return logits, loss
