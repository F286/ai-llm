import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from character_delimited_attention_mask  import CharacterDelimitedAttentionMask

class CharacterDelimitedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, delimiter_chars: List[str], normalize_v: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
        self.attention_mask = CharacterDelimitedAttentionMask(delimiter_chars)
        self.normalize_v = normalize_v

    def forward(self, x: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        qkv = self.generate_qkv(x)
        q, k, v = self.split_qkv(qkv)
        attention_mask = self.create_attention_mask(char_ids)
        attention_weights = self.calculate_attention_weights(q, k, attention_mask)
        output = self.apply_attention(v, attention_weights)
        return output

    def generate_qkv(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj(x)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(-1, q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.normalize_v:
            v = v / self.head_dim

        return q, k, v

    def create_attention_mask(self, char_ids: torch.Tensor) -> torch.Tensor:
        return self.attention_mask.create_causal_delimiter_mask(char_ids)

    def calculate_attention_weights(self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_scale = 1.0 / (self.head_dim ** 0.5)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * attention_scale
        attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), -1e9)
        return F.softmax(attention_scores, dim=-1)

    def apply_attention(self, v: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        context = torch.matmul(attention_weights, v)
        return context.transpose(1, 2).contiguous().view(context.size(0), -1, self.embed_dim)