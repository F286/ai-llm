import torch
from typing import List

class CharacterDelimitedAttentionMask:
    def __init__(self, delimiter_chars: List[str]):
        self.delimiter_chars = delimiter_chars

    def create_causal_delimiter_mask(self, char_ids: torch.Tensor) -> torch.Tensor:
        delimiter_mask = self._create_delimiter_mask(char_ids)
        causal_mask = self._create_causal_mask(char_ids.shape[1], char_ids.device)
        return delimiter_mask & causal_mask

    def _create_delimiter_mask(self, char_ids: torch.Tensor) -> torch.Tensor:
        delimiter_tensor = self._create_delimiter_tensor(char_ids)
        region_tensor = self._create_region_tensor(delimiter_tensor)
        return self._create_region_mask(region_tensor)

    def _create_delimiter_tensor(self, char_ids: torch.Tensor) -> torch.Tensor:
        delimiter_tensor = torch.zeros_like(char_ids, dtype=torch.float)
        for char in self.delimiter_chars:
            delimiter_tensor += (char_ids == ord(char)).float()
        return delimiter_tensor

    def _shift_delimiter_tensor(self, delimiter_tensor: torch.Tensor) -> torch.Tensor:
        shifted_delimiter_tensor = torch.zeros_like(delimiter_tensor)
        shifted_delimiter_tensor[:, 1:] = delimiter_tensor[:, :-1]
        return shifted_delimiter_tensor

    def _create_region_tensor(self, delimiter_tensor: torch.Tensor) -> torch.Tensor:
        shifted_delimiter_tensor = self._shift_delimiter_tensor(delimiter_tensor)
        return torch.cumsum(shifted_delimiter_tensor, dim=1)

    def _create_region_mask(self, region_tensor: torch.Tensor) -> torch.Tensor:
        return region_tensor.unsqueeze(1) == region_tensor.unsqueeze(2)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))