import torch
from typing import Optional
import itertools

class CompressedTensor:
    def __init__(self, dense_tensor: torch.Tensor, indices: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self._validate_inputs(dense_tensor, indices, mask)
        self.data = dense_tensor
        self.indices = indices
        self.mask = mask if mask is not None else (indices != -1)
        self.original_shape = dense_tensor.shape
        self.compressed_data = self._compress()

    def _validate_inputs(self, dense_tensor: torch.Tensor, indices: torch.Tensor, mask: Optional[torch.Tensor]):
        if not isinstance(dense_tensor, torch.Tensor) or not isinstance(indices, torch.Tensor):
            raise TypeError("dense_tensor and indices must be torch.Tensor")
        if dense_tensor.dim() != indices.dim() + 1:
            raise ValueError(f"indices must have one dimension less than dense_tensor. Got dense_tensor.dim()={dense_tensor.dim()} and indices.dim()={indices.dim()}")
        if dense_tensor.shape[:-2] != indices.shape[:-1]:
            raise ValueError(f"Shape mismatch: dense_tensor batch dimensions {dense_tensor.shape[:-2]} must match indices batch dimensions {indices.shape[:-1]}")
        if mask is not None and mask.shape != indices.shape:
            raise ValueError(f"mask shape {mask.shape} must match indices shape {indices.shape}")
        if dense_tensor.device != indices.device or (mask is not None and dense_tensor.device != mask.device):
            raise ValueError("dense_tensor, indices, and mask must all be on the same device")

    def _compress(self) -> torch.Tensor:
        batch_dims = self.data.shape[:-2]
        max_valid_indices = self.mask.sum(dim=-1).max().item()
        compressed_shape = batch_dims + (max_valid_indices, self.data.shape[-1])
        compressed = torch.zeros(compressed_shape, dtype=self.data.dtype, device=self.data.device)
        
        for idx in itertools.product(*[range(dim) for dim in batch_dims]):
            valid_indices = self.indices[idx][self.mask[idx]]
            compressed[idx][:len(valid_indices)] = self.data[idx][valid_indices]
        
        return compressed

    def to_compressed(self) -> torch.Tensor:
        return self.compressed_data

    def to_dense(self) -> torch.Tensor:
        dense = torch.zeros_like(self.data)
        batch_dims = self.data.shape[:-2]
        
        for idx in itertools.product(*[range(dim) for dim in batch_dims]):
            valid_indices = self.indices[idx][self.mask[idx]]
            dense[idx][valid_indices] = self.compressed_data[idx][:len(valid_indices)]
        
        return dense

    def __repr__(self):
        return f'CompressedTensor(original shape={self.original_shape}, compressed shape={self.compressed_data.shape})'

    @property
    def shape(self):
        return self.compressed_data.shape

    @property
    def device(self):
        return self.compressed_data.device

    @property
    def dtype(self):
        return self.compressed_data.dtype

def test_compressed_tensor_3d():
    # Create a dense tensor with 2 batches, 4 rows, and 3 elements each
    dense_tensor = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
    ])
    
    # Indices to 'grab' a separate number of elements for each batch
    indices = torch.tensor([[1, 2, -1], [0, 2, 3]])
    
    # Create CompressedTensor
    compressed = CompressedTensor(dense_tensor, indices)

    print("3D CompressedTensor:")
    print(compressed)

    compressed_data = compressed.to_compressed()
    print("\nCompressed data:")
    print(compressed_data)

    assert compressed_data.shape == (2, 3, 3), f"Expected shape (2, 3, 3), but got {compressed_data.shape}"
    assert torch.all(compressed_data[0, 2] == 0), "Third row in first batch should be padded with zeros"

    reconstructed = compressed.to_dense()
    print("\nReconstructed dense tensor:")
    print(reconstructed)

    mask = compressed.mask
    for batch in range(2):
        batch_indices = indices[batch, mask[batch]]
        assert torch.all(reconstructed[batch, batch_indices] == dense_tensor[batch, batch_indices]), f"Mismatch in batch {batch}"

    print("3D test passed successfully!")

def test_compressed_tensor_4d():
    # Create a 4D dense tensor: 2 batches, 3 channels, 4 rows, and 5 elements each
    dense_tensor = torch.randn(2, 3, 4, 5)
    
    # Indices to 'grab' a separate number of elements for each batch and channel
    indices = torch.tensor([
        [[0, 2, -1, -1], [1, 3, -1, -1], [0, 1, 2, -1]],
        [[1, 2, 3, -1], [0, 1, -1, -1], [2, 3, -1, -1]]
    ])
    
    # Create CompressedTensor
    compressed = CompressedTensor(dense_tensor, indices)

    print("\n4D CompressedTensor:")
    print(compressed)

    compressed_data = compressed.to_compressed()
    print("\nCompressed data shape:")
    print(compressed_data.shape)

    assert compressed_data.shape == (2, 3, 3, 5), f"Expected shape (2, 3, 3, 5), but got {compressed_data.shape}"

    reconstructed = compressed.to_dense()
    print("\nReconstructed dense tensor shape:")
    print(reconstructed.shape)

    assert reconstructed.shape == dense_tensor.shape, "Reconstructed shape doesn't match original"

    mask = compressed.mask
    for batch in range(2):
        for channel in range(3):
            channel_indices = indices[batch, channel, mask[batch, channel]]
            assert torch.all(reconstructed[batch, channel, channel_indices] == dense_tensor[batch, channel, channel_indices]), f"Mismatch in batch {batch}, channel {channel}"

    print("4D test passed successfully!")

if __name__ == "__main__":
    test_compressed_tensor_3d()
    test_compressed_tensor_4d()