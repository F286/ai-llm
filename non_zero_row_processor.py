import torch

# Fully removes non-zero rows from within batches for efficiency reasons.
# 1. 'compacts' the rows so that all the rows that are fully zero are at the end.
# 2. If possible, trim the size of the batch so that 'all zero' rows at the very bottom are removed if they exist the same in all batches. do not overtrim (do not remove rows that have actual elements in them)
class NonZeroRowProcessor:
    def __init__(self, x):
        self.x = x
        self.original_shape = x.shape
        self.non_zero_mask = (x.abs().sum(dim=-1) != 0)
        self.max_non_zero_rows = self.non_zero_mask.sum(dim=1).max().item()
        
        if self.max_non_zero_rows > 0:
            # Create indices for non-zero rows, preserving their original order
            indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            
            # Gather non-zero rows
            self.x_non_zero = torch.zeros((x.shape[0], self.max_non_zero_rows, x.shape[2]), device=x.device)
            for i in range(x.shape[0]):
                non_zero_indices = indices[i][self.non_zero_mask[i]]
                self.x_non_zero[i, :len(non_zero_indices)] = x[i, non_zero_indices]
        else:
            # Handle the case where all rows are zero
            self.x_non_zero = x.new_zeros((x.shape[0], 0, x.shape[2]))

    def __enter__(self):
        return self.x_non_zero

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            result = torch.zeros_like(self.x)
            for i in range(self.x.shape[0]):
                non_zero_indices = torch.nonzero(self.non_zero_mask[i]).squeeze(1)
                if non_zero_indices.numel() > 0:
                    result[i, non_zero_indices] = self.x_non_zero[i, :len(non_zero_indices)]
            self.x.data.copy_(result)
        return False  # Propagate exceptions