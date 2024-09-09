import unittest
import torch
from model import GPTConfig, MLP
from non_zero_row_processor import NonZeroRowProcessor

class TestNonZeroRowProcessor(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=1024,
            bias=False,
            vocab_size=50257,
            dropout=0.0
        )
        self.mlp = MLP(self.config)

    def test_non_zero_row_processor(self):
        x = torch.tensor([
            [[1.0, 2.0, 3.0] + [0.0] * 125],
            [[0.0] * 128],
            [[4.0, 5.0, 6.0] + [0.0] * 125],
            [[0.0] * 128],
            [[7.0, 8.0, 9.0] + [0.0] * 125]
        ])

        with NonZeroRowProcessor(x) as x_processed:
            self.assertEqual(x_processed.shape[0], 5)
            self.assertEqual(x_processed.shape[1], 1)  # Should be 1, as there's only one non-zero row per batch
            result_processed = self.mlp(x_processed)

        self.assertEqual(x_processed.shape, result_processed.shape)
        self.assertEqual(x.shape, torch.Size([5, 1, 128]))
        self.assertFalse(torch.all(x[0] == 0))
        self.assertTrue(torch.all(x[1] == 0))
        self.assertFalse(torch.all(x[2] == 0))
        self.assertTrue(torch.all(x[3] == 0))
        self.assertFalse(torch.all(x[4] == 0))

    def test_all_non_zero_input(self):
        x = torch.rand((5, 1, 128))

        with NonZeroRowProcessor(x) as x_processed:
            self.assertEqual(x_processed.shape[0], 5)
            result = self.mlp(x_processed)

        self.assertEqual(x.shape, torch.Size([5, 1, 128]))
        for i in range(x.shape[0]):
            self.assertFalse(torch.all(x[i] == 0))

    def test_multi_batch_with_different_zero_rows(self):
        x = torch.tensor([
            [[0.0] * 128,
             [0.0] * 128,
             [4.0, 5.0, 6.0] + [0.0] * 125,
             [7.0, 8.0, 9.0] + [0.0] * 125],
            [[10.0, 11.0, 12.0] + [0.0] * 125,
             [13.0, 14.0, 15.0] + [0.0] * 125,
             [16.0, 17.0, 18.0] + [0.0] * 125,
             [0.0] * 128]
        ])

        original_shape = x.shape
        x_original = x.clone()

        with NonZeroRowProcessor(x) as x_processed:
            self.assertEqual(x_processed.shape[0], 2)
            self.assertEqual(x_processed.shape[1], 3)
            result_processed = self.mlp(x_processed)

        self.assertEqual(x_processed.shape, result_processed.shape)
        self.assertEqual(x.shape, original_shape)
        
        # Check that the non-zero rows are preserved in their original order
        self.assertTrue(torch.all(x_processed[0, 0] == x_original[0, 2]))
        self.assertTrue(torch.all(x_processed[0, 1] == x_original[0, 3]))
        self.assertTrue(torch.all(x_processed[0, 2] == 0))
        
        self.assertTrue(torch.all(x_processed[1, 0] == x_original[1, 0]))
        self.assertTrue(torch.all(x_processed[1, 1] == x_original[1, 1]))
        self.assertTrue(torch.all(x_processed[1, 2] == x_original[1, 2]))

        # Check that the original tensor is correctly reconstructed
        self.assertTrue(torch.all(x[0, 0] == 0))
        self.assertTrue(torch.all(x[0, 1] == 0))
        self.assertTrue(torch.all(x[0, 2] == x_original[0, 2]))
        self.assertTrue(torch.all(x[0, 3] == x_original[0, 3]))
        
        self.assertTrue(torch.all(x[1, 0] == x_original[1, 0]))
        self.assertTrue(torch.all(x[1, 1] == x_original[1, 1]))
        self.assertTrue(torch.all(x[1, 2] == x_original[1, 2]))
        self.assertTrue(torch.all(x[1, 3] == 0))

    def test_mixed_batch_types(self):
        x = torch.tensor([
            [[0.0] * 128,
             [0.0] * 128,
             [0.0] * 128],  # All-zero batch
            [[1.0, 2.0, 3.0] + [0.0] * 125,
             [0.0] * 128,
             [4.0, 5.0, 6.0] + [0.0] * 125],  # Partially zero batch
            [[7.0, 8.0, 9.0] + [0.0] * 125,
             [10.0, 11.0, 12.0] + [0.0] * 125,
             [13.0, 14.0, 15.0] + [0.0] * 125],  # All non-zero batch
        ])

        original_shape = x.shape
        x_original = x.clone()

        with NonZeroRowProcessor(x) as x_processed:
            self.assertEqual(x_processed.shape[0], 3)
            self.assertEqual(x_processed.shape[1], 3)  # Max non-zero rows across all batches
            result_processed = self.mlp(x_processed)

        self.assertEqual(x_processed.shape, result_processed.shape)
        self.assertEqual(x.shape, original_shape)

        # Check all-zero batch
        self.assertTrue(torch.all(x[0] == 0))

        # Check partially zero batch
        self.assertTrue(torch.all(x[1, 0] == x_original[1, 0]))
        self.assertTrue(torch.all(x[1, 1] == 0))
        self.assertTrue(torch.all(x[1, 2] == x_original[1, 2]))

        # Check all non-zero batch
        self.assertTrue(torch.all(x[2, 0] == x_original[2, 0]))
        self.assertTrue(torch.all(x[2, 1] == x_original[2, 1]))
        self.assertTrue(torch.all(x[2, 2] == x_original[2, 2]))

        # Check processed tensor
        self.assertTrue(torch.all(x_processed[0] == 0))
        self.assertTrue(torch.all(x_processed[1, 0] == x_original[1, 0]))
        self.assertTrue(torch.all(x_processed[1, 1] == x_original[1, 2]))
        self.assertTrue(torch.all(x_processed[1, 2] == 0))
        self.assertTrue(torch.all(x_processed[2, 0] == x_original[2, 0]))
        self.assertTrue(torch.all(x_processed[2, 1] == x_original[2, 1]))
        self.assertTrue(torch.all(x_processed[2, 2] == x_original[2, 2]))

if __name__ == '__main__':
    unittest.main()