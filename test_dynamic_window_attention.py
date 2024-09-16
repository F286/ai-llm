import torch
import torch.nn as nn
import unittest
from dynamic_window_attention import DynamicWindowAttention

class TestDynamicWindowAttentionStepByStep(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 8
        self.num_heads = 2
        self.model = DynamicWindowAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            delimiter_chars=[',', '.'], 
            normalize_v=True  # Enable V normalization
        )
        
        self.char_ids = torch.tensor([[ord('a'), ord(','), ord('b'), ord('c'), ord('.'), ord('d')]])

    def test_attention_output(self):
        # Set all weights to 1 for easier calculation
        with torch.no_grad():
            nn.init.ones_(self.model.qkv_proj.weight)
            nn.init.zeros_(self.model.qkv_proj.bias)

        # Set input tensor to all 1s
        x = torch.ones(1, 6, self.embed_dim)

        # Print intermediate steps
        print("\nInput x:")
        print(x)

        qkv = self.model.generate_qkv(x)
        print("\nQKV:")
        print(qkv)

        q, k, v = self.model.split_qkv(qkv)
        print("\nQ:")
        print(q)
        print("\nK:")
        print(k)
        print("\nV (normalized):")
        print(v)

        attention_mask = self.model.create_attention_mask(self.char_ids)
        print("\nAttention Mask:")
        print(attention_mask)

        attention_weights = self.model.calculate_attention_weights(q, k, attention_mask)
        print("\nAttention Weights:")
        print(attention_weights)

        output = self.model(x, self.char_ids)
        print("\nFinal Output:")
        print(output)

        # Calculate expected output
        expected_output = torch.ones(1, 6, self.embed_dim) * self.model.num_heads

        print("\nExpected Output:")
        print(expected_output)

        # Compare actual output with expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

        print("\nOutput matches expected values!")

class TestDynamicWindowAttention(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 768
        self.num_heads = 12
        self.model = DynamicWindowAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            delimiter_chars=[' ', '.', '?', '!', '\n']
        )
        
        self.batch_size = 64
        self.seq_length = 256
        self.vocab_size = 65

        self.char_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))

    def test_forward_pass(self):
        # Create input tensor
        x = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
        
        # Create character IDs tensor
        char_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))

        try:
            # Run forward pass
            output = self.model(x, char_ids)

            # Print shapes for debugging
            print(f"\nInput shape: {x.shape}")
            print(f"Character IDs shape: {char_ids.shape}")
            print(f"Output shape: {output.shape}")

            # Check if output shape matches input shape
            self.assertEqual(output.shape, x.shape)

            # Check if output is not all zeros or NaNs
            self.assertFalse(torch.all(output == 0))
            self.assertFalse(torch.isnan(output).any())

            print("Forward pass successful!")

        except Exception as e:
            print(f"Forward pass failed with error: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main()