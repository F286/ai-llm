import torch
import torch.nn as nn
import unittest
from character_delimited_attention import CharacterDelimitedAttention

class TestCharacterDelimitedAttentionStepByStep(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 8
        self.num_heads = 2
        self.model = CharacterDelimitedAttention(
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

if __name__ == '__main__':
    unittest.main()