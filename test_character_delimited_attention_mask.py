import torch
import torch.nn as nn
import unittest
from character_delimited_attention_mask import CharacterDelimitedAttentionMask

class TestCharacterDelimitedAttentionMask(unittest.TestCase):
    def setUp(self):
        self.attention_mask = CharacterDelimitedAttentionMask(delimiter_chars=[',', '.'])
        self.char_ids = torch.tensor([[ord('a'), ord(','), ord('b'), ord('c'), ord('.'), ord('d')]])

    def test_create_delimiter_tensor(self):
        delimiter_tensor = self.attention_mask._create_delimiter_tensor(self.char_ids)
        expected = torch.tensor([[0., 1., 0., 0., 1., 0.]])
        self.assertTrue(torch.all(delimiter_tensor == expected))

    def test_shift_delimiter_tensor(self):
        delimiter_tensor = self.attention_mask._create_delimiter_tensor(self.char_ids)
        shifted_delimiter_tensor = self.attention_mask._shift_delimiter_tensor(delimiter_tensor)
        expected = torch.tensor([[0., 0., 1., 0., 0., 1.]])
        self.assertTrue(torch.all(shifted_delimiter_tensor == expected))

    def test_create_region_tensor(self):
        delimiter_tensor = self.attention_mask._create_delimiter_tensor(self.char_ids)
        region_tensor = self.attention_mask._create_region_tensor(delimiter_tensor)
        expected = torch.tensor([[0., 0., 1., 1., 1., 2.]])
        self.assertTrue(torch.all(region_tensor == expected))

    def test_create_region_mask(self):
        delimiter_tensor = self.attention_mask._create_delimiter_tensor(self.char_ids)
        region_tensor = self.attention_mask._create_region_tensor(delimiter_tensor)
        region_mask = self.attention_mask._create_region_mask(region_tensor)
        expected = torch.tensor([[[1, 1, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1]]], dtype=torch.bool)
        
        print("region_tensor")
        print(region_tensor)
        print("region_mask")
        print(region_mask)
        self.assertTrue(torch.all(region_mask == expected))

    def test_create_causal_mask(self):
        causal_mask = self.attention_mask._create_causal_mask(6, self.char_ids.device)
        expected = torch.tril(torch.ones(6, 6, dtype=torch.bool, device=self.char_ids.device))
        self.assertTrue(torch.all(causal_mask == expected))

    def test_create_causal_delimiter_mask(self):
        mask = self.attention_mask.create_causal_delimiter_mask(self.char_ids)
        expected = torch.tensor([[[1, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 0, 0],
                                  [0, 1, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 1, 1]]], dtype=torch.bool)
        self.assertTrue(torch.all(mask == expected))

    def test_create_causal_delimiter_mask_with_batch(self):
        char_ids_batch = torch.tensor([
            [ord('a'), ord(','), ord('b'), ord('c'), ord('.'), ord('d')],
            [ord('x'), ord('y'), ord(','), ord('z'), ord('.'), ord('w')]
        ])
        
        mask = self.attention_mask.create_causal_delimiter_mask(char_ids_batch)
        
        expected = torch.tensor([
            [[1, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 1]],
            
            [[1, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1, 1]]
        ], dtype=torch.bool)
        
        self.assertTrue(torch.all(mask == expected))

if __name__ == '__main__':
    unittest.main()