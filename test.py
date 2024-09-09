import unittest
import torch
import tiktoken
from model import GPT, GPTConfig, SentenceEndProcessor

def verify_sentence_end_tokens(tokenizer, sentence_end_tokens):
    encoded_tokens = [tokenizer.encode(token) for token in sentence_end_tokens]
    for token, encoded in zip(sentence_end_tokens, encoded_tokens):
        if len(encoded) == 1:
            print(f"Token '{token}' is correctly encoded as: {encoded[0]}")
        else:
            print(f"Token '{token}' is encoded as multiple tokens: {encoded}")
    return [e[0] for e in encoded_tokens]  # Return the first token ID for each encoding

class TestSentenceEndTokens(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.default_sentence_end_tokens = ['.', '?', '!', '\n']
        self.config = GPTConfig(
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=1024,
            bias=False,
            vocab_size=50257,  # GPT-2 vocab_size
            dropout=0.0,
            sentence_end_tokens=self.default_sentence_end_tokens
        )

    def test_verify_sentence_end_tokens(self):
        encoded = verify_sentence_end_tokens(self.tokenizer, self.default_sentence_end_tokens)
        self.assertEqual(len(encoded), len(self.default_sentence_end_tokens))
        for token in encoded:
            self.assertIsInstance(token, int)

    def test_sentence_end_processor(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        self.assertEqual(len(processor.sentence_end_ids), len(self.default_sentence_end_tokens))
        for token, token_id in zip(self.default_sentence_end_tokens, processor.sentence_end_ids):
            self.assertEqual(token_id, self.tokenizer.encode(token)[0])

    def test_custom_sentence_end_tokens(self):
        custom_tokens = ['.', '?', '!', '\n', ';']
        config = GPTConfig(
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=1024,
            bias=False,
            vocab_size=50257,
            dropout=0.0,
            sentence_end_tokens=custom_tokens
        )
        model = GPT(config)
        self.assertEqual(len(model.sentence_end_processor.sentence_end_ids), len(custom_tokens))
        for token, token_id in zip(custom_tokens, model.sentence_end_processor.sentence_end_ids):
            self.assertEqual(token_id, self.tokenizer.encode(token)[0])

    def test_gpt_sentence_end_processor(self):
        model = GPT(self.config)
        self.assertEqual(len(model.sentence_end_processor.sentence_end_ids), len(self.default_sentence_end_tokens))
        for token, token_id in zip(self.default_sentence_end_tokens, model.sentence_end_processor.sentence_end_ids):
            self.assertEqual(token_id, self.tokenizer.encode(token)[0])

    def test_sentence_end_mask_sparse(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        idx = torch.tensor([[1, 2, 3, processor.sentence_end_ids[0], 5, processor.sentence_end_ids[1]]])
        sparse_mask = processor.create_sentence_end_mask(idx)
        
        self.assertTrue(sparse_mask.is_sparse)
        self.assertEqual(sparse_mask.size(), idx.size())
        
        dense_mask = sparse_mask.to_dense()
        expected_mask = torch.tensor([[0, 0, 0, 1, 0, 1]], dtype=torch.float)
        self.assertTrue(torch.all(dense_mask == expected_mask))

    def test_sentence_end_mask_sparse_comprehensive(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        
        sample_text = "Hello world.\nHow are you? I'm fine! This is a test."
        encoded = torch.tensor([self.tokenizer.encode(sample_text)])
        sparse_mask = processor.create_sentence_end_mask(encoded)
        
        self.assertTrue(sparse_mask.is_sparse)
        self.assertEqual(sparse_mask.size(), encoded.size())
        
        dense_mask = sparse_mask.to_dense()
        expected_mask = torch.zeros_like(encoded, dtype=torch.float)
        for i, token_id in enumerate(encoded[0]):
            if token_id in processor.sentence_end_ids:
                expected_mask[0, i] = 1
        
        self.assertTrue(torch.all(dense_mask == expected_mask))

    def test_process_middle_layer_sparse(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        
        # Mock block that doubles its input
        mock_block = lambda x: x * 2
        
        # Create a sample input
        sample_text = "Hello world.\nHow are you? I'm fine! This is a test."
        encoded = torch.tensor([self.tokenizer.encode(sample_text)])
        x = torch.randn(encoded.size(0), encoded.size(1), self.config.n_embd)
        
        # Process the middle layer
        result = processor.process_middle_layer(x, encoded, mock_block)
        
        # Check that only sentence end tokens are processed
        for i, token_id in enumerate(encoded[0]):
            if token_id in processor.sentence_end_ids:
                self.assertTrue(torch.all(result[0, i] == x[0, i] * 2))
            else:
                self.assertTrue(torch.all(result[0, i] == x[0, i]))

if __name__ == '__main__':
    unittest.main()