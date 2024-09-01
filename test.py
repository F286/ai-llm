# To run tests, in Terminal type: python test.py

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
        self.config = GPTConfig(
            n_layer=4,
            n_head=4,
            n_embd=128,
            block_size=1024,
            bias=False,
            vocab_size=50257,  # GPT-2 vocab_size
            dropout=0.0
        )

    def test_verify_sentence_end_tokens(self):
        tokens = ['.', '?', '!']
        encoded = verify_sentence_end_tokens(self.tokenizer, tokens)
        self.assertEqual(len(encoded), 3)
        for token in encoded:
            self.assertIsInstance(token, int)

    def test_sentence_end_processor(self):
        processor = SentenceEndProcessor(self.config.vocab_size)
        self.assertEqual(len(processor.sentence_end_ids), 3)
        for token, token_id in zip(['.', '?', '!'], processor.sentence_end_ids):
            self.assertEqual(token_id, self.tokenizer.encode(token)[0])

    def test_invalid_token(self):
        tokens = ['.', '?', '!', 'invalid_token']
        encoded = verify_sentence_end_tokens(self.tokenizer, tokens)
        self.assertEqual(len(encoded), 4)
        # Check that the first three tokens are encoded as expected
        self.assertEqual(encoded[:3], [13, 30, 0])
        # Check that 'invalid_token' is encoded differently
        self.assertNotIn(encoded[3], [13, 30, 0])
        print(f"'invalid_token' was encoded as: {encoded[3]}")

    def test_gpt_sentence_end_processor(self):
        model = GPT(self.config)
        self.assertEqual(len(model.sentence_end_processor.sentence_end_ids), 3)
        for token, token_id in zip(['.', '?', '!'], model.sentence_end_processor.sentence_end_ids):
            self.assertEqual(token_id, self.tokenizer.encode(token)[0])

    def test_forward_pass(self):
        model = GPT(self.config)
        x = torch.randint(0, self.config.vocab_size, (1, self.config.block_size))
        logits, _ = model(x)
        self.assertEqual(logits.shape, (1, self.config.block_size, self.config.vocab_size))

    def test_sentence_end_mask(self):
        processor = SentenceEndProcessor(self.config.vocab_size)
        idx = torch.tensor([1, 2, 3, processor.sentence_end_ids[0], 5, processor.sentence_end_ids[1]])
        mask = processor.create_sentence_end_mask(idx)
        expected_mask = torch.tensor([False, False, False, True, False, True])
        self.assertTrue(torch.all(mask == expected_mask))

if __name__ == '__main__':
    unittest.main()