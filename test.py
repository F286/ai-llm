# To run tests, in Powershell type: python test.py
# To run evaluation, in Powershell type: python sample.py --out_dir=out-shakespeare-char --device=cpu

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

    def test_sentence_end_mask(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        idx = torch.tensor([1, 2, 3, processor.sentence_end_ids[0], 5, processor.sentence_end_ids[1]])
        mask = processor.create_sentence_end_mask(idx)
        expected_mask = torch.tensor([False, False, False, True, False, True])
        self.assertTrue(torch.all(mask == expected_mask))

    def test_sentence_end_mask_comprehensive(self):
        processor = SentenceEndProcessor(self.config.vocab_size, self.default_sentence_end_tokens)
        
        # Create a sample input with a mix of sentence-end and non-sentence-end tokens
        sample_text = "Hello world.\nHow are you? I'm fine! This is a test."
        encoded = self.tokenizer.encode(sample_text)
        idx = torch.tensor(encoded)

        # Get the mask
        mask = processor.create_sentence_end_mask(idx)

        # Verify the mask
        expected_mask = torch.zeros_like(idx, dtype=torch.bool)
        for i, token_id in enumerate(encoded):
            if token_id in processor.sentence_end_ids:
                expected_mask[i] = True

        self.assertTrue(torch.all(mask == expected_mask))

        # Print out the results for visual inspection
        print("\nSentence-end masking test:")
        print(f"Input text: {sample_text}")
        print(f"Encoded: {encoded}")
        print(f"Mask: {mask}")
        print("Tokens kept:")
        for token, is_kept in zip(sample_text.split(), mask.tolist()):
            if is_kept:
                print(f"  {token}")

        # Verify that only sentence-end tokens are masked
        masked_tokens = [self.tokenizer.decode([token]) for token, keep in zip(encoded, mask) if keep]
        self.assertEqual(set(masked_tokens), set(self.default_sentence_end_tokens))

    def test_sentence_end_tokens_required(self):
        with self.assertRaises(AssertionError):
            SentenceEndProcessor(self.config.vocab_size, None)

        with self.assertRaises(AssertionError):
            GPTConfig(
                n_layer=4,
                n_head=4,
                n_embd=128,
                block_size=1024,
                bias=False,
                vocab_size=50257,
                dropout=0.0,
                sentence_end_tokens=None
            )

        with self.assertRaises(AssertionError):
            GPT(GPTConfig(
                n_layer=4,
                n_head=4,
                n_embd=128,
                block_size=1024,
                bias=False,
                vocab_size=50257,
                dropout=0.0,
                sentence_end_tokens=None
            ))

if __name__ == '__main__':
    unittest.main()