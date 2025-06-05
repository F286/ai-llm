import unittest
import torch

from genesis_model import ConceptMatcher, GenesisAttention, GPT
from genesis_model import GPTConfig


class TestConceptMatcher(unittest.TestCase):
    def test_output_shape(self):
        matcher = ConceptMatcher(embed_dim=32, num_heads=2, num_concepts=3, concept_dim=4)
        x = torch.randn(1, 5, 32)
        scores = matcher(x)
        self.assertEqual(scores.shape, (1, 2, 5, 5, 3))


class TestGenesisAttention(unittest.TestCase):
    def test_forward(self):
        cfg = GPTConfig(n_embd=32, n_head=2, n_layer=1, block_size=8, vocab_size=100)
        attn = GenesisAttention(cfg)
        x = torch.randn(1, 4, 32)
        out, loss = attn(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(loss.dim() == 0)


class TestConceptMatcherGradient(unittest.TestCase):
    def test_backward(self):
        matcher = ConceptMatcher(embed_dim=16, num_heads=2, num_concepts=4, concept_dim=2)
        x = torch.randn(3, 5, 16, requires_grad=True)
        scores = matcher(x)
        scores.sum().backward()
        self.assertIsNotNone(matcher.q_proj.weight.grad)
        self.assertIsNotNone(matcher.k_proj.weight.grad)
        self.assertIsNotNone(x.grad)


class TestGenesisAttentionTraining(unittest.TestCase):
    def test_backward(self):
        cfg = GPTConfig(n_embd=16, n_head=4, n_layer=1, block_size=4, vocab_size=50)
        attn = GenesisAttention(cfg)
        x = torch.randn(2, 4, 16, requires_grad=True)
        out, extra_loss = attn(x)
        loss = out.mean() + extra_loss
        loss.backward()
        for param in attn.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class TestGenesisGPTTrainable(unittest.TestCase):
    def test_train_step(self):
        cfg = GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=4, vocab_size=32)
        model = GPT(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
        targets = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
        optimizer.zero_grad()
        logits, loss = model(idx, targets)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        optimizer.step()
        logits_after, loss_after = model(idx, targets)
        self.assertFalse(torch.isnan(loss_after))


if __name__ == "__main__":
    unittest.main()
