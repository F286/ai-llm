import torch
import matplotlib.pyplot as plt
from model import SentenceEndProcessor, GPTConfig
import tiktoken

def visualize_attention_mask(sentence):
    # Initialize tokenizer and SentenceEndProcessor
    tokenizer = tiktoken.get_encoding("gpt2")
    config = GPTConfig()
    processor = SentenceEndProcessor(config.vocab_size)

    # Tokenize the sentence
    tokens = tokenizer.encode(sentence)
    idx = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

    # Create the sentence end mask
    mask = processor.create_sentence_end_mask(idx.squeeze())

    # Create the causal attention mask
    seq_length = len(tokens)
    causal_mask = torch.tril(torch.ones(seq_length, seq_length))

    # Visualize the masks
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(causal_mask, cmap='binary')
    ax1.set_title('Causal Attention Mask (Used in All Layers)')
    ax1.set_xlabel('Key')
    ax1.set_ylabel('Query')

    ax2.imshow(mask.unsqueeze(0).repeat(seq_length, 1), cmap='binary')
    ax2.set_title('Sentence End Mask (For Special Processing)')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Is Sentence End')

    # Add token labels
    token_labels = [tokenizer.decode([t]) for t in tokens]
    for ax in (ax1, ax2):
        ax.set_xticks(range(seq_length))
        ax.set_xticklabels(token_labels, rotation=90)
        ax.set_yticks(range(seq_length))
        ax.set_yticklabels(token_labels)

    plt.tight_layout()
    plt.show()

    # Print the sentence end tokens
    sentence_end_tokens = [t for t, m in zip(token_labels, mask) if m]
    print("Sentence end tokens:", sentence_end_tokens)
    print("Note: Sentence end tokens receive special processing in middle layers, but don't change the attention mask.")

# Example usage
sentence1 = "Hello, world! How are you? I'm fine."
sentence2 = "This is a test. It has multiple sentences. Does it work?"

print("Visualizing attention masks for sentence 1:")
visualize_attention_mask(sentence1)

print("\nVisualizing attention masks for sentence 2:")
visualize_attention_mask(sentence2)