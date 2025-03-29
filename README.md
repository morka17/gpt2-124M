# GPT-2 Training Implementation

A PyTorch implementation of GPT-2 training from scratch, featuring modern optimizations and best practices.

## Features

- Pure PyTorch implementation of GPT-2 architecture
- Flash Attention support for improved performance
- Mixed precision training with bfloat16
- Weight sharing between input embeddings and output layer
- Configurable model size and architecture
- Support for loading pretrained weights from HuggingFace
- Efficient data loading with tiktoken tokenization

## Requirements

```bash
pip install torch tiktoken transformers
```

## Model Architecture

The implementation includes:
- Multi-head self-attention with causal masking
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Weight sharing between input and output embeddings

## Usage

### Training

```python
from train_gpt2 import GPT, GPTConfig, DataLoaderLite

# Initialize model and data loader
model = GPT(GPTConfig())
train_loader = DataLoaderLite(B=4, T=32)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
```

### Loading Pretrained Weights

```python
model = GPT.from_pretrained('gpt2')  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
```

## Configuration

The model can be configured through `GPTConfig`:

```python
@dataclass 
class GPTConfig: 
    block_size: int = 1024      # Maximum sequence length
    vocab_size: int = 50257     # Vocabulary size (GPT-2 default)
    n_layer: int = 6           # Number of transformer layers
    n_head: int = 6            # Number of attention heads
    n_embd: int = 384          # Embedding dimension
```

## Performance Optimizations

- Flash Attention for efficient attention computation
- Mixed precision training with bfloat16
- High-precision matrix multiplication
- Efficient data loading with tiktoken

## Data Requirements

The training script expects an `input.txt` file in the same directory containing the training text. The text will be tokenized using the GPT-2 tokenizer (tiktoken).

## License

MIT License

## Acknowledgments

- Implementation based on the original GPT-2 paper and architecture
- Flash Attention implementation from PyTorch
- Tokenization using tiktoken 