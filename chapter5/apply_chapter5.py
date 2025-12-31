import torch
from chapter4parts import GPTModel

GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def main():

    _apply_text_verification()


def _apply_text_verification():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()


if __name__ == '__main__':
    main()
