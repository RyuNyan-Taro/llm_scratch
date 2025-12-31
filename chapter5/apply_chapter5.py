import tiktoken
import torch
from chapter4parts import GPTModel, generate_text_simple

import parts

GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 256,
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

    start_context = 'Every effort moves you'
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model,
        parts.text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M['context_length']
    )

    print('Output text:\n', parts.token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    main()
