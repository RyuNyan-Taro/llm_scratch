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

    # _apply_text_verification()

    _apply_loss()


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


def _apply_loss():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print('token_ids:', token_ids)

    print(f'targets batch1: {parts.token_ids_to_text(targets[0], tokenizer)}')
    print(f'output batch1: {parts.token_ids_to_text(token_ids[0].flatten(), tokenizer)}')


if __name__ == '__main__':
    main()
