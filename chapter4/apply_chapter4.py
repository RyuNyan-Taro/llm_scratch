import tiktoken
import torch
import torch.nn as nn

import parts


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
    print('start main')

    # print('\napply dummy transformer')
    # _apply_dummy_transformer()

    print('\napply normalization')
    _apply_normalization()


def _apply_dummy_transformer():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = parts.DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print('output shape:', logits.shape)
    print(logits)


def _apply_normalization():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    print('batch_example:', batch_example, '\n')
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print('mean:', mean)
    print('var:', var)

    out_norm = (out - mean) / torch.sqrt(var)
    print('out_norm:', out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print('mean:', mean)
    print('var:', var)


if __name__ == "__main__":
    main()
