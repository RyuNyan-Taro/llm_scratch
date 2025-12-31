import matplotlib.pyplot as plt
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

    # print('\napply normalization')
    # _apply_normalization()

    print('\n apply gelu')
    _apply_gelu()


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

    torch.set_printoptions(sci_mode=False)
    print('mean:', mean)
    print('var:', var)

    ln = parts.LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print('mean:', mean)
    print('var:', var)


def _apply_gelu():
    gelu, relu = parts.GELU(), nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'ReLU']), 1):

        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel('x')
        plt.ylabel(f'{label}(x)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
