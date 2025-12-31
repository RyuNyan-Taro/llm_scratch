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

    # print('\napply dummy transformer')
    # _apply_dummy_transformer()

    # print('\napply normalization')
    # _apply_normalization()

    # print('\n apply gelu')
    # _apply_gelu()

    # print('\n apply shortcut')
    # _apply_shortcut()

    # print('apply transformer')
    # _apply_transformer()

    print('apply gpt model')
    _apply_gpt_model()


def _get_batch():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)

    return batch


def _apply_dummy_transformer():
    batch = _get_batch()
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

    # for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'ReLU']), 1):
    #
    #     plt.subplot(1, 2, i)
    #     plt.plot(x, y)
    #     plt.title(f"{label} activation function")
    #     plt.xlabel('x')
    #     plt.ylabel(f'{label}(x)')
    #     plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    ffn = parts.FeedForward(GPT_CONFIG_124M)

    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)


def _apply_shortcut():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor(
        [[1., 0., -1.]]
    )

    torch.manual_seed(123)
    model_without_shortcut = parts.ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

    parts.print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = parts.ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print('\nwith shortcut:')
    parts.print_gradients(model_with_shortcut, sample_input)


def _apply_transformer():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = parts.TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print('input_shape -> output_shape:', x.shape, '->', output.shape)


def _apply_gpt_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    batch = _get_batch()
    batch = batch.to(device)

    torch.manual_seed(123)
    model = parts.GPTModel(GPT_CONFIG_124M)
    model.to(device)
    out = model(batch)

    print('Input batch:\n', batch)
    print('Output shape:\n', out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params:,}')

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f'Total size of parameters in MB: {total_size_mb:.2f}')


if __name__ == "__main__":
    main()
