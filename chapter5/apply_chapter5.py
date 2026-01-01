import time

import tiktoken
import torch
from matplotlib import pyplot as plt

from parts.chapter4parts import GPTModel, generate_text_simple

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

    # _apply_loss()

    # _apply_calculate_loss()

    # _apply_training_process()

    _apply_decoding_ideas()


def _get_loaders():
    file_path = 'the-verdict.txt'
    with open(file_path, 'r') as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    total_characters = len(text)
    total_tokens = len(tokenizer.encode(text))

    print('characters:', total_characters)
    print('tokens:', total_tokens)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]

    torch.manual_seed(123)

    train_loader = parts.create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = parts.create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


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

    probas_list = list()
    for _text_idx in range(2):
        target_probas = probas[_text_idx, [0, 1, 2], targets[_text_idx]]
        print(f'target probas batch {_text_idx+1}: {target_probas}')
        probas_list.append(target_probas)

    log_probas = torch.log(torch.cat(probas_list))
    print(f'log probas: {log_probas}')

    avg_log_probas = torch.mean(log_probas)
    print(f'avg log probas: {avg_log_probas}')

    neg_avg_log_probas = avg_log_probas * -1
    print(f'neg avg log probas: {neg_avg_log_probas}')

    print('logits shape:', logits.shape, '\n')
    print('targets shape:', targets.shape, '\n')

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print('logits_flat:', logits_flat, '\n')
    print('targets_flat:', targets_flat, '\n')

    loss = torch.nn.CrossEntropyLoss()(logits_flat, targets_flat)
    print(f'loss: {loss}')

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(f'loss: {loss}')


def _apply_calculate_loss():
    train_loader, val_loader = _get_loaders()

    print('train_loader:')
    for _x, _y in train_loader:
        print(_x.shape, _y.shape)
    print('\nval_loader:')
    for _x, _y in val_loader:
        print(_x.shape, _y.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}')

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    _time = time.time()
    with torch.no_grad():
        train_loss = parts.calc_loss_loader(train_loader, model, device)
        val_loss = parts.calc_loss_loader(val_loader, model, device)

    print(f'time: {time.time() - _time:.2f}s')

    print(f'train loss: {train_loss}')
    print(f'val loss: {val_loss}')


def _apply_training_process():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = 'cpu'
    print(f'device: {device}')

    train_loader, val_loader = _get_loaders()

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = parts.train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    parts.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


def _apply_decoding_ideas():

    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8
    }

    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas):
        torch.manual_seed(123)

        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))

        for i, freq in enumerate(sampled_ids):
            print(f'{freq} x {inverse_vocab[i]}')

    print_sampled_tokens(probas)

    temperatures = [1, 0.1, 5]
    scaled_probas = [parts.softmax_with_temperature(next_token_logits, T) for T in temperatures]

    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))

    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature= {T}')

    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()
