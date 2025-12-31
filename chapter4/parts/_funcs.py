__all__ = ['print_gradients', 'generate_text_simple']

import torch
from torch import nn


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generates text by repeatedly sampling from model outputs"""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        prob = nn.functional.softmax(logits, dim=-1)

        idx_next = torch.argmax(prob, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx

