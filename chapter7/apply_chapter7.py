from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import parts


def main():

    # _apply_get_dataset()

    # _apply_custom_collate()

    _apply_custom_dataloader()


def _get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def _apply_get_dataset():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = parts.download_and_load_file(file_path, url)

    print('number of entries:', len(data))
    print(data[:2])

    print('Example entry:\n', data[50])
    print('Another example entry:\n', data[999], '\n')

    for _i in [50, 999]:
        print('\nid:', _i)
        model_input = parts.format_input(data[_i])
        desired_response = f"\n\n### Response:\n{data[_i]['output']}"

        print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion+test_portion]
    val_data = data[train_portion+test_portion:]

    print('train data:', len(train_data))
    print('test data:', len(test_data))
    print('val data:', len(val_data))

    return train_data, test_data, val_data


def _apply_custom_collate():
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    inputs_4 = []

    batch = (
        inputs_1,
        inputs_2,
        inputs_3,
        inputs_4
    )

    print('\napply custom collate 1')
    print(parts.custom_collate_draft_1(batch))

    print('\napply custom collate 2')
    print(parts.custom_collate_draft_2(batch))

    print('\napply custom collate fn')
    print(parts.custom_collate_fn(batch))


def _apply_custom_dataloader():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'device: {device}')

    customized_collate_fn = partial(
        parts.custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    train_data, test_data, val_data = _apply_get_dataset()
    tokenizer = _get_tokenizer()

    num_loaders = 0
    batch_size = 8
    torch.manual_seed(123)

    train_dataset = parts.InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_loaders
    )

    val_dataset = parts.InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_loaders
    )

    test_dataset = parts.InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_loaders
    )

    print('train loader:')
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)


if __name__ == '__main__':
    main()
