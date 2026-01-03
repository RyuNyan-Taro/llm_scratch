from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import parts
from parts.chapter5parts import gpt_download, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
from parts.chapter4parts import GPTModel


def main():

    # _apply_get_dataset()

    # _apply_custom_collate()

    # _apply_custom_dataloader()

    _apply_read_learned_model()


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
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
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


def _apply_read_learned_model():
    """Reads learned model; generates response from validation data"""

    train_data, test_data, val_data = _apply_get_dataset()
    tokenizer = _get_tokenizer()

    BASE_CONFIG = {
        'vocab_size': 50257,
        'context_length': 1024,
        'drop_rate': 0.0,
        'qkv_bias': True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, 'n_layers': 12, 'n_heads': 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, 'n_layers': 24, 'n_heads': 16},
        "gpt2-large (774M)": {"emb_dim": 1280, 'n_layers': 36, 'n_heads': 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, 'n_layers': 48, 'n_heads': 25}
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(' ')[-1].lstrip('(').rstrip(')')

    settings, params = gpt_download.download_and_load_gpt2(
        model_size=model_size, models_dir='gpt2'
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)

    input_text = parts.format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model, idx=text_to_token_ids(input_text, tokenizer=tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG['context_length'],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text):].strip()
    # response_text: object = generated_text.strip()
    print('\nresponse text:\n')
    print(response_text)


if __name__ == '__main__':
    main()
