__all__ = [
    'download_and_load_file',
    'format_input',
    'custom_collate_draft_1'
]

import json
import os.path
import urllib.request

import torch


def download_and_load_file(file_path: str, url: str):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w') as file:
            file.write(text_data)

    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()

    with open(file_path, 'r') as file:
        data = json.load(file)

    return data


def format_input(entry: dict) -> str:
    """
    Generates an Alpaca prompt formatted text string based on the provided dictionary input containing
    instruction and optional input values. The function creates a specific structure
    with headers and content to be used in a variety of contexts where instructional
    guidance and input data are required.

    :param entry: Dictionary containing the keys 'instruction' and 'input'. The 'instruction'
                  key holds a string that describes a task, while the 'input' key can either
                  contain additional related information as a string or be empty.
    :type entry: dict

    :return: The generated string that combines the formatted instruction and input text
             based on the content of the entry dictionary.
    :rtype: str
    """

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction: \n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input\n{entry['input']}" if entry['input'] else ""
    )

    return instruction_text + input_text


def custom_collate_draft_1(batch, pad_token_id=50256, device='cpu'):
    batch_max_length = max(len(item) for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded)
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)

    return inputs_tensor

