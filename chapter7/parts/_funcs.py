__all__ = ['download_and_load_file', 'format_input']

import json
import os.path
import urllib.request


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


def format_input(entry: dict):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction: \n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input\n{entry['input']}" if entry['input'] else ""
    )

    return instruction_text + input_text
