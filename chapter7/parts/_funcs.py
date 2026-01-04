__all__ = [
    'download_and_load_file',
    'format_input',
    'custom_collate_draft_1',
    'custom_collate_draft_2',
    'custom_collate_fn',
    'check_if_running',
    'query_model',
    'generate_model_scores'
]

import json
import os.path
import urllib.request

import psutil
import torch
from tqdm import tqdm


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


def custom_collate_draft_2(batch, pad_token_id=50256, device='cpu'):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device='cpu'):

    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    # Iterates batch; pads; masks targets; truncates if needed
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def check_if_running(process_name: str):
    running = False
    for proc in psutil.process_iter(['name']):
        if process_name in proc.info['name']:
            running = True
            break

    return running


def query_model(prompt, model: str = 'llama3', url='http://localhost:11434/api/chat'):
    data = {
        'model': model,
        'messages': [{"role": "user", "content": prompt}],
        'options': {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    payload = json.dumps(data).encode('utf-8')

    request = urllib.request.Request(
        url,
        data=payload,
        method='POST',
    )
    request.add_header('Content-Type', 'application/json')

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode('utf-8')
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json['message']['content']
            print(f"\rReceived response length: {len(response_data)}", end="", flush=True)

    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc='Scoring entries'):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model=model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f'could not convert score: {score} to int')
            continue

    return scores

