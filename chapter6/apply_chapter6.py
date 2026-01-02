from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import parts
from parts.chapter5parts import gpt_download
from parts.chapter5parts import load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from parts.chapter5parts.chapter4parts import GPTModel, generate_text_simple


def main():

    # _apply_file_download()

    # _apply_dataset()

    _apply_load_gpt()


def _get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def _get_base_config(choose_model: str = "gpt2-small (124M)"):
    INPUT_PROMPT = 'Every effort moves'

    BASE_CONFIG = {
        'vocab_size': 50257,
        'context_length': 1024,
        'drop_rate': 0.0,
        'qkv_bias': True,
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, 'n_layers': 12, 'n_heads': 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, 'n_layers': 24, 'n_heads': 16},
        "gpt2-large (774M)": {"emb_dim": 1280, 'n_layers': 36, 'n_heads': 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, 'n_layers': 48, 'n_heads': 25}
    }

    BASE_CONFIG.update(model_configs[choose_model])

    return BASE_CONFIG


def _apply_file_download():
    url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
    zip_path = 'sms_spam_collection.zip'
    extracted_path = 'sms_spam_collection'
    data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv'

    parts.download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])

    print(df.info())
    print(df)

    # df['TextSize'] = df['Text'].str.len()
    # df['WordCount'] = df['Text'].str.split().str.len()
    #
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #
    # for _label, _group in df.groupby('Label'):
    #     axes[0].hist(_group['TextSize'], label=_label, range=(0, 600), bins=50, alpha=0.5)
    #     axes[1].hist(_group['WordCount'], label=_label, range=(0, 100), bins=25, alpha=0.5)
    # axes[0].legend()
    # axes[0].set_title('Text Size Distribution by Label')
    # axes[1].set_title('Word Count Distribution by Label')
    #
    # plt.show()

    print(df['Label'].value_counts())

    balanced_df = parts.create_balanced_dataset(df)
    print(balanced_df['Label'].value_counts())

    balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})

    train_df, validation_df, test_df = parts.random_split(balanced_df, 0.7, 0.1)

    for _file_name, _df in zip(['train', 'validation', 'test'], [train_df, validation_df, test_df]):
        _df.to_csv(f'{_file_name}.csv', index=None)


def _apply_dataset():
    tokenizer = _get_tokenizer()

    train_dataset = parts.SpamDataset(
        csv_file='train.csv', max_length=None, tokenizer=tokenizer
    )

    print(train_dataset.max_length)

    val_dataset = parts.SpamDataset(
        csv_file='validation.csv', max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    test_dataset = parts.SpamDataset(
        csv_file='test.csv', max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    for input_batch, target_batch in train_loader:
        pass
    print(input_batch.shape, target_batch.shape)

    for _loader in [train_loader, val_loader, test_loader]:
        print(len(_loader))


def _apply_load_gpt():
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = _get_base_config(CHOOSE_MODEL)
    tokenizer = _get_tokenizer()

    model_size = CHOOSE_MODEL.split(' ')[-1].lstrip('(').rstrip(')')
    settings, params = gpt_download.download_and_load_gpt2(model_size=model_size, models_dir='gpt2')

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(model, text_to_token_ids(text_1, tokenizer), max_new_tokens=15, context_size=BASE_CONFIG['context_length'])
    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(model, text_to_token_ids(text_2, tokenizer), max_new_tokens=23, context_size=BASE_CONFIG['context_length'])
    print(token_ids_to_text(token_ids, tokenizer))

    print(model)


if __name__ == "__main__":
    main()
