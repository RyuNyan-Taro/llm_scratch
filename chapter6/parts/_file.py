__all__ = ['download_and_unzip_spam_data', 'create_balanced_dataset']

import urllib.request
import zipfile
import os
from pathlib import Path

import pandas as pd


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f'{data_file_path} already exists. Skipping download and extraction.')
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as file:
            file.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / 'SMSSpamCollection'
    os.rename(original_file_path, data_file_path)
    print(f'File downloaded and extracted to {data_file_path}.')


def create_balanced_dataset(df: pd.DataFrame):
    num_spam = df[df['Label'] == 'spam'].shape[0]
    ham_subset = df[df['Label'] == 'ham'].sample(num_spam, random_state=123)

    balanced_df = pd.concat([ham_subset, df[df['Label'] == 'spam']])

    return balanced_df
