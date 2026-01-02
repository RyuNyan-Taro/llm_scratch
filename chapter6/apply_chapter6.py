from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import parts


def main():

    _apply_file_download()


def _apply_file_download():
    url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
    zip_path = 'sms_spam_collection.zip'
    extracted_path = 'sms_spam_collection'
    data_file_path = Path(extracted_path) / 'SMSSpamCollection.tsv'

    parts.download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text'])

    print(df.info())
    print(df)

    df['TextSize'] = df['Text'].str.len()
    df['WordCount'] = df['Text'].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for _label, _group in df.groupby('Label'):
        axes[0].hist(_group['TextSize'], label=_label, range=(0, 600), bins=50, alpha=0.5)
        axes[1].hist(_group['WordCount'], label=_label, range=(0, 100), bins=25, alpha=0.5)
    axes[0].legend()
    axes[0].set_title('Text Size Distribution by Label')
    axes[1].set_title('Word Count Distribution by Label')

    plt.show()


if __name__ == "__main__":
    main()
