from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    main()
