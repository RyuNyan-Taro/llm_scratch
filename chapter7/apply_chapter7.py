import parts


def main():

    _apply_get_dataset()


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


if __name__ == '__main__':
    main()
