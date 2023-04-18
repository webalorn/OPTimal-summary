from pathlib import Path

import pandas as pd

REQUIRE_FILES = ['data/all_tests.txt', 'data/all_train.txt', 'data/all_val.txt', 'data/wikihowAll.csv']

def check_required_files(required_files):
    missings_files = [f for f in required_files if not Path(f).exists()]
    if missings_files:
        required_files = ', '.join(required_files)
        raise Exception(f"[ERROR] Missing files: ({required_files}) (Please check that you have run download.sh)")

def load_rows_for_split(split_name):
    with open(f'data/all_{split_name}.txt') as rows_file:
        rows = rows_file.readlines()
        rows = map(str.strip, rows)
        rows = filter(any, rows)
        return list(rows)

def load_dataset(config, maxrows=None):
    check_required_files(REQUIRE_FILES)
    data = pd.read_csv("data/wikihowAll.csv", nrows=maxrows)
    data.set_index('title')

    for split in ['train', 'test', 'val']:
        rows = load_rows_for_split(split)
        split_rows = data.loc(rows)

        max_rows_split = config.data[f'max_{split}']
        if max_rows_split is not None:
            split_rows = split_rows[:max_rows_split]
        return split_rows