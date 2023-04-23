from pathlib import Path

import pandas as pd

from .utils import min_with_none

REQUIRE_FILES = ['data/test.csv', 'data/train.csv', 'data/val.csv']

def check_required_files(required_files):
    missings_files = [f for f in required_files if not Path(f).exists()]
    if missings_files:
        missings_files = ', '.join(missings_files)
        raise Exception(f"[ERROR] Missing files: ({missings_files}) (Please check that you have run download.sh and preprocess_data.py)")
    
def load_dataset(config, maxrows=None):
    check_required_files(REQUIRE_FILES)
    
    data = {}
    for split in ['train', 'test', 'val']:
        max_rows_cfg = config.data[f'max_{split}']
        data[split] = pd.read_csv(f"data/{split}.csv", nrows=min_with_none([max_rows_cfg, maxrows]))

    return list(data.values())