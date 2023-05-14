from pathlib import Path
import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from datasets import load_from_disk

from opts.utils import print_gpu_utilization
from opts.dataset import load_dataset, preprocess_data, get_data_loaders
from opts.model import load_tokenizer, OPTSModel

@hydra.main(version_base=None, config_path="conf", config_name="run")
def run_main(config : DictConfig) -> None:
    print('Configuration')
    print('-------------')
    print(OmegaConf.to_yaml(config), end='')
    print('-------------')
    print()

    print_gpu_utilization()

    random.seed(config.seed)
    np.random.seed(config.seed)

    # Loading tokenizer
    tokenizer = load_tokenizer(config)

    processed_path = Path('data/processed')
    if Path('data/processed/').exists() and len(list(processed_path.iterdir())) != 0:
        print("Loading Dataset")
        dataset = load_from_disk('data/processed', keep_in_memory=True)
    else:
        # train_data, test_data, valid_data = load_dataset(config)
        dataset = load_dataset(config)
        dataset = preprocess_data(dataset, config, tokenizer)
        dataset.save_to_disk('data/processed')

    print(dataset)
    # dataset = dataset.remove_columns(column_names=['article', 'summary', 'text', 'prompt_ques', 'prompt_ans'])
    # print(dataset)
    train_loader, test_loader, val_loader = get_data_loaders(dataset, config, tokenizer)

    print_gpu_utilization()

    # Loading model
    if config.train:
        model = OPTSModel(config)
    else:
        model = OPTSModel(config, load_from='results/opts_model_iter4642')

    model.print_trainable_parameters()

    print_gpu_utilization()

    # Training
    model.finetune_model(train_loader, val_loader, tokenizer)
    model.save()


if __name__ == "__main__":
    run_main()
