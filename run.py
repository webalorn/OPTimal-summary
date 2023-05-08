import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from opts.utils import print_trainable_parameters
from opts.dataset import load_dataset, preprocess_data, get_data_loaders
from opts.model import load_tokenizer, OPTSModel
from opts.training import train_opts

@hydra.main(version_base=None, config_path="conf", config_name="run")
def run_main(config : DictConfig) -> None:
    print('Configuration')
    print('-------------')
    print(OmegaConf.to_yaml(config), end='')
    print('-------------')
    print()

    random.seed(config.seed)
    np.random.seed(config.seed)

    # Loading tokenizer
    tokenizer = load_tokenizer(config)

    # train_data, test_data, valid_data = load_dataset(config)
    dataset = load_dataset(config)
    dataset = preprocess_data(dataset, config, tokenizer)
    print(dataset)
    train_loader, test_loader, val_loader = get_data_loaders(dataset, config, tokenizer)

    # Loading model
    model = OPTSModel(config)
    # model = OPTSModel(config, load_from='opts_model')
    model.print_trainable_parameters()

    # Training
    model.finetune_model(train_loader, val_loader)
    model.save()


if __name__ == "__main__":
    run_main()