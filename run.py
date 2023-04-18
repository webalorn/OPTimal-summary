import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from opts.dataset import load_dataset

@hydra.main(version_base=None, config_path="conf", config_name="run")
def run_main(config : DictConfig) -> None:
    print('Configuration:', OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)

    # TODO: maxrows=100 only for tests
    train_data, test_data, valid_data = load_dataset(config, maxrows=100)

if __name__ == "__main__":
    run_main()