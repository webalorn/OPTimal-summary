import random

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from opts.dataset import load_dataset, load_dataset_df

@hydra.main(version_base=None, config_path="conf", config_name="run")
def run_main(config : DictConfig) -> None:
    print('Configuration:', OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)

    # maxrows trumps config, otherwise both are working
    train_data, test_data, valid_data = load_dataset(config, maxrows=100)
    
    # Just for verification
    print(train_data.shape)
    print(test_data.shape)
    print(valid_data.shape)

if __name__ == "__main__":
    run_main()