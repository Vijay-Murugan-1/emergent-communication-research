import os
import hydra
from omegaconf import DictConfig, OmegaConf
from configs.schema import Config
from experiments.runner import ExperimentRunner

# We need to import these so that the decorators register the classes
import datasets
import communication
import agents
import envs

# Set Hydra to use structured configs if necessary, but here we just pass the yaml
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
