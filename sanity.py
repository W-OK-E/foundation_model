import os
import hydra
import tqdm
import torch
import warnings

from shutil import copyfile
from omegaconf import OmegaConf
from os.path import isfile, join
from hydra.utils import instantiate

warnings.filterwarnings("ignore")


# Registering the "eval" resolver allows for advanced config, i.e. basically the values can be dynamic now
# interpolation with arithmetic operations in hydra:
OmegaConf.register_new_resolver("eval", eval)



def project_init(cfg):
    print("Working directory set to {}".format(os.getcwd()))
    # Create a per-run subdirectory inside the configured checkpoint dir so
    # multiple runs on the same dataset+model don't overwrite each other.
    base_dir = cfg.checkpoints.dirpath
    directory = base_dir
    os.makedirs(directory, exist_ok=True)
    # copy the active hydra config for reproducibility
    try:
        copyfile(".hydra/config.yaml", join(directory, "config.yaml"))
    except Exception:
        # best-effort: don't crash if hydra metadata isn't present
        pass



def init_datamodule(cfg):
    datamodule = instantiate(cfg.datamodule)
    return datamodule

def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    datamodule = init_datamodule(cfg)
    if(cfg.mode != "test"):
        project_init(cfg)
    return datamodule

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    datamodule = hydra_boilerplate(cfg)
    assert(cfg.mode == "train","Sanity Checks are only supposed to happen before training")
    num_classes = cfg.dataset.num_classes
    class_check = True
    all_zeros = []
    try:
        for batch in tqdm.tqdm(datamodule.train_dataloader()):
            unique_values = len(torch.unique(batch[1]))
            all_zeros.append(unique_values == 1)
            if(unique_values > (num_classes+1)):
                class_check = False
            print(torch.unique(batch[1]))
        if(not class_check):
            print("Max pixel exceeds number of classes")
        if(all(all_zeros)):
            print("All the masks are a bust")
    except Exception as exc:
        print("="*70)
        print("Errors during sanity checks")
        print("="*70)
        print(exc)

if __name__ == "__main__":
    main()
