import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.combined_loader import CombinedLoader

class SegDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            test_dataset, 
            global_batch_size=32, 
            num_workers=4,
            **kwargs
    ):
        super(SegDataModule, self).__init__()
        self.batch_size = global_batch_size
        self.num_workers = num_workers
        
        # Instantiate datasets from configs
        self.train_dataset = self._instantiate_dataset(train_dataset)
        self.val_dataset = self._instantiate_dataset(val_dataset)
        self.test_dataset = self._instantiate_dataset(test_dataset)

    def _instantiate_dataset(self, ds_cfg):
        if ds_cfg is None:
            return None
        if isinstance(ds_cfg, (dict, DictConfig)):
            if "_target_" in ds_cfg:
                # Single dataset config
                return instantiate(ds_cfg)
            else:
                # Multi-head: dict of dataset configs
                return {name: instantiate(cfg) for name, cfg in ds_cfg.items()}
        # Already an instantiated Dataset object
        return ds_cfg


    def _get_dataloader(self, dataset, shuffle=False):
        if isinstance(dataset, dict):
            # MULTI-DATASET MODE: Use CombinedLoader
            loaders = {
                name: DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
                for name, ds in dataset.items()
            }
            # 'max_size_cycle' ensures all datasets are seen even if sizes differ
            return CombinedLoader(loaders, mode="max_size_cycle")
        else:
            # SINGLE DATASET MODE
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)
