import pytorch_lightning as pl
from torch.utils.data import DataLoader

class SegDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            test_dataset, 
            global_batch_size=32, 
            num_workers=4,
            num_nodes=1,
            num_devices=1
    ):
        super(SegDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = global_batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
