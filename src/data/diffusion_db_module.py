import pytorch_lightning as pl
from .diffusion_db import DiffusionDBDataset
import torch
from torch.utils.data import DataLoader

class DiffusionDBModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32,
                 img_transform=None,
                 subset_name="large_first_50k"):
        super().__init__()
        self.batch_size = batch_size
        self.subset_name = subset_name
        self.img_transform = img_transform

    def setup(self, stage: str):
        diffusion_db_full = DiffusionDBDataset(subset_name=self.subset_name,
                img_transform=self.img_transform)
        self.diffusion_db_train, self.diffusion_db_valid, self.diffusion_db_test = torch.utils.data.random_split(diffusion_db_full, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(12))

    def train_dataloader(self):
        return DataLoader(self.diffusion_db_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.diffusion_db_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.diffusion_db_test, batch_size=self.batch_size)