import pytorch_lightning as pl
import diffusion_db
import torch
from torch.utils.data import DataLoader

class DiffusionDBModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        diffusion_db_full = diffusion_db.DiffusionDBDataset()
        self.diffusion_db_train, self.diffusion_db_valid, self.diffusion_db_test = torch.utils.data.random_split(diffusion_db_full, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(12))

    def train_dataloader(self):
        return DataLoader(self.diffusion_db_train)

    def valid_dataloader(self):
        return DataLoader(self.diffusion_db_valid)

    def test_dataloader(self):
        return DataLoader(self.diffusion_db_test)