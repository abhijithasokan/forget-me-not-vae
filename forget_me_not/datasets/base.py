from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pytorch_lightning import LightningDataModule


class DatasetBase(ABC, Dataset):
    pass


class DataModuleBase(ABC, LightningDataModule):
    @abstractmethod
    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self, batch_size: int = None):
        batch_size = batch_size if batch_size is not None else len(self.train_ds)
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
    
    def test_dataloader(self, batch_size: int = None):
        batch_size = batch_size if batch_size is not None else len(self.test_ds)
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

    def val_dataloader(self, batch_size: int = None):
        batch_size = batch_size if batch_size is not None else len(self.eval_ds)
        return DataLoader(self.eval_ds, batch_size=batch_size, shuffle=False)

