import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from forget_me_not.datasets.base import DatasetBase, DataModuleBase
from forget_me_not.utils.misc import DeterministicRandomness


class OmniglotDataset(DatasetBase):
    def __init__(self, data_dir: str, img_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.ds = torchvision.datasets.Omniglot(self.data_dir, background=True, download=True)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label =  self.ds[index]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label



class OmniglotDataModule(DataModuleBase):
    def __init__(self, data_dir: str, train_fraction: float, eval_fraction: float, *args, **kwargs):
        super().__init__()
        
        self.data_dir = data_dir
        img_transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize(64), transforms.ConvertImageDtype(torch.float32)])
        self.ds = OmniglotDataset(self.data_dir, img_transform=img_transform)
    
        tf = train_fraction * (1 - eval_fraction)
        ef = train_fraction * eval_fraction
        with DeterministicRandomness(0):    
            self.train_ds, self.eval_ds, self.test_ds = random_split(self.ds, [tf, ef, 1 - train_fraction])
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass
        elif stage == 'test':
            pass
        else:
            raise ValueError(f'Unknown stage: {stage}')