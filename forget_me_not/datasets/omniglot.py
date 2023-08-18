import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, TensorDataset

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
        



class OmniglotDatasetPreprocessed(DataModuleBase):
    '''
        This dataset is a preprocessed version of the Omniglot dataset, used in the paper SA-VAE - https://arxiv.org/pdf/1802.02550.pdf
        Details to download and preprocess the dataset can be found here - https://github.com/harvardnlp/sa-vae
        Originally from - https://arxiv.org/pdf/1509.00519.pdf

        Img size: 28x28
        Eval and test sets are binarized (see here for exact process that was done - https://github.com/harvardnlp/sa-vae/blob/e5187a2116a22ef00a8a806c9685f02f05bf2374/preprocess_img.py#L10)
        Train set should dynamically binarized.
    '''

    class TensorDatasetWithTransform(DatasetBase):
        def __init__(self, data: torch.Tensor, label: torch.tensor, transform: callable = None):
            super().__init__()
            self.data = data
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            item = self.data[index]
            if self.transform is not None:
                item = self.transform(item)
            return item, self.label[index]
        
        
    def __init__(self, data_mat_path: str):
        super().__init__()
        train_x, train_y, eval_x, eval_y, test_x, test_y = torch.load(data_mat_path)
        self.train_ds = OmniglotDatasetPreprocessed.TensorDatasetWithTransform(train_x, train_y, transform=torch.bernoulli)
        self.eval_ds = OmniglotDatasetPreprocessed.TensorDatasetWithTransform(eval_x, eval_y)
        self.test_ds = OmniglotDatasetPreprocessed.TensorDatasetWithTransform(test_x, test_y)
    

    @classmethod
    def to_pil_img(cls, mat):
        return Image.fromarray( (mat.numpy() * 255).astype(np.uint8)[0] )
   
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass
        elif stage == 'test':
            pass
        else:
            raise ValueError(f'Unknown stage: {stage}')