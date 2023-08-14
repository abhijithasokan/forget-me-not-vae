import os
import numpy as np
import torch

from forget_me_not.datasets.base import DatasetBase
class GaussianMixture(DatasetBase):
    def __init__(
            self, 
            n_samples: int, 
            n_features: int, 
            n_classes: int,
            variance_scale: tuple = (0.5, 1.0), 
            mean_scale: tuple = (0, 1.0),
            seed:int = 0
        ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed
        self.variance_scale = variance_scale
        self.mean_scale = mean_scale
        self.data, self.labels = self._generate_data()


    def _generate_data(self):
        np.random.seed(self.seed)
        data = []
        self.means = []
        self.covs = []
        labels = []
        
        for class_ind in range(self.n_classes):
            mean_from_std = np.random.randn(self.n_features)
            normalized_mean = mean_from_std / np.linalg.norm(mean_from_std)
            scale_corrected_mean = normalized_mean * np.random.uniform(*self.mean_scale)
            mean = scale_corrected_mean

            cov = np.random.uniform(*self.variance_scale)
            self.means.append(mean)
            self.covs.append(cov)
        
            cov = np.eye(self.n_features) * cov
            data.append(np.random.multivariate_normal(mean, cov, self.n_samples))
            labels.append(np.ones(self.n_samples) * class_ind)
        
        data = np.concatenate(data)
        labels = np.concatenate(labels)

        shuffled_indices = np.arange((len(data)))
        np.random.shuffle(shuffled_indices)
        data = data[shuffled_indices]
        labels = labels[shuffled_indices]
        np.random.seed()
        return torch.from_numpy(data), torch.from_numpy(labels)
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def plot(self):
        return self.plot_helper(self.data, self.labels)

    def plot_filtered(self, filtered_inds, save_fig_path: str = None):
        return self.plot_helper(self.data[filtered_inds], self.labels[filtered_inds], save_fig_path)
    
    @staticmethod
    def plot_helper(data, labels, save_fig_path: str = None):
        from sklearn.decomposition import PCA
        if data.shape[1] > 2:
            data = PCA(n_components=2).fit_transform(data)
        import matplotlib.pyplot as plt
        plt.scatter(data[:, 0], data[:, 1], c=labels.unsqueeze(1), marker='.')
        if save_fig_path is not None:
            plt.savefig(save_fig_path)
        else:
            plt.show()
        plt.close()


from forget_me_not.utils.misc import DeterministicRandomness
from torch.utils.data import random_split
from forget_me_not.datasets.base import DataModuleBase

class GaussianMixtureDataModule(DataModuleBase):
    def __init__(self, train_fraction: float, eval_fraction: float, *args, **kwargs):
        super().__init__()
        self.ds = GaussianMixture(*args, **kwargs)
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

    def plot_train(self, report_dir: str = None):
        fig_save_path = None if report_dir is None else os.path.join(report_dir, 'train.png')
        self.ds.plot_filtered(self.train_ds.indices, fig_save_path)

    def plot_test(self, report_dir: str = None):
        fig_save_path = None if report_dir is None else os.path.join(report_dir, 'test.png')
        self.ds.plot_filtered(self.test_ds.indices, fig_save_path)

    def plot_eval(self, report_dir: str = None):
        fig_save_path = None if report_dir is None else os.path.join(report_dir, 'eval.png')
        self.ds.plot_filtered(self.eval_ds.indices, fig_save_path)




def test():
    dataset = GaussianMixture(n_samples=200, n_features=2, n_classes=10, variance_range=(0.01, 0.1))
    dataset.plot()

