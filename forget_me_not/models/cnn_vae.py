import logging
from copy import deepcopy
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from .vae_base import VAEWithGaussianPrior, CriticNetworkBase

'''
    The CNN VAE implementation here is adapted from - https://github.com/sksq96/pytorch-vae/blob/14ce22796fe0ee5120f4f1cdb1dbaf515e72559d/vae.py#L15
'''

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNNEncoder(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNEncoder, self).__init__()
        self.shared_layers = nn.Sequential(nn.Identity())
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1), # 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 12x12
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # 5x5
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2), # 2x2
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1), # 1x1
            nn.ReLU(),
            Flatten()
        )
        self._dtype = self.layers[0].weight.dtype
         
    def forward(self, x):
        x = self.shared_layers(x)
        return self.layers(x)

    
    def split_layers_to_shared_and_non_shared(self, num_shared_layers: int):
        if num_shared_layers == 0:
            return 
        if num_shared_layers > len(self.layers):
            raise ValueError(f'num_shared_layers ({num_shared_layers}) should be less than the number of layers ({len(self.layers)})')
        
        self.shared_layers = nn.Sequential(*self.layers[:num_shared_layers])
        self.layers = nn.Sequential(*self.layers[num_shared_layers:])
        if len(self.layers) == 0:
            self.layers = nn.Sequential(nn.Identity())
        logging.info(f'First {num_shared_layers} layers are shared.  The last 2 of the shared layers are: {self.shared_layers[-2:]}')

    def get_copy(self):
        copy_encoder = deepcopy(self)
        copy_encoder.shared_layers = self.shared_layers  # Share the shared_layers parameter
        copy_encoder.reset_parameters()
        return copy_encoder


    @property
    def dtype(self):
        return self._dtype

    def reset_parameters(self):
        for layer in itertools.chain(self.shared_layers, self.layers):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()



class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class CNNDecoder(nn.Module):
    '''
        params
            num_channels: number of channels in the input image
            dim: latent dimension
    '''
    def __init__(self, num_channels: int, dim: int):
        super(CNNDecoder, self).__init__()
        self.layers = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(dim, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )
         
    def forward(self, x):
        return self.layers(x)




class CNNVAE(VAEWithGaussianPrior):
    def __init__(self, img_encoder: CNNEncoder, img_decoder: CNNDecoder, hidden_dim: int = 1024, latent_dim : int =32):
        super(CNNVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = img_encoder
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)


        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            self.fc_latent_to_hidden,
            img_decoder
        )
    
    
    def get_posterir_dist_params(self, data):
        hidden_rep = self.encoder(data)
        mean, log_var = self.fc_mean(hidden_rep), self.fc_logvar(hidden_rep)
        return mean, log_var

    @property
    def dtype(self):
        return self.encoder.dtype


    def add_hybrid_critic(self, num_shared_layer: int, contrast_dim: int, hidden_dim_x: int, hidden_dim_z: int):
        if hasattr(self, 'critic'):
            raise ValueError('A critic already exists')
        self.encoder.split_layers_to_shared_and_non_shared(num_shared_layer)
        copy_encoder = self.encoder.get_copy()
        self.critic = CriticNetworkForCNNVAE(copy_encoder, self.hidden_dim, self.latent_dim, contrast_dim, hidden_dim_x, hidden_dim_z, dtype=self.dtype)


class CriticNetworkForCNNVAE(CriticNetworkBase):
    def __init__(self, img_encoder: CNNEncoder, img_enc_dim: int, latent_dim: int, contrast_dim: int, hidden_dim_x: int, hidden_dim_z: int, dtype=torch.float64):
        super(CriticNetworkForCNNVAE, self).__init__()

        self.latent_dim = latent_dim
        self.contrast_dim = contrast_dim

        self.x_enc = nn.Sequential(
            img_encoder,
            nn.Linear(img_enc_dim, hidden_dim_x, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_x, contrast_dim, dtype=dtype),
        )

        self.z_enc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_z, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, contrast_dim, dtype=dtype),
        )


