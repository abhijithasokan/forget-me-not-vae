import torch
from torch import nn
import torch.nn.functional as F


from .vae_base import VAEWithGaussianPrior, CriticNetworkBase

'''
    The CNN VAE implementation here is adapted from - https://github.com/sksq96/pytorch-vae/blob/14ce22796fe0ee5120f4f1cdb1dbaf515e72559d/vae.py#L15
'''


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNNEncoder(nn.Module):
    def __init__(self, num_channels: int):
        super(CNNEncoder, self).__init__()
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
         
    def forward(self, x):
        return self.layers(x)


    @property
    def dtype(self):
        return self.layers[0].weight.dtype



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




class CNNVAEWithCriticNetwork(CNNVAE):
    def __init__(self, critic_model: CriticNetworkForCNNVAE, *args, **kwargs):
        super(CNNVAEWithCriticNetwork, self).__init__(*args, **kwargs)
        self.critic_model = critic_model
        
