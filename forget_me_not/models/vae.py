import torch
from torch import nn

from .vae_base import VAEBase, CriticNetworkBase, VAEWithGaussianPrior

class VAE(VAEWithGaussianPrior):
    def __init__(self, dim: int, hidden_dim: int, latent_dim: int, dtype=torch.float64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self._dtype = dtype
        middle_layer_dim = hidden_dim #int((dim*latent_dim) ** 0.5) 
        

        self.encoder = nn.Sequential(
            nn.Linear(dim, middle_layer_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(middle_layer_dim, 2*latent_dim, dtype=dtype),  # both mean, (log)variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, middle_layer_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(middle_layer_dim, dim, dtype=dtype),
            #nn.Sigmoid(),
        )
            
    
    def get_posterir_dist_params(self, data):
        mean_and_variance = self.encoder(data).view(-1, 2, self.latent_dim)
        mean, log_var = mean_and_variance.permute(1,0,2)
        return mean, log_var

    @property
    def dtype(self):
        return self._dtype
    

    
    
class CriticNetwork(CriticNetworkBase):
    def __init__(self, dim: int, latent_dim: int, contrast_dim: int, hidden_dim_x: int, hidden_dim_z: int, dtype=torch.float64):
        super(CriticNetwork, self).__init__()

        self.latent_dim = latent_dim
        self.dim = dim
        self.contrast_dim = contrast_dim

        self.x_enc = nn.Sequential(
            nn.Linear(dim, hidden_dim_x, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_x, contrast_dim, dtype=dtype),
        )

        self.z_enc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_z, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, contrast_dim, dtype=dtype),
        )




