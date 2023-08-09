import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, latent_dim: int, dtype=torch.float64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.dtype = dtype
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
        
    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)    # std = sqrt(exp(log_var))
        eps = torch.randn_like(std)
        return mean + eps * std    # equivalent to sampling from N(mean, variance)
    
    def forward(self, x, deterministic=False):
        mean, log_var = self.get_posterir_dist_params(x)
        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)    # latent_representation
        
        reconstruction = self.decoder(z)
        return z, reconstruction, mean, log_var
    
    def get_posterir_dist_params(self, data):
        mean_and_variance = self.encoder(data).view(-1, 2, self.latent_dim)
        mean, log_var = mean_and_variance.permute(1,0,2)
        return mean, log_var
    
    def get_latent_representation(self, data, deterministic=False):
        mean, log_var = self.get_posterir_dist_params(data)
        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)    # latent_representation
        return z

    @classmethod
    def negative_elbo(cls, x, x_recons, mean, log_var, beta):
        # The below term corresponds to the Logliklihood term in the VAE loss
        # bce_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        # reconn_loss = bce_loss
        reconn_loss = (x_recons - x).pow(2).sum()
        
        # Below is the KL divergence part of the VAE loss 
        # For eqn, see - https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        kl_div = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mean.pow(2))
        return reconn_loss + beta*kl_div
    

    def generate_sample_from_latent_prior(self, num_samples):
        z = torch.randn(1, num_samples, self.latent_dim, dtype=self.dtype)
        return self.decoder(z)
    




    
class CriticNetwork(nn.Module):
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

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self, z, x):
        z_emb = self.z_enc(z)
        z_emb = z_emb / z_emb.norm(dim=1, keepdim=True)
        x_emb = self.x_enc(x)
        x_emb = x_emb / x_emb.norm(dim=1, keepdim=True)


        logits1 = self.temperature * z_emb @ x_emb.t()
        logits2 = logits1.t()
        labels = torch.arange(logits1.size(0), device=logits1.device)
        loss1 = F.cross_entropy(logits1, labels)
        loss2 = F.cross_entropy(logits2, labels)
        loss = (loss1 + loss2)/2
        return loss





class VAEWithCriticNetwork(VAE):
    def __init__(self, critic_model: CriticNetwork, *args, **kwargs):
        super(VAEWithCriticNetwork, self).__init__(*args, **kwargs)
        self.critic_model = critic_model
        
