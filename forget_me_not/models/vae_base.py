from abc import abstractmethod, ABC
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class VAEBase(ABC, nn.Module):
    @staticmethod
    @abstractmethod
    def reparameterize(mean, log_var):
        pass

    
    @abstractmethod
    def negative_elbo(self, x, x_recons, mean, log_var, beta):
        pass
    
    @abstractmethod
    def generate_sample_from_latent_prior(self, num_samples):
        pass

    @abstractmethod
    def get_posterir_dist_params(self, data):
        pass
    
    def forward(self, x, deterministic=False):
        mean, log_var = self.get_posterir_dist_params(x)
        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)    # latent_representation
        
        reconstruction = self.decoder(z)
        return z, reconstruction, mean, log_var

    def add_nn_critic(self, critic: 'CriticNetworkBase'):
        if hasattr(self, 'critic'):
            raise ValueError('Critic already exists')
        self.critic = critic

    def add_hybrid_critic(self, critic: 'CriticNetworkBase'):
        raise NotImplementedError
    
    def get_latent_representation(self, data, deterministic=False):
        mean, log_var = self.get_posterir_dist_params(data)
        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)    # latent_representation
        return z

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError




class CriticNetworkBase(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super(CriticNetworkBase, self).__init__()
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





class VAEWithGaussianPrior(VAEBase): #is this a good name?
    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)    # std = sqrt(exp(log_var))
        eps = torch.randn_like(std)
        return mean + eps * std    # equivalent to sampling from N(mean, variance)
    

    @staticmethod
    def mse_loss(x, x_recons):
        return (x_recons - x).pow(2).sum()

    
    def negative_elbo(self, x, x_recons, mean, log_var, beta):
        # The below term corresponds to the Logliklihood term in the VAE loss
        # bce_loss = nn.functional.binary_cross_entropy(x_recons, x, reduction='sum')
        # reconn_loss = bce_loss
        if hasattr(self, 'reconstruction_loss'):
            reconn_loss = self.reconstruction_loss(x, x_recons)
        else:
            reconn_loss = VAEWithGaussianPrior.mse_loss(x, x_recons)
            
        # Below is the KL divergence part of the VAE loss 
        # For eqn, see - https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        kl_div = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mean.pow(2))
        return reconn_loss + beta*kl_div
    

    def generate_sample_from_latent_prior(self, num_samples):
        # when prior is a normal distribution with mean 0 and variance 1 
        z = torch.randn(num_samples, self.latent_dim, dtype=self.dtype)
        return self.decoder(z)