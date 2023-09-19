import logging

import torch
from torch import nn
import torch.nn.functional as F

from .vae_base import VAEWithGaussianPrior, CriticNetworkBase
from .encoders.lstm_encoder import LSTMEncoder
from .decoders.lstm_decoder import LSTMDecoder

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)



class LSTMVAE(VAEWithGaussianPrior):
    def __init__(self, lstm_encoder: LSTMEncoder, lstm_decoder: LSTMDecoder, enc_dim: int = 1024, latent_dim : int = 32):
        super(LSTMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = enc_dim

        self.encoder = lstm_encoder
        self.fc_mean = nn.Linear(enc_dim, latent_dim, bias=False)
        self.fc_logvar = nn.Linear(enc_dim, latent_dim, bias=False)


        self.fc_latent_to_hidden = nn.Linear(latent_dim, self.hidden_dim, bias=False)
        self.lstm_decoder = lstm_decoder
        
    
    def get_posterir_dist_params(self, data):
        hidden_rep = self.encoder(data)
        mean, log_var = self.fc_mean(hidden_rep), self.fc_logvar(hidden_rep)
        return mean, log_var

    @property
    def dtype(self):
        return self.encoder.dtype
    
    @property
    def device(self):
        return self.encoder.device


    def reconstruction_loss(self, x, x_recons):
        reconn_loss = self.lstm_decoder.reconstruction_loss(x, x_recons).sum(dim=-1)
        return reconn_loss


    def forward(self, x, deterministic=False):
        mean, log_var = self.get_posterir_dist_params(x)
        if deterministic:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)    # latent_representation
        
        zh = self.fc_latent_to_hidden(z)
        reconstruction = self.lstm_decoder(zh, x)
        return z, reconstruction, mean, log_var


    def add_hybrid_critic_with_embedding_sharing(self, critic_text_enc_dim, *args, **kwargs):
        if hasattr(self, 'critic'):
            raise ValueError('A critic already exists')
        
        text_encoder = LSTMEncoder(dim=critic_text_enc_dim, emb_dim=self.encoder.embed.embedding_dim, vocab_size=self.encoder.embed.num_embeddings)
        text_encoder.share_embedding_layer_with(self.encoder)
        self.critic = CriticNetworkForLSTMVAE(text_encoder, critic_text_enc_dim, *args, dtype=self.dtype, **kwargs)


class CriticNetworkForLSTMVAE(CriticNetworkBase):
    def __init__(self, text_encoder: LSTMEncoder, text_enc_dim: int, latent_dim: int, contrast_dim: int, hidden_dim_x: int, hidden_dim_z: int, dtype=torch.float64):
        super(CriticNetworkForLSTMVAE, self).__init__()

        self.latent_dim = latent_dim
        self.contrast_dim = contrast_dim

        self.x_enc = nn.Sequential(
            text_encoder,
            nn.Linear(text_enc_dim, hidden_dim_x, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_x, contrast_dim, dtype=dtype),
        )

        self.z_enc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_z, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim_z, contrast_dim, dtype=dtype),
        )

