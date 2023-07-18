import torch
import numpy as np

from prdc import compute_prdc

from forget_me_not.models.vae import VAE
from forget_me_not.utils.misc import cliped_iter_dataloder


def compute_negative_log_likelihood(vae_model, dataloader, dim, num_importance_sampling):
    """
    Computes the negative log likelihood of the samples under the VAE model.
    Adapted from: https://github.com/ioangatop/srVAE/blob/dfee765c53f11f4653e7c6e7118a339832656867/src/modules/loss.py#L24
    """ 
    vae_model.eval()
    
    nll_batches = []
    ds_size = len(dataloader.dataset)
    for batch in dataloader:
        x, _ = batch
        elbo_samples = []
        for _ in range(num_importance_sampling):
            with torch.no_grad():
                _, x_recons, mean, log_var = vae_model.forward(x)
            elbo = -VAE.negative_elbo(x, x_recons, mean, log_var, beta=1).item()
            elbo_samples.append(elbo)
        elbo_samples = torch.tensor(elbo_samples)
        nll_batch = - ( torch.logsumexp(elbo_samples, dim=0) - np.log(num_importance_sampling) )
        nll_batches.append(nll_batch)
    nll = torch.tensor(nll_batches).sum().item()/ds_size # negative log likelihood
    bpd = nll / (dim * np.log(2.)) # bits per dimension
    return bpd


def active_units(vae_model, dataloader):
    vae_model.eval()
    all_latents = []
    for batch in dataloader:
        x, _ = batch
        with torch.no_grad():
            latents = vae_model.get_latent_representation(x)
        all_latents.append(latents)

    all_latents = torch.cat(all_latents, dim=0)
    active_units = (all_latents.std(dim=0) > 0.1).sum().item()
    return active_units




def mutual_information(vae_model, dataloader, num_samples):
    """
    Computes the mutual information between the latent variables and the data.
    Adapted from: https://github.com/jxhe/vae-lagging-encoder/blob/cdc4eb9d9599a026bf277db74efc2ba1ec203b15/modules/encoders/encoder.py#L111

    This is an approximation of the mutual information between the latent variables (z) and the data (x).
    The formula:
        I(z,x) = E[log q(z|x)] - E[log q(z)]
        The expectation is taken over the  approximate posterior distribution
    """
    vae_model.eval()
    all_mu, all_logvar = [], []
    for batch in cliped_iter_dataloder(dataloader, num_samples):
        x, _ = batch
        with torch.no_grad():
            _, _, mean, log_var = vae_model.forward(x)
        all_mu.append(mean)
        all_logvar.append(log_var)  

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    dim_z = all_mu.shape[1]

    # Compute the first term: E[log q(z|x)]
    # Since this is same as entropy, we can use the equation of entropy for a Gaussian distribution
    # Entropy = D/2 * (1 + log(2pi)) + 1/2 * log(det(sigma))
    # You can find the equation at - https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    neg_gaussian_entropy = - (0.5 * dim_z * (1 + np.log(2*np.pi)) + 0.5 * all_logvar.sum(dim=-1))



    # Compute the second term: E[log q(z)]
    z = VAE.reparameterize(all_mu, all_logvar)
    
    # Exapnding dimensions for the broadcasting trick
    z = z.unsqueeze(1) # shape: (num_samples, 1, dim_z) # TODO: use the z from the previous step
    all_mu = all_mu.unsqueeze(0) # shape: (1, num_samples, dim_z)
    all_logvar = all_logvar.unsqueeze(0)
    

    dev = z - all_mu # this computes distance of z from each mean.  shape: (num_samples, num_samples, dim_z)
    all_var = all_logvar.exp()

    log_densities = -0.5 * (dim_z * np.log(2 * np.pi) + all_logvar.sum(dim=-1) ) \
                    -0.5 * ( (dev**2) / all_var).sum(dim=-1) # shape: (num_samples, num_samples)
    
    # shape: (num_samples,)
    # q_z[i] = sum(log_densities[i]) / num_samples
    log_q_z = torch.logsumexp(log_densities, dim=1) - np.log(num_samples) 

    # Finally, compute the mutual information
    mutual_information = (neg_gaussian_entropy.mean() - log_q_z.mean()).item()
    return mutual_information



def compute_density_and_coverage(vae_model, dataloader, num_samples, nearest_k=5):
    vae_model.eval()
    real_samples = []
    for batch in cliped_iter_dataloder(dataloader, num_samples):
        x, _ = batch
        real_samples.append(x)
    real_samples = torch.cat(real_samples, dim=0)
    with torch.no_grad():
        generated_samples = vae_model.generate_sample_from_latent_prior(num_samples).squeeze(0)
    return compute_prdc(real_samples, generated_samples, nearest_k=nearest_k)