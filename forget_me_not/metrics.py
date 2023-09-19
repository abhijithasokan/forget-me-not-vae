import logging
from operator import itemgetter

from prdc import compute_prdc
import torch
import numpy as np

from forget_me_not.utils.misc import cliped_iter_dataloder


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

DEFAULT_SIZE_FN = lambda x: x.size(0)

def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    else:
        raise NotImplementedError(f"Unknown type: {type(x)}")
    

def compute_negative_log_likelihood(vae_model, dataloader, num_importance_sampling, size_fn=DEFAULT_SIZE_FN):
    """
    Computes the negative log likelihood of the samples under the VAE model.
    Adapted from: https://github.com/ioangatop/srVAE/blob/dfee765c53f11f4653e7c6e7118a339832656867/src/modules/loss.py#L24
    """ 
    vae_model.eval()
    
    nll_batches = []
    ds_size = 0
    for batch in dataloader:
        x, _ = batch
        x = move_to_device(x, vae_model.device)
        elbo_samples = []
        for _ in range(num_importance_sampling):
            with torch.no_grad():
                _, x_recons, mean, log_var = vae_model.forward(x)
            elbo = -vae_model.negative_elbo(x, x_recons, mean, log_var, beta=1).item()
            elbo_samples.append(elbo)
        elbo_samples = torch.tensor(elbo_samples)
        nll_batch = - ( torch.logsumexp(elbo_samples, dim=0) - np.log(num_importance_sampling) )
        nll_batches.append(nll_batch)
        ds_size += size_fn(x)
    nll = torch.tensor(nll_batches).sum().item()/ds_size # negative log likelihood
    # bpd = nll / (dim * np.log(2.)) # bits per dimension
    # return bpd
    return nll

def compute_negative_log_likelihood_for_batch(vae_model, batch, *args, **kwargs):
    dataloader = [batch]
    return compute_negative_log_likelihood(vae_model, dataloader, *args, **kwargs)




def active_units(vae_model, dataloader):
    vae_model.eval()
    all_latents = []
    for batch in dataloader:
        x, _ = batch
        x = move_to_device(x, vae_model.device)
        with torch.no_grad():
            latents = vae_model.get_latent_representation(x)
        all_latents.append(latents)

    all_latents = torch.cat(all_latents, dim=0)
    # correction=1 is for unbiased std? Saw similar implementation at - https://github.com/jxhe/vae-lagging-encoder/blob/cdc4eb9d9599a026bf277db74efc2ba1ec203b15/image.py#L145
    active_units = (all_latents.std(dim=0, correction=1) > 0.1).sum().item() 
    return active_units


def active_units_for_batch(vae_model, batch, *args, **kwargs):
    dataloader = [batch]
    return active_units(vae_model, dataloader, *args, **kwargs)




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
        x = move_to_device(x, vae_model.device)
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
    z = vae_model.reparameterize(all_mu, all_logvar)
    
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


def mutual_information_for_batch(vae_model, batch, *args, **kwargs):
    dataloader = [batch]
    return mutual_information(vae_model, dataloader, *args, **kwargs)




def compute_density_and_coverage(vae_model, dataloader, nearest_k=5, num_samples=None):
    vae_model.eval()
    all_latents = []
    latent_of_generated_samples = []
    for batch in cliped_iter_dataloder(dataloader, num_samples):
        x, _ = batch
        with torch.no_grad():
            latent_samples = vae_model.get_latent_representation(x)
            all_latents.append(latent_samples)
            generated_samples = vae_model.generate_sample_from_latent_prior(latent_samples.size(0))
            latent_of_generated_samples.append(vae_model.get_latent_representation(generated_samples))

    all_latents = torch.cat(all_latents, dim=0)
    latent_of_generated_samples = torch.cat(latent_of_generated_samples, dim=0)
    return compute_prdc(latent_samples, latent_of_generated_samples, nearest_k=nearest_k)


def compute_density_and_coverage_for_batch(vae_model, batch, *args, **kwargs):
    dataloader = [batch]
    return compute_density_and_coverage(vae_model, dataloader, *args, **kwargs)





def compute_metrics(vae_model, test_data_loader, metric_and_its_params, size_fn=DEFAULT_SIZE_FN):
    logging.info("Computing metrics")
    vae_model.eval()
    with torch.no_grad():
        res = {}
        if "negative_log_likelihood" in metric_and_its_params:
            nll = compute_negative_log_likelihood(vae_model, test_data_loader, size_fn=size_fn, **metric_and_its_params["negative_log_likelihood"])
            res["Negative log likelihood"] = nll
            logging.info(f"Negative log likelihood: {nll}")

        if "active_units" in metric_and_its_params:
            au = active_units(vae_model, test_data_loader)
            res["Active units"] =  au
            logging.info(f"Active units: {au}")

        if "mutual_information" in metric_and_its_params:
            mi = mutual_information(vae_model, test_data_loader, **metric_and_its_params["mutual_information"])
            res["Mutual information"] = mi
            logging.info(f"Mutual information: {mi}")

        if "density_and_coverage" in metric_and_its_params:
            dc = compute_density_and_coverage(vae_model, test_data_loader, **metric_and_its_params["density_and_coverage"])
            res["Density"] = dc["density"]
            res["Coverage"] = dc["coverage"]
            logging.info(f"Density: {dc['density']}")
            logging.info(f"Coverage: {dc['coverage']}")
    
        return res