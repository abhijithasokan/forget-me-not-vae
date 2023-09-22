import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from forget_me_not.utils.misc import move_to_device
 

def plot_latent_representation_2d(vae_model, samples, labels, report_dir: str = None):
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
    vae_model.eval()
    samples = move_to_device(samples, vae_model.device)

    with torch.no_grad():
        latent_rep = vae_model.get_latent_representation(samples, deterministic=False)
        latent_rep = latent_rep.cpu().numpy()
    
    pca = PCA(n_components=2)
    reduced_rep = pca.fit_transform(latent_rep)
    
    data = reduced_rep
    plt.scatter(data[:, 0], data[:, 1], c=labels.unsqueeze(1), marker='.')
    if report_dir is not None:
        plt.savefig(os.path.join(report_dir, 'latent_rep.png'))
    else:
        plt.show()
    plt.close()

def plot_reconstruction_2d(vae_model, samples, labels, report_dir: str = None):
    vae_model.eval()
    samples = move_to_device(samples, vae_model.device)
    with torch.no_grad():
        _, recon, *_ = vae_model.forward(samples, deterministic=False)
        recon = recon.cpu().numpy()
        
    pca = PCA(n_components=2)
    reduced_rep = pca.fit_transform(recon)
    
    data = reduced_rep
    plt.scatter(data[:, 0], data[:, 1], c=labels.unsqueeze(1), marker='.')
    if report_dir is not None:
        plt.savefig(os.path.join(report_dir, 'recon.png'))
    else:
        plt.show()
    plt.close()
    
def plot_latent_and_reconstruction(vae_model, test_data_loader, report_dir: str = None):
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
    vae_model.eval()
    with torch.no_grad():
        data, labels = next(iter(test_data_loader))
        plot_latent_representation_2d(vae_model, data, labels, report_dir)
        plot_reconstruction_2d(vae_model, data, labels, report_dir)




def plot_latent_representation_2d_with_batches(vae_model, test_data_loader, report_dir: str = None):
    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
    vae_model.eval()

    all_latents = []
    all_labels = []
    for batch in test_data_loader:
        samples, labels = batch
        all_labels.append(labels)
        samples = move_to_device(samples, vae_model.device)
        with torch.no_grad():
            latent_rep = vae_model.get_latent_representation(samples, deterministic=False)
            latent_rep = latent_rep
            all_latents.append(latent_rep)
        del latent_rep
        del samples
        del labels
    
    all_latents = torch.cat(all_latents, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0)
    pca = PCA(n_components=2)
    reduced_rep = pca.fit_transform(all_latents)
    
    data = reduced_rep
    plt.scatter(data[:, 0], data[:, 1], c=all_labels.unsqueeze(1), marker='.')
    if report_dir is not None:
        plt.savefig(os.path.join(report_dir, 'latent_rep.png'))
    else:
        plt.show()
    plt.close()
