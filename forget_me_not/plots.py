import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

def plot_latent_representation_2d(vae_model, samples, labels, report_dir: str = None):
    vae_model.eval()
    with torch.no_grad():
        latent_rep = vae_model.get_latent_representation(samples, deterministic=False)
        
    pca = PCA(n_components=2)
    reduced_rep = pca.fit_transform(latent_rep)
    
    data = reduced_rep
    plt.scatter(data[:, 0], data[:, 1], c=labels.unsqueeze(1), marker='.')
    if report_dir is not None:
        plt.savefig(os.path.join(report_dir, 'latent_rep.png'))
    else:
        plt.show()

def plot_reconstruction_2d(vae_model, samples, labels, report_dir: str = None):
    vae_model.eval()
    with torch.no_grad():
        _, recon, *_ = vae_model.forward(samples, deterministic=False)
        
    pca = PCA(n_components=2)
    reduced_rep = pca.fit_transform(recon)
    
    data = reduced_rep
    plt.scatter(data[:, 0], data[:, 1], c=labels.unsqueeze(1), marker='.')
    if report_dir is not None:
        plt.savefig(os.path.join(report_dir, 'recon.png'))
    else:
        plt.show()
    
def plot_latent_and_reconstruction(vae_model, test_data_loader, report_dir: str = None):
    os.makedirs(report_dir, exist_ok=True)
    vae_model.eval()
    with torch.no_grad():
        data, labels = next(iter(test_data_loader))
        plot_latent_representation_2d(vae_model, data, labels, report_dir)
        plot_reconstruction_2d(vae_model, data, labels, report_dir)