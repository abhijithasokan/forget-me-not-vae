import time
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from forget_me_not.models.vae import VAE, VAEWithCriticNetwork



class Losses:
    def __init__(self, vae_module, loss:str):
        self.vae_module = vae_module
        self.loss_func = self.get_loss_function(loss)

    def __call__(self, *args):
        return self.loss_func(*args)
        
    def vanilla_beta_vae_loss(self, batch):
        x, _ = batch
        _, x_recons, mean, log_var = self.vae_module.model.forward(x)
        return VAE.negative_elbo(x, x_recons, mean, log_var, self.vae_module.beta) / x.size(0)
    

    def _critic_loss(self, batch, critic_func):
        x, _ = batch
        z, x_recons, mean, log_var = self.vae_module.model.forward(x)
        vanilla_loss = VAE.negative_elbo(x, x_recons, mean, log_var, beta=1.0) / x.size(0)
        critic_loss = critic_func(z, x, mean, log_var)
        return vanilla_loss + self.vae_module.lmbda * critic_loss
        

    def _self_critic_loss(self, z, _, mean, log_var):
        num_samples, dim_z = z.shape
        # Although num_samples = num_latents, we keep the names different for clarity
        num_latents = num_samples

        # Exapnding dimensions for the broadcasting trick
        z = z.unsqueeze(1) # shape: (num_latents, 1, dim_z)
        mean = mean.unsqueeze(0) # shape: (1, num_samples, dim_z)
        log_var = log_var.unsqueeze(0)
        

        dev = z - mean # this computes distance of z from each mean. shape: (num_latents, num_samples, dim_z)
        var = log_var.exp()

        log_densities = -0.5 * (dim_z * np.log(2 * np.pi) + log_var.sum(dim=-1) ) \
                        -0.5 * ( (dev**2) / var).sum(dim=-1) # shape: (num_latents, num_samples)
        
        log_p_z_given_x = log_densities

        labels = torch.arange(num_latents, device=z.device)
        loss = F.cross_entropy(log_p_z_given_x, labels)
        return loss


    def _nn_critic_loss(self, z, x, *args):
        if not isinstance(self.vae_module.model, VAEWithCriticNetwork):
            raise ValueError("The model doesn't have a critic network")

        critic_loss = self.vae_module.model.critic_model.forward(z, x)
        return critic_loss


    def self_critic_loss(self, batch):
        return self._critic_loss(batch, self._self_critic_loss)
    
    def nn_critic_loss(self, batch):
        return self._critic_loss(batch, self._nn_critic_loss)
    
    def get_loss_function(self, loss: str):
        all_losses = { 
            'vanilla-beta-vae' : self.vanilla_beta_vae_loss,
            'self-critic' : self.self_critic_loss,
            'nn-critic' : self.nn_critic_loss,
        }

        if loss not in all_losses:
            raise ValueError(f"{loss} isn't a supported loss. Supported losses are - {', '.join(all_losses.keys())}")

        return all_losses[loss]




class BetaVAEModule(pl.LightningModule):
    def __init__(self, vae_model: Union[VAE, VAEWithCriticNetwork], loss: str, beta: float, learning_rate: float):
        super().__init__()
        #self.save_hyperparameters()
        self.model = vae_model
        self.learning_rate = learning_rate
        self.beta = beta    
        self.loss_func = Losses(self, loss=loss)
        # fix this
        self.lmbda = self.beta 
        self.additional_monitoring_metric = {"train": {}, "test" : {}, "validation" : {}}


    def on_train_epoch_start(self):
        self.train_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss = self.loss_func(batch)     
        self.train_step_outputs.append({
            "loss" : loss,
        })
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss = self.loss_func(batch) 
        self.validation_step_outputs.append({
            "loss" : loss,
            "metrics" : self.compute_matrics('validation', batch),
        })
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.log_metrics('validation', self._aggregate_metrics('validation', [x['metrics'] for x in self.validation_step_outputs]))


    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(self.model.parameters(), self.learning_rate)  
        return {
            "optimizer": optimizer,
        }


    def add_additional_monitoring_metric(self, stage, metric_label, metric_func, agg_func=torch.mean, timeit=False):
        metric = self.additional_monitoring_metric[stage][metric_label] = {}
        metric['metric_func'] = metric_func
        metric['agg_func']  = agg_func
        metric['timeit'] = timeit

    def compute_matrics(self, stage, batch):
        with torch.set_grad_enabled(False):
            results = {}
            for metric_label, metric in self.additional_monitoring_metric[stage].items():
                res = results[metric_label] = {}
                if metric['timeit']:
                    start = time.time()
                    res['value'] = metric['metric_func'](self.model, batch)
                    end = time.time()
                    res['time'] = end - start
                else:
                    res['value'] = metric['metric_func'](self.model, batch)

            return results

    def _aggregate_metrics(self, stage, results):
        agg_result = {}
        for metric_label, metric in self.additional_monitoring_metric[stage].items():
            agg_result[metric_label] = {}
            metric_results = torch.tensor([res[metric_label]['value'] for res in results] )
            agg_result[metric_label]['value'] = metric['agg_func'](metric_results)

            if metric['timeit']:
                metric_time = sum(res[metric_label]['time'] for res in results)
                agg_result[metric_label]['time'] = metric_time
        return agg_result

    def log_metrics(self, stage, results):
        for metric_label, metric_res in results.items():
            self.logger.experiment.add_scalar(f"{metric_label}/{stage.title()}", metric_res['value'], self.current_epoch)
            if 'time' in metric_res:
                self.logger.experiment.add_scalar(f"{metric_label}-time/{stage.title()}", metric_res['time'], self.current_epoch)




def train(
        model, 
        train_data_loader, 
        val_data_loader, 
        num_epochs, 
        accelerator,  
        check_val_every_n_epoch=5,
        enable_progress_bar=True, 
        early_stop=False
    ):
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = []
    if early_stop:    
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        callbacks.append(early_stop_callback)

    logger = TensorBoardLogger('train_logs', 'vae_experiments')

    trainer = pl.Trainer(accelerator=accelerator,max_epochs=num_epochs, 
        check_val_every_n_epoch=check_val_every_n_epoch, 
        enable_progress_bar=enable_progress_bar, callbacks=callbacks, logger=logger, 
    )
             
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)