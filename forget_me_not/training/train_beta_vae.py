import torch
from torch import nn
import pytorch_lightning as pl
from forget_me_not.models.vae import VAE


class BetaVAEModule(pl.LightningModule):
    def __init__(self, vae_model, beta, learning_rate):
        super().__init__()
        #self.save_hyperparameters()
        self.model = vae_model
        self.learning_rate = learning_rate
        self.beta = beta    
        self.setup_log_ouputs()

    def setup_log_ouputs(self):
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def compute_loss(self, batch):
        x, _ = batch
        x_recons, mean, log_var = self.model.forward(x)
        return VAE.negative_elbo(x, x_recons, mean, log_var, self.beta) / len(x)


    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)     
        self.train_step_outputs.append({
            "loss" : loss,
        })

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.train_step_outputs = []
        

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch) 
        self.validation_step_outputs.append({
            "loss" : loss
        })
    

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", avg_loss)
        self.logger.experiment.add_scalar("Loss/Validation", avg_loss, self.current_epoch)
        self.validation_step_outputs = []


    def configure_optimizers(self):
        optimizer =  torch.optim.AdamW(self.model.parameters(), self.learning_rate)  
        return {
            "optimizer": optimizer,
        }




def train(model, train_data_loader, val_data_loader, num_epochs=100, accelerator=None,  
          enable_progress_bar=True, early_stop=True):
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = []
    if early_stop:    
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        callbacks.append(early_stop_callback)

    logger = TensorBoardLogger('train_logs', 'vae_experiments')

    trainer = pl.Trainer(accelerator=accelerator,max_epochs=num_epochs, 
        check_val_every_n_epoch=1, enable_progress_bar=enable_progress_bar, callbacks=callbacks, logger=logger, 
    )
             
    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)