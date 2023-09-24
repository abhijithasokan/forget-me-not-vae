import os

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from forget_me_not.models.cnn_vae import CNNVAE, CNNEncoder, CNNDecoder, CriticNetworkForCNNVAE
from forget_me_not.training.train_beta_vae import BetaVAEModule
from forget_me_not import metrics 
from .exp_base import Config, ExperimentRunner




class ImgExpConfig(Config):
    DEFAULTS_CONFIGS = {
        'HIDDEN_DIM' : 512,
        'LATENT_DIM' : 32,
        'BETA' : 1.0,
        'LAMBDA' : 20.0,

        # Training settings
        'LEARNING_RATE' : 0.0002,
        'BATCH_SIZE' : 1024,
        'MAX_NUM_EPOCHS' : 30,
        'ACCELERATOR' : 'cpu',
        'EARLY_STOP' : True,
        'ROOT_TRAIN_LOG_DIR' : './train_logs/img_exps',
        'CHECK_VAL_EVERY_N_EPOCH' : 3,

        # Misc
        'REPORT_ROOT_DIR' : './reports_img',
        'PBAR' : False,

        'ROOT_DS_DIR' : './ext/datasets/',

        'CONTRAST_DIM' : 128,
        'HIDDEN_DIM_X' : 256,
        'HIDDEN_DIM_Z' : 64,

        'DUMP_PLOTS' : True,
    }




class ImgExperimentRunner(ExperimentRunner):
    def __init__(self, config: ImgExpConfig, *args, **kwargs):
        super(ImgExperimentRunner, self).__init__(config, *args, **kwargs)

    @staticmethod
    def size_func(x):
        return x.shape[0]
    

    def setup_data_module(self):
        if self.config.dataset == 'omniglot':
            from forget_me_not.datasets.omniglot import OmniglotDatasetPreprocessed
            dm = OmniglotDatasetPreprocessed(data_mat_path=os.path.join(self.config.ROOT_DS_DIR, 'omniglot.pt'))
        elif self.config.dataset == 'mnist':
            raise NotImplementedError("MNIST dataset is not implemented yet")
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        dm.setup('fit')
        dm.setup('test')
        self.dm = dm
        

    def build_model(self):
        vae_model = CNNVAE(
            img_encoder = CNNEncoder(num_channels=1),
            img_decoder = CNNDecoder(num_channels=1, dim=self.config.HIDDEN_DIM),
            hidden_dim = self.config.HIDDEN_DIM,
            latent_dim = self.config.LATENT_DIM,
        ) 

        if self.config.model_name == 'beta-vae':
            loss = 'vanilla-beta-vae'
            
        elif self.config.model_name == 'self-critic':
            loss = 'self-critic'

        elif self.config.model_name == 'nn-critic':
            critic_network = CriticNetworkForCNNVAE(
                img_encoder = CNNEncoder(num_channels=1),
                img_enc_dim = self.config.HIDDEN_DIM, 
                latent_dim = self.config.LATENT_DIM, 
                contrast_dim = self.config.CONTRAST_DIM, 
                hidden_dim_x = self.config.HIDDEN_DIM_X,
                hidden_dim_z = self.config.HIDDEN_DIM_Z,
                dtype=torch.float32
            )
            vae_model.add_nn_critic(critic_network)
            loss = 'nn-critic'
            
        elif self.config.model_name == 'hybrid-critic':
            vae_model.add_hybrid_critic(
                num_shared_layer=8, 
                contrast_dim=self.config.CONTRAST_DIM, 
                hidden_dim_x=self.config.HIDDEN_DIM_X, 
                hidden_dim_z=self.config.HIDDEN_DIM_Z
            )
            loss = 'nn-critic'
        
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        if self.checkpoint_path is not None:
            return BetaVAEModule.load_from_checkpoint(self.checkpoint_path, vae_model)
        
        if self.config.model_name == 'beta-vae':
            return BetaVAEModule(vae_model, loss, beta=self.config.BETA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)
        else:
            return BetaVAEModule(vae_model, loss, beta=self.config.LAMBDA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)


    def setup_metrics(self):
        self.metric_and_its_params = {
            "negative_log_likelihood" : { 
                'num_importance_sampling' : 50, #500
            },
            "active_units" : {},
            "mutual_information" : {
                'num_samples' : 1000,
            },
            "density_and_coverage" : {
                'nearest_k' : 5
            }
        }
        
    
    def setup_trainer(self):
        self.logger = TensorBoardLogger(self.config.ROOT_TRAIN_LOG_DIR, self.train_log_sub_dir)

        callbacks = []
        if self.config.EARLY_STOP:    
            early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
            callbacks.append(early_stop_callback)

        self.trainer = pl.Trainer(accelerator=self.config.ACCELERATOR, max_epochs=self.config.MAX_NUM_EPOCHS, 
            check_val_every_n_epoch=self.config.CHECK_VAL_EVERY_N_EPOCH, 
            enable_progress_bar=self.config.PBAR, callbacks=callbacks, logger=self.logger
        )




def main(args):
    cfg = ImgExpConfig()
    cfg.set('model_name', args.model)
    cfg.set('dataset', args.dataset)
    cfg.set('REPORT_ROOT_DIR', args.report_root_dir)
    cfg.set('MAX_NUM_EPOCHS', args.max_epochs)
    cfg.set('ACCELERATOR', args.accelerator)

    runner = ImgExperimentRunner(cfg, checkpoint_path=args.checkpoint, only_eval=args.only_eval, seed=args.seed) 
    runner.run()




