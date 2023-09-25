import os

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from forget_me_not.models.encoders.lstm_encoder import LSTMEncoder
from forget_me_not.models.decoders.lstm_decoder import LSTMDecoder
from forget_me_not.models.lstm_vae import LSTMVAE, CriticNetworkForLSTMVAE
from forget_me_not.training.train_beta_vae import BetaVAEModule
from forget_me_not import metrics 
from .exp_base import Config, ExperimentRunner




class TextExpConfig(Config):
    DEFAULTS_CONFIGS = {
        'HIDDEN_DIM' : 1024,
        'LATENT_DIM' : 32,
        'BETA' : 1.0,
        'LAMBDA' : 10.0,

        # Training settings
        'LEARNING_RATE' : 0.005,
        'BATCH_SIZE' : 64,
        'MAX_NUM_EPOCHS' : 30,
        'ACCELERATOR' : 'cpu',
        'EARLY_STOP' : True,
        'ROOT_TRAIN_LOG_DIR' : './train_logs/text_exps',
        'CLIP_GRAD_NORM' : 5.0,
        'CHECK_VAL_EVERY_N_EPOCH' : 3,

        # Misc
        'REPORT_ROOT_DIR' : './reports',
        'PBAR' : False,

        'ROOT_DS_DIR' : './ext/datasets/',

        'EMBEDDING_DIM' : 256,

        'CRITIC_ENC_DIM' : 1024,
        'CRITIC_EMBEDDING_DIM' : 128,


        'CONTRAST_DIM' : 128,
        'HIDDEN_DIM_X' : 256,
        'HIDDEN_DIM_Z' : 64,

        'DUMP_PLOTS' : False,
    }




class TextExperimentRunner(ExperimentRunner):
    def __init__(self, config: TextExpConfig, *args, **kwargs):
        super(TextExperimentRunner, self).__init__(config, *args, **kwargs)

    @staticmethod
    def size_func(x):
        return x['tokenised_text'].shape[0]
    
    @staticmethod
    def size_by_words(x):
        with torch.no_grad():
            text_tensor = x['tokenised_text']
            return torch.sum(text_tensor != 0)


    def setup_data_module(self):
        if self.config.dataset == 'yahoo':
            from forget_me_not.datasets.text.yahoo import YahooDatasetModule
            dm = YahooDatasetModule(data_dir=os.path.join(self.config.ROOT_DS_DIR, 'yahoo_data'))
        elif self.config.dataset == 'yelp':
            from forget_me_not.datasets.text.yelp import YelpDatasetModule
            dm = YelpDatasetModule(data_dir=os.path.join(self.config.ROOT_DS_DIR, 'yelp_data'))

        dm.setup('fit')
        dm.setup('test')
        self.config.vocab_size = len(dm.vocab)
        self.dm = dm
        

    def build_model(self):
        vocab_size = self.config.vocab_size
        vae_model = LSTMVAE(
            lstm_encoder = LSTMEncoder(
                dim = self.config.HIDDEN_DIM, 
                emb_dim = self.config.EMBEDDING_DIM, 
                vocab_size = vocab_size
            ),
            lstm_decoder = LSTMDecoder(
                latent_dim = self.config.HIDDEN_DIM, 
                emb_dim = self.config.EMBEDDING_DIM, 
                lstm_hidden_dim = self.config.HIDDEN_DIM, 
                vocab_size = vocab_size, 
                dropout_in=0.5, dropout_out=0.5
            ),
            enc_dim = self.config.HIDDEN_DIM, 
            latent_dim = self.config.LATENT_DIM
        )


        if self.config.model_name == 'beta-vae':
            loss = 'vanilla-beta-vae'
            
        elif self.config.model_name == 'self-critic':
            loss = 'self-critic'

        elif self.config.model_name == 'nn-critic':
            critic_network = CriticNetworkForLSTMVAE(
                text_encoder = LSTMEncoder(dim=self.config.CRITIC_ENC_DIM, emb_dim=self.config.CRITIC_EMBEDDING_DIM, vocab_size=vocab_size),
                text_enc_dim = self.config.CRITIC_ENC_DIM, 
                latent_dim = self.config.LATENT_DIM, 
                contrast_dim = self.config.CONTRAST_DIM, 
                hidden_dim_x = self.config.HIDDEN_DIM_X,
                hidden_dim_z = self.config.HIDDEN_DIM_Z,
                dtype=torch.float32
            )
            vae_model.add_nn_critic(critic_network)
            loss = 'nn-critic'
            
        elif self.config.model_name == 'hybrid-critic':
            vae_model.add_hybrid_critic_with_embedding_sharing(
                critic_text_enc_dim = self.config.CRITIC_ENC_DIM, 
                latent_dim = self.config.LATENT_DIM, 
                contrast_dim = self.config.CONTRAST_DIM, 
                hidden_dim_x = self.config.HIDDEN_DIM_X,
                hidden_dim_z = self.config.HIDDEN_DIM_Z,
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
                'num_samples' : None,
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
            enable_progress_bar=self.config.PBAR, callbacks=callbacks, logger=self.logger, 
            gradient_clip_val=self.config.CLIP_GRAD_NORM,
        )


    def dump_plots(self):
        raise NotImplementedError("Plots for text experiments are not implemented yet")




def main(args):
    cfg = TextExpConfig()
    cfg.set('model_name', args.model)
    cfg.set('dataset', args.dataset)
    cfg.set('REPORT_ROOT_DIR', args.report_root_dir)
    cfg.set('MAX_NUM_EPOCHS', args.max_epochs)
    cfg.set('ACCELERATOR', args.accelerator)

    runner = TextExperimentRunner(cfg, checkpoint_path=args.checkpoint, only_eval=args.only_eval, seed=args.seed)  
    runner.run()




