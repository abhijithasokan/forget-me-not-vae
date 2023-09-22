import os
from functools import partial
from datetime import datetime
import logging

import torch

from forget_me_not.models.encoders.lstm_encoder import LSTMEncoder
from forget_me_not.models.decoders.lstm_decoder import LSTMDecoder
from forget_me_not.models.lstm_vae import LSTMVAE, CriticNetworkForLSTMVAE
from forget_me_not.training.train_beta_vae import BetaVAEModule
from forget_me_not import metrics 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


class Config:
    __DEFAULTS_CONFIGS = {
        'HIDDEN_DIM' : 1024,
        'LATENT_DIM' : 32,
        'BETA' : 10.0,
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
    }


    def __init__(self):
        self.__dict__.update(self.__DEFAULTS_CONFIGS)
        

    def set(self, key, value):
        self.__setattr__(key, value)


    def dump_config(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)




class ExperimentRunner:
    def __init__(self, config: Config, checkpoint_path=None, only_eval=True):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.only_eval = only_eval
        if checkpoint_path is not None and (not only_eval):
            raise ValueError("Resuming training from checkpoint is not supported yet.")


    def set_config_attr(self, attr, value):
        self.config.set(attr, value)


    def run(self):
        self.setup_data_module()
        self.model = self.build_model()
        self.setup_report_dir()

        self.setup_metrics()

        if not self.only_eval:
            self.add_monitoring_metrics(self.model, self.metric_and_its_params)            
            self.setup_trainer()
            self.train()
    
        self.report_metrics()
        #self.dump_plots()


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

        if self.checkpoint_path is not None:
            return BetaVAEModule.load_from_checkpoint(self.checkpoint_path, vae_model)

        if self.config.model_name == 'beta-vae':
            loss = 'vanilla-beta-vae'
            return BetaVAEModule(vae_model, loss, beta=self.config.BETA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)

        elif self.config.model_name == 'self-critic':
            loss = 'self-critic'
            return BetaVAEModule(vae_model, loss, beta=self.config.LAMBDA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)

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
            return BetaVAEModule(vae_model, loss, beta=self.config.LAMBDA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)

        elif self.config.model_name == 'hybrid-critic':
            vae_model.add_hybrid_critic_with_embedding_sharing(
                critic_text_enc_dim = self.config.CRITIC_ENC_DIM, 
                latent_dim = self.config.LATENT_DIM, 
                contrast_dim = self.config.CONTRAST_DIM, 
                hidden_dim_x = self.config.HIDDEN_DIM_X,
                hidden_dim_z = self.config.HIDDEN_DIM_Z,
            )

            loss = 'nn-critic'
            return BetaVAEModule(vae_model, loss, beta=self.config.LAMBDA, learning_rate=self.config.LEARNING_RATE, size_fn=self.size_func)

        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")



    def add_monitoring_metrics(self, model, metric_and_its_params):
        model.add_additional_monitoring_metric('validation', 'NLL', partial(metrics.compute_negative_log_likelihood_for_batch, size_fn = self.size_func, **metric_and_its_params['negative_log_likelihood']), timeit=True)
        model.add_additional_monitoring_metric('validation', 'AU', partial(metrics.active_units_for_batch, **metric_and_its_params['active_units']), timeit=True, agg_func=partial(torch.mean, dtype=torch.float32))
        model.add_additional_monitoring_metric('validation', 'MI', partial(metrics.mutual_information_for_batch, size_fn = self.size_func, **metric_and_its_params['mutual_information']), timeit=True)


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
        
    def setup_report_dir(self):
        self.report_dir = os.path.join(self.config.REPORT_ROOT_DIR, self.config.model_name, self.config.dataset, datetime.now().strftime('%y_%m_%d_%H_%M') )
        os.makedirs(self.report_dir, exist_ok=True)
        self.train_log_sub_dir = f'{self.config.dataset}_{self.config.model_name}'
        self.config.dump_config(os.path.join(self.report_dir, 'config.json'))


    def train(self):
        val_data_loader = self.dm.val_dataloader(batch_size=self.config.BATCH_SIZE)
        train_data_loader = self.dm.train_dataloader(batch_size=self.config.BATCH_SIZE)

        self.trainer.fit(self.model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
        self.trainer.save_checkpoint(os.path.join(self.logger.log_dir, f'{self.config.model_name}.ckpt'))


    def tune_lr(self):
        val_data_loader = self.dm.val_dataloader(batch_size=self.config.BATCH_SIZE)
        train_data_loader = self.dm.train_dataloader(batch_size=self.config.BATCH_SIZE)
    
        lr_find_kwargs = {'min_lr': 1e-06, 'max_lr': 1.0, 'early_stop_threshold': None, 'num_training' : 30 }
        tuner = pl.tuner.Tuner(self.trainer)
        tuner.lr_find(self.model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader, **lr_find_kwargs)


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


    def report_metrics(self):
        import json
        model = self.model.model
        if torch.cuda.is_available():
            model = model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=self.config.BATCH_SIZE)
        results = metrics.compute_metrics(model, test_data_loader, self.metric_and_its_params, size_fn=self.size_func)

        with open(os.path.join(self.report_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)


    def dump_plots(self):
        from forget_me_not.plots import plot_latent_representation_2d
        model = self.model.model
        if torch.cuda.is_available():
            model = model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=self.config.BATCH_SIZE)
        data, labels = next(iter(test_data_loader))
        plot_latent_representation_2d(model, data, labels, self.report_dir)




def main(args):
    cfg = Config()
    cfg.set('model_name', args.model)
    cfg.set('dataset', args.dataset)
    cfg.set('REPORT_ROOT_DIR', args.report_root_dir)
    cfg.set('MAX_NUM_EPOCHS', args.max_epochs)
    cfg.set('ACCELERATOR', args.accelerator)

    runner = ExperimentRunner(cfg, checkpoint_path=args.checkpoint, only_eval=args.only_eval)   
    runner.run()




