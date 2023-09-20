import os
from functools import partial
from datetime import datetime

import torch

from forget_me_not.models.encoders.lstm_encoder import LSTMEncoder
from forget_me_not.models.decoders.lstm_decoder import LSTMDecoder
from forget_me_not.models.lstm_vae import LSTMVAE, CriticNetworkForLSTMVAE
from forget_me_not.training.train_beta_vae import BetaVAEModule, train
from forget_me_not import metrics 




class Config:
    __DEFAULTS_CONFIGS = {
        'HIDDEN_DIM' : 1024,
        'LATENT_DIM' : 32,
        'BETA' : 10.0,
        'LAMBDA' : 10.0,

        # Training settings
        'LEARNING_RATE' : 0.05,
        'BATCH_SIZE' : 64,
        'MAX_NUM_EPOCHS' : 30,
        'ACCELERATOR' : 'cpu',
        'EARLY_STOP' : True,
        'ROOT_TRAIN_LOG_DIR' : './train_logs/text_exps',

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
    def __init__(self, config: Config):
        self.config = config


    def set_config_attr(self, attr, value):
        self.config.set(attr, value)


    def run(self):
        self.setup_data_module()
        self.model = self.build_model()
        self.setup_report_dir()

        self.setup_metrics()
        self.add_monitoring_metrics(self.model, self.metric_and_its_params)
        
        self.train()
    
        self.report_metrics()
        self.dump_plots()


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
        model.add_additional_monitoring_metric('validation', 'NLL', partial(metrics.compute_negative_log_likelihood_for_batch, size_fn = self.size_by_words, **metric_and_its_params['negative_log_likelihood']), timeit=True)
        model.add_additional_monitoring_metric('validation', 'AU', partial(metrics.active_units_for_batch, **metric_and_its_params['active_units']), timeit=True, agg_func=partial(torch.mean, dtype=torch.float32))
        model.add_additional_monitoring_metric('validation', 'MI', partial(metrics.mutual_information_for_batch, **metric_and_its_params['mutual_information']), timeit=True)


    def setup_metrics(self):
        self.metric_and_its_params = {
            "negative_log_likelihood" : { 
                'num_importance_sampling' : 5, #500
            },
            "active_units" : {},
            "mutual_information" : {
                'num_samples' : 1000,
            },
            # "density_and_coverage" : {
            #     'nearest_k' : 5
            # }
        }
        
    def setup_report_dir(self):
        self.report_dir = os.path.join(self.config.REPORT_ROOT_DIR, self.config.model_name, self.config.dataset, datetime.now().strftime('%y_%m_%d_%H_%M') )
        os.makedirs(self.report_dir, exist_ok=True)
        self.config.dump_config(os.path.join(self.report_dir, 'config.json'))



    def train(self):
        val_data_loader = self.dm.val_dataloader(batch_size=self.config.BATCH_SIZE)
        train_data_loader = self.dm.train_dataloader(batch_size=self.config.BATCH_SIZE)

        train(self.model, train_data_loader, val_data_loader, num_epochs=self.config.MAX_NUM_EPOCHS, 
            accelerator=self.config.ACCELERATOR, enable_progress_bar=self.config.PBAR, early_stop=self.config.EARLY_STOP,
            root_train_log_dir=self.config.ROOT_TRAIN_LOG_DIR, sub_dir=self.config.model_name)


    def report_metrics(self):
        import json
        if torch.cuda.is_available():
            model = self.model.model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=self.config.BATCH_SIZE)
        results = metrics.compute_metrics(model, test_data_loader, self.metric_and_its_params, size_fn=self.size_by_words)

        with open(os.path.join(self.report_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)


    def dump_plots(self):
        from forget_me_not.plots import plot_latent_representation_2d
        if torch.cuda.is_available():
            model = self.model.model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=None)
        data, labels = next(iter(test_data_loader))
        plot_latent_representation_2d(model, data, labels, self.report_dir)




def main(args):
    cfg = Config()
    cfg.set('model_name', args.model)
    cfg.set('dataset', args.dataset)
    cfg.set('REPORT_ROOT_DIR', args.report_root_dir)
    cfg.set('MAX_NUM_EPOCHS', args.max_epochs)
    cfg.set('ACCELERATOR', args.accelerator)

    runner = ExperimentRunner(cfg)
    runner.run()




