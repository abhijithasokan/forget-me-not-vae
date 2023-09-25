import os
from functools import partial
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from forget_me_not import metrics 




class Config(ABC):
    def __init__(self):
        self.__dict__.update(self.DEFAULTS_CONFIGS)
        
    def set(self, key, value):
        self.__setattr__(key, value)

    def dump_config(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)




class ExperimentRunner(ABC):
    def __init__(self, config: Config, checkpoint_path=None, only_eval=True, seed: int = 0):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.only_eval = only_eval
        if checkpoint_path is not None and (not only_eval):
            raise ValueError("Resuming training from checkpoint is not supported yet.")
        self.seed = seed
        seed_everything(seed)


    @staticmethod
    @abstractmethod
    def size_func(x):
        return x.size(0)
    
    @abstractmethod
    def setup_data_module(self):
        pass
        
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def setup_metrics(self):
        pass
    
    @abstractmethod
    def setup_trainer(self):
        pass


    def set_config_attr(self, attr, value):
        self.config.set(attr, value)


    def run(self):
        self.setup_data_module()
        seed_everything(self.seed)
        self.model = self.build_model()
        self.setup_report_dir()

        self.setup_metrics()

        if not self.only_eval:
            self.add_monitoring_metrics(self.model, self.metric_and_its_params)            
            self.setup_trainer()
            seed_everything(self.seed)
            self.train()
    
        seed_everything(self.seed)
        self.report_metrics()
        
        if self.config.DUMP_PLOTS:
            seed_everything(self.seed)
            self.dump_plots()
    

    def add_monitoring_metrics(self, model, metric_and_its_params):
        model.add_additional_monitoring_metric('validation', 'NLL', partial(metrics.compute_negative_log_likelihood_for_batch, size_fn = self.size_func, **metric_and_its_params['negative_log_likelihood']), timeit=True)
        model.add_additional_monitoring_metric('validation', 'AU', partial(metrics.active_units_for_batch, **metric_and_its_params['active_units']), timeit=True, agg_func=partial(torch.mean, dtype=torch.float32))
        model.add_additional_monitoring_metric('validation', 'MI', partial(metrics.mutual_information_for_batch, size_fn = self.size_func, **metric_and_its_params['mutual_information']), timeit=True)

        
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


    def report_metrics(self):
        import json
        model = self.model.model
        if torch.cuda.is_available():
            model = model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=self.config.BATCH_SIZE)
        results = metrics.compute_metrics(model, test_data_loader, self.metric_and_its_params, seed = self.seed, size_fn=self.size_func)

        with open(os.path.join(self.report_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)


    def dump_plots(self):
        from forget_me_not.plots import plot_latent_representation_2d_with_batches
        model = self.model.model
        if torch.cuda.is_available():
            model = model.cuda()
        test_data_loader = self.dm.test_dataloader(batch_size=self.config.BATCH_SIZE)
        plot_latent_representation_2d_with_batches(model, test_data_loader, self.report_dir)




