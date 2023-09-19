import os

from .text_ds_base import TextDatasetBaseModule


class YahooDatasetModule(TextDatasetBaseModule):
    def __init__(self, data_dir: str):
        data_file_paths = {
            'train': os.path.join(data_dir, 'yahoo.train.txt'),
            'eval': os.path.join(data_dir, 'yahoo.valid.txt'),
            'test': os.path.join(data_dir, 'yahoo.test.txt'),
            'vocab': os.path.join(data_dir, 'vocab.txt')
        }
        super().__init__(data_file_paths, max_len=None, has_label=False)

