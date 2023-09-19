import os

from .text_ds_base import TextDatasetBaseModule


class YelpDatasetModule(TextDatasetBaseModule):
    def __init__(self, data_dir: str):
        data_file_paths = {
            'train': os.path.join(data_dir, 'yelp.train.txt'),
            'eval': os.path.join(data_dir, 'yelp.valid.txt'),
            'test': os.path.join(data_dir, 'yelp.test.txt')
        }
        super().__init__(data_file_paths, max_len=None, has_label=True)



def test(data_dir):
    dm = YelpDatasetModule(data_dir=data_dir)

    dm.setup('fit')
    dm.setup('test')

    train_dl = dm.train_dataloader(batch_size=64)
    batch = next(iter(train_dl))
    text_data, labels = batch
    first_text = text_data['tokenised_text'][0]
    sentence = ' '.join(dm.train_ds.decode(first_text))
    label = labels[0]
    assert len(dm.train_ds.decode(first_text)) == text_data['lengths'][0]
    
    return dm, batch, sentence, label