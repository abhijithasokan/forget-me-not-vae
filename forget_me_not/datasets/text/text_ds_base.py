from typing import Iterable, Dict
import itertools

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from forget_me_not.datasets.base import DatasetBase, DataModuleBase




class Vocab:
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    EOS_TOKEN = '</s>'
    SOS_TOKEN = '<s>'
    PAD_TOKEN_ID = 0
    
    __SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        for token in self.__SPECIAL_TOKENS:
            self.add_word(token, skip_check=True)

        assert Vocab.PAD_TOKEN_ID == self.word2idx[Vocab.PAD_TOKEN]


    def add_word(self, word: str, skip_check: bool = False):
        if not skip_check and word in self.word2idx:
            return
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)


    def decode(self, sentence_of_tokens: Iterable):
        sent = itertools.takewhile(lambda x: x != self.word2idx[self.PAD_TOKEN], sentence_of_tokens)
        decoded_sentence = list(map(lambda x: self.idx2word[x], sent))
        return decoded_sentence


    @classmethod
    def from_corpus(cls, vocab_fname: str):
        vocab = cls()
        with open(vocab_fname) as fin:
            for line in fin:
                for word in line.split():
                    vocab.add_word(word)
        return vocab
    

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, word):
        return self.word2idx[word]

    def __contains__(self, word):
        return word in self.word2idx
    



class TextDatasetBase(DatasetBase):
    def __init__(self, text_fname: str, max_len: int = None, has_label: bool = False, vocab: Vocab = None):
        super().__init__()
        self.max_len = max_len
        self.data = []
        self.has_label = has_label
        self.have_to_build_vocab = vocab is None
        if self.have_to_build_vocab:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

        self.data, self.labels, self.dropped_line_count = self.read_data(text_fname)


    def read_data(self, text_fname: str):
        dropped_line_count = 0
        data = []
        labels = []

        with open(text_fname) as fin:
            for line in fin:
                if self.has_label:
                    label, text = line.split('\t')
                    labels.append(int(label.strip()))
                else:
                    text = line
                
                words = text.split()
                if len(words) == 0 or (self.max_len is not None and len(words) > self.max_len):
                    dropped_line_count += 1
                    continue
                
                for word in words:
                    if self.have_to_build_vocab and word not in self.vocab:
                        self.vocab.add_word(word)
                
                token_ids = [self.vocab[Vocab.SOS_TOKEN]]
                token_ids.extend([self.vocab[word] for word in words])
                token_ids.append(self.vocab[Vocab.EOS_TOKEN])
                data.append( torch.tensor(token_ids) )

        return data, labels, dropped_line_count


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.has_label:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]
    

    def decode(self, sentence_of_tokens: Iterable):
        decoded_sentence = self.vocab.decode(sentence_of_tokens)
        return decoded_sentence
    



class TextDatasetBaseModule(DataModuleBase):
    def __init__(self, data_file_paths: Dict[str, str], max_len: int = None, has_label: bool = False):
        super().__init__()
        self.max_len = max_len
        self.data_file_paths = data_file_paths
        self.has_label = has_label

        self.vocab = None
        if 'vocab' in data_file_paths:
            self.vocab = Vocab.from_corpus(self.data_file_paths['vocab'])
        

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_ds = TextDatasetBase(self.data_file_paths['train'], self.max_len, self.has_label, self.vocab)
            if self.vocab is None:
                self.vocab = self.train_ds.vocab
            
            self.eval_ds = TextDatasetBase(self.data_file_paths['eval'], self.max_len, self.has_label, self.vocab)

        elif stage == 'test':
            if self.vocab is None:
                raise ValueError("Vocab must be set before test stage")
            self.test_ds = TextDatasetBase(self.data_file_paths['test'], self.max_len, self.has_label, self.vocab)

        else:
            raise ValueError(f"Unknown stage: {stage}")
        

    def collate_fn(self, batch):
        if self.has_label:
            tokenised_text, labels = zip(*batch)
            labels = torch.tensor(labels)
        else:
            tokenised_text = batch
            labels = None

        lengths = torch.tensor([len(tt) for tt in tokenised_text])
        tokenised_text_padded = pad_sequence(tokenised_text, batch_first=True, padding_value=self.vocab[Vocab.PAD_TOKEN])

        collated_batch = {'tokenised_text': tokenised_text_padded, 'lengths': lengths}
        
        return collated_batch, labels
