import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence




class LSTMEncoder(nn.Module):
    def __init__(self, dim: int, emb_dim: int, vocab_size: int):
        super(LSTMEncoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(
            input_size = emb_dim,
            hidden_size = dim,
            num_layers = 1, 
            batch_first = True
        )
        #self.init_params()


    def init_params(self):
        # The init args used are are discussed in the Lagging VAE paper : Section B.2
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.01, 0.01) 
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)


    def forward(self, x):
        tokenised_text, lengths = x['tokenised_text'], x['lengths']

        word_embed = self.embed(tokenised_text)
        packed_embed = pack_padded_sequence(word_embed, lengths.tolist(), batch_first=True, enforce_sorted=False)

        _, (last_state, _) = self.lstm(packed_embed)

        return last_state.squeeze(0)

    def share_embedding_layer_with(self, other):
        self.embed = other.embed

    @property
    def device(self):
        return self.lstm.weight_ih_l0.device

    @property
    def dtype(self):
        return self.lstm.weight_ih_l0.dtype