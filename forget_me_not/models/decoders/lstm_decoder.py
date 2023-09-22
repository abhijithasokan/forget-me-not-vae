import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from forget_me_not.datasets.text.text_ds_base import Vocab




'''
    Architecture from the paper titled "Generating Sentences from a Continuous Space"
    - https://arxiv.org/pdf/1511.06349.pdf

    From the above paper - "The decoder serves as a special RNN language model that is conditioned on this hidden code"
        Here the hidden code is the latent variable z

    A short summary of the architecture is also discussed in the paper titled "Improved Variational Autoencoders for Text Modeling using Dilated Convolutions" under section 2.1

    Below implementation is adapted from the project - https://github.com/jxhe/vae-lagging-encoder/blob/cdc4eb9d9599a026bf277db74efc2ba1ec203b15/modules/decoders/dec_lstm.py
'''
class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, emb_dim:int, lstm_hidden_dim: int, vocab_size: int, dropout_in: float, dropout_out: float, pad_token_id: int = Vocab.PAD_TOKEN_ID):
        super(LSTMDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self.dropout_in_layer = nn.Dropout(dropout_in)

        self.latent_to_lstm_hidden = nn.Linear(latent_dim, lstm_hidden_dim, bias=False)  

        self.lstm = nn.LSTM(
            input_size = emb_dim + latent_dim,
            hidden_size = lstm_hidden_dim,
            num_layers = 1,
            batch_first = True
        )

        self.dropout_out_layer = nn.Dropout(dropout_out)
        self.prediction_layer = nn.Linear(lstm_hidden_dim, vocab_size, bias=False)
            
        vocab_mask = torch.ones(vocab_size, requires_grad=False)
        vocab_mask[pad_token_id] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
        
        #self.init_params()


    def init_params(self):
        # The init args used are are discussed in the Lagging VAE paper : Section B.2
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.01, 0.01) 
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        

    def _concat_z_and_word_embed(self, word_embed, z):
        batch_size, _ = z.size()
        seq_len = word_embed.size(1)

        # creating copies of z for each word in the sequence
        z_replicates = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)
        word_embed_with_z = torch.cat((word_embed, z_replicates), -1)
        return word_embed_with_z


    def forward(self, z, x):
        return self.decode(z, x)

    def decode(self, z, x):
        # x['tokenised_text']: (batch_size, seq_len)
        # z: (batch_size, n_sample, self.latent_dim)
        
        tokenised_text = x['tokenised_text']
        lengths = x['lengths'] - 1 # skip predicting start symbol

        word_embed = self.embed(tokenised_text)
        word_embed = self.dropout_in_layer(word_embed)

        # Concatenating word embeddings and z
        word_embed_with_z = self._concat_z_and_word_embed(word_embed, z)

        # ------- Push through LSTM -------
        packed_embed = pack_padded_sequence(word_embed_with_z, lengths.tolist(), batch_first=True, enforce_sorted=False)

        c_init = self.latent_to_lstm_hidden(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(packed_embed, (h_init, c_init))

        output, _ = pad_packed_sequence(output, batch_first=True)
        # ------- End of LSTM --------------

        output = self.dropout_out_layer(output)
        output_logits = self.prediction_layer(output)

        return output_logits


    def reconstruction_loss(self, x, x_recons):
        batch_size, seq_len, _ = x_recons.size()
        tokenised_text = x['tokenised_text'][:, 1:] # since start symbol is not predicted

        # Reshaping to compute loss
        x_recons_reshaped = x_recons.view(-1, x_recons.size(2))
        tokenised_text_flat = tokenised_text.reshape(-1)

        loss = self.loss(x_recons_reshaped, tokenised_text_flat)
        loss = loss.view(batch_size, seq_len).sum(-1)
        return loss


    @property
    def dtype(self):
        return self.lstm.weight_ih_l0.dtype
    
    @property
    def device(self):
        return self.lstm.weight_ih_l0.device
    

    # def reconstruct_error(self, x, z):
    #     '''
    #     Returns:
    #         loss: (batch_size, n_sample). Loss
    #         across different sentence and z
    #     '''

    #     tokenised_text = x['tokenised_text']
    #     lengths = x['lengths'] 

    #     src = tokenised_text[:, :-1] # remove end symbol
    #     tgt = tokenised_text[:, 1:]  # remove start symbol

    #     batch_size, seq_len = src.size()
    #     n_sample = z.size(1)

    #     # (batch_size * n_sample, seq_len, vocab_size)
    #     output_logits = self.decode(z, {'tokenised_text' : src, 'lengths' : lengths})

    #     if n_sample == 1:
    #         tgt = tgt.contiguous().view(-1)
    #     else:
    #         tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len).contiguous().view(-1)

    #     # (batch_size * n_sample * seq_len)
    #     loss = self.loss(output_logits.view(-1, output_logits.size(2)), tgt)
    #     return loss.view(batch_size, n_sample, -1).sum(-1)

    def generate_from_latent(self, z, output_list_of_tokens = False, max_len = 100):
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.latent_to_lstm_hidden(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([Vocab.SOS_TOKEN_ID] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1) # (batch_size, 1)
        end_symbol = torch.tensor([Vocab.EOS_TOKEN_ID] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < max_len:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.prediction_layer(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            max_index = torch.argmax(output_logits, dim=1)

            decoder_input = max_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    # decoded_batch[i].append(self.vocab.id2word(max_index[i].item()))
                    decoded_batch[i].append(max_index[i].item())

            mask = torch.mul((max_index != end_symbol), mask)

        if output_list_of_tokens:
            return decoded_batch

        else:
            tokenised_text = [torch.tensor(sentence, dtype=torch.long, device=self.device) for sentence in decoded_batch]
            lengths = torch.tensor([len(tt) for tt in tokenised_text])
            tokenised_text_padded = pad_sequence(tokenised_text, batch_first=True, padding_value=Vocab.PAD_TOKEN_ID)
            return {'tokenised_text': tokenised_text_padded, 'lengths': lengths}