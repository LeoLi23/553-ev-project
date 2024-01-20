# with attention

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
    def forward(self, hist_h, prev_h):
        # hist_h: (num_layers, batch_size, num_hist, hidden_size)
        # prev_h: (num_layers, batch_size, hidden_size, 1)
        # => (num_layers, batch_size, num_hist)
        # => (num_layers, batch_size, hidden_size)
        scores = torch.bmm(hist_h, prev_h).squeeze(3)
        attn_weights = torch.softmax(scores, dim=2)
        context = torch.bmm(hist_h.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return context, attn_weights    


"""
The idea is to retrieve all the hidden states of the LSTM in encoder and use them as the context vector.
Then we do cross attention between the context vector and the hidden state of the decoder, to 
generate the attention weights. Finally, we use the attention weights to generate the cell state
of the decoder.

However, we haven't figured out how to retrieve all the hidden states of the LSTM in encoder with
performance guaranteed.
"""


class LSTM_LSTM_att(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, input_len, output_len, lstm_num_hidden, num_layers=1, covariate_size=0, covariate=False):
        super(LSTM_LSTM_att, self).__init__()
        self.name = 'LSTM_LSTM_att'
        pass