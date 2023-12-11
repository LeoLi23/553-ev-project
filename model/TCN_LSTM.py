import torch
import torch.nn as nn
from model.TCN import TemporalConvNet


class Encoder(nn.Module):
    def __init__(self, input_size, seq_len, tcn_num_channels, lstm_num_hidden, tcn_kernel_size=2, tcn_dropout=0.2):
        super(Encoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, tcn_num_channels, tcn_kernel_size, tcn_dropout)
        self.fc_feature = nn.Linear(tcn_num_channels[-1], lstm_num_hidden)
        self.fc_time = nn.Linear(seq_len, 1)
        self.lstm_num_hidden = lstm_num_hidden
    
    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)) # (batch_size, tcn_num_channels[-1], seq_len)
        output = output.transpose(1, 2) # (batch_size, seq_len, tcn_num_channels[-1])
        output = self.fc_feature(output) # (batch_size, seq_len, lstm_num_hidden)

        h = output[:, -1, :] # (batch_size, lstm_num_hidden)
        c = output.transpose(1, 2) # (batch_size, lstm_num_hidden, seq_len)
        c = self.fc_time(c).squeeze(2) # (batch_size, lstm_num_hidden)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers=1, covariate=False, covariate_size=0):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        if covariate:
            self.fc = nn.Linear(hidden_size+covariate_size, input_size)
        else:
            self.fc = nn.Linear(hidden_size, input_size)
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.covariate = covariate

    def forward(self, xt, hidden, cell, covariates=None):
        # repeat the hidden states according to the number of layers
        h = hidden.repeat(self.num_layers, 1, 1)
        c = cell.repeat(self.num_layers, 1, 1)

        outputs = []
        for t in range(self.seq_len):
            output, (h, c) = self.lstm(xt, (h, c))
            if self.covariate:
                output = torch.cat((output, covariates[:, t, :].unsqueeze(1)), dim=2)
            output = self.fc(output)
            outputs.append(output[:, :, -1].unsqueeze(2))
            xt = output  # use the decoder output as the next input

        outputs = torch.cat(outputs, dim=1)
        return outputs


class TCN_LSTM(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, input_len, output_len, tcn_num_channels, lstm_num_hidden, 
                 tcn_kernel_size=2, tcn_dropout=0.2, num_layers=1, covariate=False, covariate_size=0):
        super(TCN_LSTM, self).__init__()
        self.encoder = Encoder(encoder_input_size, input_len, tcn_num_channels, lstm_num_hidden, tcn_kernel_size, tcn_dropout)
        self.decoder = Decoder(decoder_input_size, output_len, lstm_num_hidden, num_layers, covariate, covariate_size)
        self.decoder_input_size = decoder_input_size

    def forward(self, historic_inputs, covariates=None):
        """
        By default, the last feature of the encoder input is the target feature.
        And the decoder_input = encoder_input[-decoder_input_size:]
        """
        # x: (batch_size, input_len, input_size)
        h, c = self.encoder(historic_inputs)
        xt = historic_inputs[:, -1, -self.decoder_input_size:].unsqueeze(1)
        if self.decoder_input_size == 1:
            xt = xt.unsqueeze(2)
        outputs = self.decoder(xt, h, c, covariates) # (batch_size, output_len, 1)
        return outputs
    