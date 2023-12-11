import torch
import torch.nn as nn


class LSTM_LSTM(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, input_len, output_len, lstm_num_hidden, num_layers=1, covariate_size=0, covariate=False):
        super(LSTM_LSTM, self).__init__()
        self.name = 'LSTM_LSTM'
        if covariate:
            assert covariate_size > 0
        self.encoder = nn.LSTM(encoder_input_size, lstm_num_hidden, num_layers, batch_first=True)
        self.decoder = nn.LSTM(decoder_input_size, lstm_num_hidden, num_layers, batch_first=True)
        if covariate:
            self.fc = nn.Linear(lstm_num_hidden+covariate_size, decoder_input_size)
        else:
            self.fc = nn.Linear(lstm_num_hidden, decoder_input_size)
        self.input_len = input_len
        self.output_len = output_len
        self.decoder_input_size = decoder_input_size
        self.covariate = covariate

    def forward(self, x, covariates=None):
        """
        By default, the last feature of the encoder input is the target feature.
        And the decoder_input = encoder_input[-decoder_input_size:]
        """
        # x: (batch_size, input_len, input_size)
        _, (h, c) = self.encoder(x)
        xt = x[:, -1, -self.decoder_input_size:].unsqueeze(1)
        if self.decoder_input_size == 1:
            xt = xt.unsqueeze(2)
        outputs = []
        for t in range(self.output_len):
            output, (h, c) = self.decoder(xt, (h, c))
            if self.covariate:
                output = torch.cat((output, covariates[:, t, :].unsqueeze(1)), dim=2)
            output = self.fc(output)
            outputs.append(output[:, :, -1].unsqueeze(2))
            xt = output  # use the decoder output as the next input

        outputs = torch.cat(outputs, dim=1)
        return outputs