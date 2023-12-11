import torch
import torch.nn as nn


class LSTM_LSTM(nn.Module):
    def __init__(self, input_size, output_size, input_len, output_len, lstm_num_hidden, num_layers=1, covariate_size=0, covariate=False):
        super(LSTM_LSTM, self).__init__()
        if covariate:
            assert covariate_size > 0
        self.encoder = nn.LSTM(input_size, lstm_num_hidden, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, lstm_num_hidden, num_layers, batch_first=True)
        if covariate:
            self.fc = nn.Linear(lstm_num_hidden+covariate_size, input_size)
        else:
            self.fc = nn.Linear(lstm_num_hidden, input_size)
        self.input_len = input_len
        self.output_len = output_len

    def forward(self, x, covariates=None):
        # x: (batch_size, input_len, input_size)
        _, (h, c) = self.encoder(x)
        xt = x[:, -1, -1].unsqueeze(1).unsqueeze(2)  # input the last time step of x into the decoder
        outputs = []
        for _ in range(self.output_len):
            output, (h, c) = self.decoder(xt, (h, c))
            output = self.fc(output)
            outputs.append(output)
            xt = output  # use the decoder output as the next input

        outputs = torch.cat(outputs, dim=1)
        return outputs