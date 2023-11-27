import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output[:, -1, :])
        return prediction, hidden, cell
        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target_len):
        batch_size = source.size(0)
        hidden, cell = self.encoder(source)

        decoder_input = torch.zeros(batch_size, 1, 1, device=source.device)  # Starting input for decoder
        outputs = torch.zeros(batch_size, target_len, 1, device=source.device)

        for t in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.view(batch_size, 1)
            decoder_input = output.view(batch_size, 1, 1)

        return outputs