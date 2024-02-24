import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden, cell):
        outputs = []
        for _ in range(2):  # output_seq_len is 2
            output, (hidden, cell) = self.lstm(x, (hidden, cell))
            output = self.fc(output[:, -1, :])
            outputs.append(output)
            x = output.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden, cell

        
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source):
        batch_size = source.size(0)
        hidden, cell = self.encoder(source)

        decoder_input = torch.zeros(batch_size, 1, 1, device=source.device)  # Starting input for decoder
        outputs, _, _ = self.decoder(decoder_input, hidden, cell)  # Get the full output sequence

        return outputs