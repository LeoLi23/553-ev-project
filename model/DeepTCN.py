import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dilated_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.dilated_conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dilated_conv2(out)
        out = self.batch_norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(in_channels, out_channels)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.dense2 = nn.Linear(out_channels, out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.dense1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dense2(out)
        out = self.batch_norm2(out)
        return out


class DeepTCN(nn.Module):
    def __init__(self, num_series, num_blocks, kernel_size, hidden_channels, num_covariates):
        super(DeepTCN, self).__init__()
        self.name = 'DeepTCN'
        self.num_covariates = num_covariates
        layers = []
        for i in range(num_blocks):
            dilation_size = 2 ** i
            in_channels = num_series if i == 0 else hidden_channels
            layers += [ResidualBlock(in_channels, hidden_channels, kernel_size, dilation_size)]

        self.encoder = nn.Sequential(*layers)
        self.decoder = Decoder(hidden_channels + num_covariates, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, x, covariates):
        # Encoder
        x = x.permute(0, 2, 1)  # [batch, seq_len, features] -> [batch, features, seq_len]
        x = self.encoder(x) # Output shape: [batch, hidden_channels, seq_len]
        # Average pooling over the sequence length dimension
        x = x.mean(dim=2)
        # Decoder
        outputs = []
        for seq_len_idx in range(covariates.shape[1]):
            current_covariate = covariates[:, seq_len_idx, :]
            combined = torch.cat((x, current_covariate), dim=1)
            decoded = self.decoder(combined)
            outputs.append(decoded)

        outputs = torch.stack(outputs, dim=1)
        outputs = self.output_layer(outputs)
        outputs = outputs.squeeze(-1)
        return outputs