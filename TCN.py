import torch
import torch.nn as nn
import lightning as pl


class TCNResidualBlock(nn.Module):
    """
    A single Residual Block for a Temporal Convolutional Network (TCN).
    It uses Dilated Causal Convolutions (though standard Conv1d is often used
    for sequence-to-sequence tasks where masking isn't strictly necessary).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # We use a padding calculation that ensures the output sequence length
        # matches the input length after two Conv1d layers.
        padding = ((kernel_size - 1) * dilation) // 2

        # --- First Layer ---
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        # --- Second Layer ---
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        # --- Skip Connection (1x1 Conv if channels don't match) ---
        self.downsample = nn.Identity()
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Save input for residual connection
        res = x
        # Pass through block layers
        out = self.conv1(x)
        # Crop the output to remove the excess padding (crucial for Causal TCNs,
        # but often omitted if the sequence length is kept consistent by other means)
        # Here we rely on standard padding to keep length consistent.
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        # Add residual connection
        out = out + self.downsample(res)
        return out


class TCNResidualUpscaler(pl.LightningModule):
    def __init__(
        self,
        input_dim=1,
        hidden_dim: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        lr: float = 1e-3,
        method="regression",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.save_hyperparameters()

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # TCN Levels with exponentially increasing dilation
        tcn_blocks = []
        for i in range(num_levels):
            # Dilation factors: 1, 2, 4, 8, ...
            dilation = 2**i
            tcn_blocks.append(
                TCNResidualBlock(
                    hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation
                )
            )

        # Stack the TCN blocks
        self.tcn_levels = nn.Sequential(*tcn_blocks)

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.loss_fn = nn.MSELoss()

    def forward(self, interpolated, target=None, mask=None):
        """
        interpolated: [B, seq_len, 1]
        """
        x = interpolated.permute(0, 2, 1)  # [B, 1, seq_len]
        x = self.input_proj(x)  # [B, hidden_dim, seq_len]

        # TCN processing
        x = self.tcn_levels(x)  # [B, hidden_dim, seq_len]

        residual = self.output_proj(x)  # [B, 1, seq_len]
        residual = residual.permute(0, 2, 1)
        return residual

    def training_step(self, batch, batch_idx):
        interpolated, ned = batch
        residual = self(interpolated)
        pred = interpolated + residual
        loss = self.loss_fn(pred, ned)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        interpolated, ned = batch
        residual = self(interpolated)
        pred = interpolated + residual
        loss = self.loss_fn(pred, ned)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    """
        def training_step(self, batch, batch_idx, threshold=0.5):
        interpolated, ned, ts = batch
        spike_mask = (torch.abs(ned - interpolated) > threshold).float()
        weight_spike = 10.0
        weights = (spike_mask * weight_spike) + 1.0

        residual = self(interpolated)
        pred = interpolated + residual
        loss = torch.mean(weights * (pred - ned) ** 2)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, threshold=0.5):
        interpolated, ned, ts = batch
        spike_mask = (torch.abs(ned - interpolated) > threshold).float()
        weight_spike = 10.0
        weights = (spike_mask * weight_spike) + 1.0

        residual = self(interpolated)
        pred = interpolated + residual
        loss = torch.mean(weights * (pred - ned) ** 2)
        self.log("val_loss", loss)
        return loss
    """

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
