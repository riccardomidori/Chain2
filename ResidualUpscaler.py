import torch
import torch.nn as nn
import lightning as pl

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResidualUpscaler(pl.LightningModule):
    def __init__(self, input_dim=1, hidden_dim: int = 64, num_blocks: int = 6, lr: float = 1e-3, method="regression"):
        super().__init__()
        self.save_hyperparameters()

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock1D(hidden_dim) for _ in range(num_blocks)])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

        self.loss_fn = nn.MSELoss()

    def forward(self, interpolated, target=None, mask=None):
        """
        interpolated: [B, seq_len, 1]
        """
        x = interpolated.permute(0, 2, 1)   # [B, 1, seq_len]
        x = self.input_proj(x)              # [B, hidden_dim, seq_len]
        x = self.res_blocks(x)              # [B, hidden_dim, seq_len]
        residual = self.output_proj(x)      # [B, 1, seq_len]
        residual = residual.permute(0, 2, 1)
        return residual

    def training_step(self, batch, batch_idx):
        interpolated, ned, ts = batch
        residual = self(interpolated)
        pred = interpolated + residual
        loss = self.loss_fn(pred, ned)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        interpolated, ned, ts = batch
        residual = self(interpolated)
        pred = interpolated + residual
        loss = self.loss_fn(pred, ned)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
