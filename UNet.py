import torch
import torch.nn as nn
import lightning as pl


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # Padding must handle dilation to keep output size same as input
        padding = (kernel_size - 1) * dilation // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),  # GELU is often preferred over ReLU for time-series
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualUpscaler(pl.LightningModule):
    def __init__(self, input_dim=2, hidden_dim=64, num_blocks=6, lr=1e-3, method="regression"):
        """
        input_dim=2: Channel 0 = Interpolated Signal, Channel 1 = Binary Mask (1=Real, 0=Interp)
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # Exponential dilation: 1, 2, 4, 8, 16, 32...
        # This expands the receptive field exponentially.
        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(hidden_dim, dilation=2 ** i) for i in range(num_blocks)
        ])

        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x shape: [Batch, 2, Seq_Len] -> (Interpolated, Mask)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        residual = self.output_proj(x)
        return residual

    def common_step(self, batch, batch_idx):
        # Unpack the tuple from the loader
        # We need the Mask now!
        interpolated, target, mask = batch

        # 1. Permute everything to [Batch, Channels, Seq_Len]
        # PyTorch Conv1d expects the channel dimension to be second.
        input_signal = interpolated.permute(0, 2, 1)  # [B, 1, Seq]
        input_mask = mask.permute(0, 2, 1)  # [B, 1, Seq]
        target_permuted = target.permute(0, 2, 1)  # [B, 1, Seq]

        # 2. Concatenate signal and mask along the channel dimension
        # Result shape: [B, 2, Seq]
        x = torch.cat([input_signal, input_mask], dim=1)

        # 3. Forward pass
        residual = self(x)  # Output: [B, 1, Seq]

        # 4. Residual Connection
        prediction = input_signal + residual

        # 5. Calculate Loss
        loss = self.loss_fn(prediction, target_permuted)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)