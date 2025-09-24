import torch
import torch.nn as nn
import lightning as pl
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UNetUpscaler(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,  # (power, time_delta)
        base_channels: int = 32,
        depth: int = 4,
        output_seq_len: int = 120,
        lr: float = 1e-3,
        method="regression",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = [input_dim]  # Track channel progression
        pool_size = output_seq_len
        # Build encoder layers
        for d in range(depth):
            in_ch = channels[-1]
            out_ch = base_channels * (2**d)
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool1d(2))
            channels.append(out_ch)
            pool_size = pool_size // 2
            if pool_size == 1:
                print(f"Stopped at depth {d} for pool_size reached 1")
                depth = d
                break

        # Bottleneck
        bottleneck_in = channels[-1]
        bottleneck_out = bottleneck_in * 2
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_in = bottleneck_out
        for d in reversed(range(depth)):
            skip_ch = channels[d + 1]  # Channels from corresponding encoder layer
            decoder_out = base_channels * (2**d)

            # Upconv reduces channels
            self.upconvs.append(
                nn.ConvTranspose1d(decoder_in, decoder_out, kernel_size=2, stride=2)
            )

            # Decoder block takes upconv output + skip connection
            self.decoders.append(ConvBlock(decoder_out + skip_ch, decoder_out))

            decoder_in = decoder_out

        # Output projection: map back to 1 channel (power)
        self.output_fc = nn.Conv1d(base_channels, 1, kernel_size=1)

        # Metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

        self.lr = lr
        self.output_seq_len = output_seq_len

    def forward(self, power, time_deltas, mask):
        # Input prep: concat features along channel dimension
        x = power
        # x = torch.cat([power, time_deltas], dim=-1)  # [B, seq_len, 2]
        x = x.permute(0, 2, 1)  # [B, C=2, seq_len]

        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Align shapes (due to pooling rounding)
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                x = nn.functional.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Output
        out = self.output_fc(x)  # [B, 1, seq_len]
        out = out.permute(0, 2, 1)  # [B, seq_len, 1]

        # If needed, crop or interpolate to output_seq_len
        if out.size(1) != self.output_seq_len:
            out = nn.functional.interpolate(
                out.transpose(1, 2),
                size=self.output_seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return out

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, False)

    def do_step(self, batch, batch_idx, is_train=True):
        power, y, ts = batch
        y_hat = self(power, None, None)
        loss = nn.MSELoss()(y_hat, y)
        if is_train:
            self.train_mse(y_hat.squeeze(-1), y.squeeze(-1))
            self.train_mae(y_hat.squeeze(-1), y.squeeze(-1))
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_mse", self.train_mse, prog_bar=True)
            self.log("train_mae", self.train_mae)
        else:
            self.val_mse(y_hat.squeeze(-1), y.squeeze(-1))
            self.val_mae(y_hat.squeeze(-1), y.squeeze(-1))
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_mse", self.val_mse, prog_bar=True)
            self.log("val_mae", self.val_mae)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
