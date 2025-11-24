import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl


class DoubleConv(nn.Module):
    """
    (Conv1d => BN => GELU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # We use simple interpolation for upsampling (lighter on memory)
        # followed by a convolution to smooth artifacts
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Input from previous decoder layer (Low Res)
        x2: Skip connection from encoder (High Res)
        """
        x1 = self.up(x1)

        # Handle cases where input size is not perfectly divisible by 2
        # Padding: [left, right]
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        # Concatenate along channel axis (dim 1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetUpscaler(pl.LightningModule):
    def __init__(self, in_channels=2, out_channels=1, lr=1e-3):
        """
        in_channels: 2 (Interpolated Signal + Mask)
        out_channels: 1 (Residual Correction)
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.loss_fn = nn.MSELoss()  # Or nn.L1Loss() for sharper spikes

        # --- U-Net Architecture ---
        # 1. Inc (Input Conv)
        self.inc = DoubleConv(in_channels, 64)

        # 2. Downsampling Path (Encoder)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # 3. Bottleneck (Lowest Resolution)
        # 1800 -> 900 -> 450 -> 225 -> ~112
        self.down4 = Down(512, 1024)

        # 4. Upsampling Path (Decoder)
        self.up1 = Up(1024 + 512, 512)  # In channels = 1024 (from up) + 512 (skip)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        # 5. Output Projector
        self.outc = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, 2, Seq_Len]

        # --- 1. Dynamic Padding ---
        # U-Net requires size to be divisible by 2^4 (16).
        # If seq_len=1800, 1800%16 != 0. We pad to 1808.
        original_len = x.shape[-1]
        divisor = 16
        pad_len = (divisor - original_len % divisor) % divisor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # Pad right side only

        # --- 2. Encoder ---
        x1 = self.inc(x)  # [B, 64, L]
        x2 = self.down1(x1)  # [B, 128, L/2]
        x3 = self.down2(x2)  # [B, 256, L/4]
        x4 = self.down3(x3)  # [B, 512, L/8]
        x5 = self.down4(x4)  # [B, 1024, L/16]

        # --- 3. Decoder ---
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # --- 4. Output Head ---
        residual = self.outc(x)

        # --- 5. Crop Padding ---
        if pad_len > 0:
            residual = residual[..., :original_len]

        return residual

    def training_step(self, batch, batch_idx):
        interpolated, target, mask = batch

        # Permute: [B, L, 1] -> [B, 1, L]
        input_signal = interpolated.permute(0, 2, 1)
        input_mask = mask.permute(0, 2, 1)
        target_signal = target.permute(0, 2, 1)

        # Concat inputs
        x = torch.cat([input_signal, input_mask], dim=1)  # [B, 2, L]

        # Predict Residual
        residual = self(x)

        # Final Prediction = Baseline + Residual
        prediction = input_signal + residual

        loss = self.loss_fn(prediction, target_signal)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        interpolated, target, mask = batch

        input_signal = interpolated.permute(0, 2, 1)
        input_mask = mask.permute(0, 2, 1)
        target_signal = target.permute(0, 2, 1)

        x = torch.cat([input_signal, input_mask], dim=1)

        residual = self(x)
        prediction = input_signal + residual

        loss = self.loss_fn(prediction, target_signal)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # Optional: Reduce LR on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
