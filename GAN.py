import lightning
import torch.nn.functional as F
import torch.nn as nn
import torch
from NNBlock import AttentionBlock, ResidualBlock, Interpolation
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)

class Discriminator(nn.Module):
    """
    Discriminator network. A simple 1D convolutional network to classify
    a sequence as real or fake.
    """

    def __init__(self, input_channels, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            base_channels, base_channels * 2, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            base_channels * 2, base_channels * 4, kernel_size=3, padding=1
        )

        self.fc = nn.Linear(base_channels * 4, 1)

    def forward(self, x):
        # x is (batch_size, seq_len, channels)
        # Reshape to (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and pass through a linear layer
        x = torch.mean(x, dim=2)
        return self.fc(x)


class Generator(nn.Module):
    """
    Enhanced Generator with skip connections, attention, and residual blocks
    """

    def __init__(
        self, input_channels, output_channels, base_channels=64, output_seq_len=120
    ):
        super().__init__()
        self.output_seq_len = output_seq_len

        # Encoder with skip connections
        self.enc1 = nn.Conv1d(input_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Conv1d(base_channels, base_channels * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck with residual blocks and attention
        self.bottleneck_conv = nn.Conv1d(
            base_channels * 4, base_channels * 8, 3, padding=1
        )
        self.bottleneck_bn = nn.BatchNorm1d(base_channels * 8)
        self.residual1 = ResidualBlock(base_channels * 8)
        self.residual2 = ResidualBlock(base_channels * 8)
        self.attention = AttentionBlock(base_channels * 8)

        # Decoder with skip connections
        self.upconv3 = nn.ConvTranspose1d(
            base_channels * 8, base_channels * 4, 2, stride=2
        )
        self.dec3 = nn.Conv1d(
            base_channels * 8, base_channels * 4, 3, padding=1
        )  # *8 due to skip
        self.bn_dec3 = nn.BatchNorm1d(base_channels * 4)

        self.upconv2 = nn.ConvTranspose1d(
            base_channels * 4, base_channels * 2, 2, stride=2
        )
        self.dec2 = nn.Conv1d(
            base_channels * 4, base_channels * 2, 3, padding=1
        )  # *4 due to skip
        self.bn_dec2 = nn.BatchNorm1d(base_channels * 2)

        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Conv1d(
            base_channels * 2, base_channels, 3, padding=1
        )  # *2 due to skip
        self.bn_dec1 = nn.BatchNorm1d(base_channels)

        # Final output layer
        self.final_conv = nn.Conv1d(base_channels, output_channels, 3, padding=1)

        # Dropout for regularization
        self.dropout = nn.Dropout1d(0.1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = F.relu(self.bn1(self.enc1(x)))
        pool1 = self.pool1(enc1)  # Store for skip connection

        enc2 = F.relu(self.bn2(self.enc2(pool1)))
        pool2 = self.pool2(enc2)  # Store for skip connection

        enc3 = F.relu(self.bn3(self.enc3(pool2)))
        pool3 = self.pool3(enc3)  # Store for skip connection

        # Bottleneck
        bottleneck = F.relu(self.bottleneck_bn(self.bottleneck_conv(pool3)))
        bottleneck = self.residual1(bottleneck)
        bottleneck = self.residual2(bottleneck)
        bottleneck = self.attention(bottleneck)
        bottleneck = self.dropout(bottleneck)

        # Decoder with skip connections
        up3 = F.relu(self.upconv3(bottleneck))
        # Concatenate skip connection
        up3 = torch.cat([up3, enc3], dim=1)
        dec3 = F.relu(self.bn_dec3(self.dec3(up3)))

        up2 = F.relu(self.upconv2(dec3))
        up2 = torch.cat([up2, enc2], dim=1)
        dec2 = F.relu(self.bn_dec2(self.dec2(up2)))

        up1 = F.relu(self.upconv1(dec2))
        up1 = torch.cat([up1, enc1], dim=1)
        dec1 = F.relu(self.bn_dec1(self.dec1(up1)))

        # Final output
        output = self.final_conv(dec1)

        # Resize to target length
        if output.size(2) != self.output_seq_len:
            output = F.interpolate(
                output, size=self.output_seq_len, mode="linear", align_corners=False
            )

        # Permute to (batch, seq_len, channels)
        return output.permute(0, 2, 1)

class GANTimeSeriesUpscaler(lightning.LightningModule):
    def __init__(
        self,
        output_sequence_length: int,
        input_channels: int = 1,
        output_channels: int = 1,
        lr_gen: float = 0.0001,
        lr_disc: float = 0.0004,
        b1: float = 0.5,
        b2: float = 0.999,
        lambda_recon: float = 100,
        lambda_freq: float = 10,
        lambda_grad: float = 10,
        method="regression",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Use enhanced generator
        self.generator = Generator(
            input_channels, output_channels, output_seq_len=output_sequence_length
        )
        self.discriminator = Discriminator(output_channels)

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = MeanSquaredError()
        self.mae_loss = MeanAbsoluteError()

        # For tracking discriminator accuracy
        self.disc_real_acc = []
        self.disc_fake_acc = []

    def frequency_domain_loss(self, pred, target):
        """Frequency domain loss using FFT"""
        pred_fft = torch.fft.rfft(pred.squeeze(-1))
        target_fft = torch.fft.rfft(target.squeeze(-1))

        return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

    def gradient_penalty(self, real_data, fake_data):
        """Gradient penalty for WGAN-GP style training"""
        batch_size = real_data.size(0)

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Get discriminator output
        d_interpolated = self.discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, power, time_delta, mask):
        # Use improved interpolation
        x_interpolated = Interpolation.adaptive_interpolation(
            power, time_delta, mask, target_length=1024
        )
        return self.generator(x_interpolated)

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        power, y, mask, time_delta, ts = batch

        if y.ndim < 3:
            y = y.unsqueeze(-1)

        # Generate fake data
        y_hat = self.forward(power, time_delta, mask)

        # Train Generator every 2 steps (to balance with discriminator)
        if batch_idx % 2 == 0:
            gen_opt.zero_grad()

            # Adversarial loss
            fake_pred = self.discriminator(y_hat)
            g_adv_loss = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))

            # Reconstruction losses
            g_recon_loss = self.reconstruction_loss(y_hat, y)
            g_mae_loss = self.mae_loss(y_hat, y)

            # Frequency domain loss
            g_freq_loss = self.frequency_domain_loss(y_hat, y)

            # Combined generator loss
            g_total_loss = (
                g_adv_loss
                + self.hparams.lambda_recon * g_recon_loss
                + self.hparams.lambda_recon * g_mae_loss
                + self.hparams.lambda_freq * g_freq_loss
            )

            self.manual_backward(g_total_loss)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

            gen_opt.step()

            self.log("g_loss", g_total_loss, prog_bar=True)
            self.log("g_adv_loss", g_adv_loss)
            self.log("g_recon_loss", g_recon_loss)
            self.log("g_freq_loss", g_freq_loss)

        # Train Discriminator
        disc_opt.zero_grad()

        # Real loss
        real_pred = self.discriminator(y)
        d_real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))

        # Fake loss
        fake_pred = self.discriminator(y_hat.detach())
        d_fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))

        # Gradient penalty
        gp = self.gradient_penalty(y, y_hat.detach())

        # Total discriminator loss
        d_total_loss = (d_real_loss + d_fake_loss) / 2 + self.hparams.lambda_grad * gp

        self.manual_backward(d_total_loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

        disc_opt.step()

        self.log("d_loss", d_total_loss, prog_bar=True)
        self.log("d_real_loss", d_real_loss)
        self.log("d_fake_loss", d_fake_loss)
        self.log("gradient_penalty", gp)

        # Track discriminator accuracy
        real_acc = (torch.sigmoid(real_pred) > 0.5).float().mean()
        fake_acc = (torch.sigmoid(fake_pred) < 0.5).float().mean()
        self.log("d_real_acc", real_acc)
        self.log("d_fake_acc", fake_acc)

    def validation_step(self, batch, batch_idx):
        power, y, mask, time_delta, ts = batch
        y_hat = self.forward(power, time_delta, mask)

        if y.ndim < 3:
            y = y.unsqueeze(-1)

        # Validation losses
        val_recon_loss = self.reconstruction_loss(y_hat, y)
        val_mae_loss = self.mae_loss(y_hat, y)
        val_freq_loss = self.frequency_domain_loss(y_hat, y)

        self.log("val_recon_loss", val_recon_loss, prog_bar=True)
        self.log("val_mae_loss", val_mae_loss, prog_bar=True)
        self.log("val_freq_loss", val_freq_loss)

        return {"val_loss": val_recon_loss}

    def configure_optimizers(self):
        # Different learning rates for generator and discriminator
        gen_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr_gen,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr_disc,
            betas=(self.hparams.b1, self.hparams.b2),
        )

        # Learning rate schedulers
        gen_scheduler = torch.optim.lr_scheduler.StepLR(
            gen_optimizer, step_size=50, gamma=0.8
        )
        disc_scheduler = torch.optim.lr_scheduler.StepLR(
            disc_optimizer, step_size=50, gamma=0.8
        )

        return [gen_optimizer, disc_optimizer], [gen_scheduler, disc_scheduler]
