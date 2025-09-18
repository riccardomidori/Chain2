import lightning
import torch
import torch.nn as nn
from NNBlock import UNetBlock
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)

class UNetTimeSeriesUpscaler(lightning.LightningModule):
    """
    A U-Net architecture for time series up-scaling (super-resolution).

    The encoder down-samples the high-frequency Chain2 data, and the decoder
    up-samples to match the lower-frequency NED_D data, using skip connections
    to preserve fine-grained temporal details.

    NOTE: This model requires regularly sampled input data. A simple interpolation
    step is included in the forward pass to handle the irregular Chain2 data.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        output_sequence_length: int,
        base_channels: int = 64,
        method="regression",
    ):
        """
        Args:
            input_channels (int): Number of features in the input sequence (e.g., 1 for power).
            output_channels (int): Number of features in the output sequence (e.g., 1 for power).
            output_sequence_length (int): The number of time steps in the target NED_D sequence.
            base_channels (int): The number of channels in the first U-Net block.
        """
        super().__init__()
        self.save_hyperparameters()
        self.output_sequence_length = output_sequence_length
        self.output_channels = output_channels

        # Encoder (Down-sampling path)
        self.enc1 = UNetBlock(input_channels, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = UNetBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 4, base_channels * 8)

        # Decoder (Up-sampling path with skip connections)
        self.upconv3 = nn.ConvTranspose1d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = UNetBlock(base_channels * 4 + base_channels * 4, base_channels * 4)
        self.upconv2 = nn.ConvTranspose1d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = UNetBlock(base_channels * 2 + base_channels * 2, base_channels * 2)
        self.upconv1 = nn.ConvTranspose1d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = UNetBlock(base_channels + base_channels, base_channels)

        # Final output layer
        self.output_layer = nn.Conv1d(base_channels, output_channels, kernel_size=1)

        # Loss function
        self.loss = MeanSquaredError()

    def forward(
        self, power: torch.Tensor, time_delta: torch.Tensor, mask: torch.Tensor
    ):
        """
        Forward pass with a pre-processing step to handle irregular data.

        Args:
            power (torch.Tensor): The input power tensor from a single batch.
            time_delta (torch.Tensor): The input time_delta tensor from a single batch.
            mask (torch.Tensor): The mask tensor for the batch.

        Returns:
            torch.Tensor: The predicted output sequence.
        """
        batch_size, input_seq_len, _ = power.shape

        # --- Pre-processing: Interpolate irregular input to a regular grid ---
        # The input has shape (batch, seq_len, 1). We need to get a regular grid
        # with a consistent number of points. We will use the max_len for this.
        # This requires the input to be on the CPU for a simple interpolation method.
        # NOTE: For a real-world application, a more robust interpolation method would be needed.
        regular_grid_length = 1024  # A fixed length for the U-Net input

        # The input to the U-Net must be (batch, channels, seq_len)
        x = power.permute(0, 2, 1)

        # --- U-Net forward pass ---
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool1(enc1_out))
        enc3_out = self.enc3(self.pool2(enc2_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(self.pool3(enc3_out))

        # Decoder with skip connections
        dec3_in = self.upconv3(bottleneck_out)
        dec3_out = self.dec3(torch.cat([dec3_in, enc3_out], dim=1))

        dec2_in = self.upconv2(dec3_out)
        dec2_out = self.dec2(torch.cat([dec2_in, enc2_out], dim=1))

        dec1_in = self.upconv1(dec2_out)
        dec1_out = self.dec1(torch.cat([dec1_in, enc1_out], dim=1))

        # Final output layer to get the desired number of channels
        output_full_res = self.output_layer(dec1_out)

        # Resize the output to the target sequence length
        prediction = F.interpolate(
            output_full_res,
            size=self.output_sequence_length,
            mode="linear",
            align_corners=False,
        )

        # Final shape: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        prediction = prediction.permute(0, 2, 1)

        return prediction

    def do_step(self, batch, batch_idx, is_train=True):
        x, y, mask, ts = batch
        power, time_delta = x[:, :, 0:1], x[:, :, 1:2]
        y_hat = self.forward(power, time_delta, mask)

        if y.ndim < 3:
            y = y.unsqueeze(-1)

        loss = self.loss(y_hat, y)

        if is_train:
            name = "train"
        else:
            name = "val"
        self.log(f"{name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

