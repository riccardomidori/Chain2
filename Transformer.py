from typing import Any

import lightning
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from NNBlock import PositionalEncoding, TemporalEncoding
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)


class TransformerTimeSeriesUpscaler(lightning.LightningModule):
    """
    Transformer-based architecture for time series upscaling
    Uses encoder-decoder structure with cross-attention
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        output_seq_len: int = 120,
        dropout: float = 0.1,
        max_input_len: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.output_seq_len = output_seq_len
        self.max_input_len = max_input_len

        # Input embeddings
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.time_delta_embedding = nn.Linear(1, d_model)

        # Temporal encoding (custom for irregular time series)
        self.temporal_encoding = TemporalEncoding(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_input_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Learnable output queries (for decoder)
        self.output_queries = nn.Parameter(torch.randn(output_seq_len, d_model))
        self.output_pos_encoding = PositionalEncoding(d_model, output_seq_len)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, input_dim),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, power, time_delta, mask):
        batch_size = power.size(0)
        # Embed inputs
        power_embedded = self.input_embedding(power)  # (batch, seq_len, d_model)
        time_embedded = self.time_delta_embedding(
            time_delta
        )  # (batch, seq_len, d_model)

        # Combine power and temporal information
        x = power_embedded + time_embedded

        # Add temporal encoding using actual time deltas
        x = self.temporal_encoding(x, time_delta)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Apply layer norm
        x = self.layer_norm(x)

        # Create attention mask for padded positions
        # Encoder
        memory = self.transformer_encoder(x, src_key_padding_mask=~mask)

        # Prepare decoder queries (learnable embeddings for output positions)
        output_queries = self.output_queries.unsqueeze(0).repeat(batch_size, 1, 1)

        # Add positional encoding to output queries
        output_queries = output_queries.transpose(
            0, 1
        )  # (output_seq_len, batch, d_model)
        output_queries = self.output_pos_encoding(output_queries)
        output_queries = output_queries.transpose(
            0, 1
        )  # (batch, output_seq_len, d_model)

        # Decoder with cross-attention to encoder memory
        decoded = self.transformer_decoder(
            tgt=output_queries,
            memory=memory,
            memory_key_padding_mask=~mask,
        )

        # Project to output dimension
        output = self.output_projection(decoded)  # (batch, output_seq_len, 1)

        return output


class PureTransformerUpscaler(lightning.LightningModule):
    """
    Pure transformer approach without adversarial training
    Often more stable and easier to train
    """

    def __init__(
        self,
        output_sequence_length: int,
        input_channels: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        method="regression",
        power_scaler=None,  # Pass the fitted scaler for de-normalization
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerTimeSeriesUpscaler(
            input_dim=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            output_seq_len=output_sequence_length,
        )

        # Multiple loss functions
        self.mse_loss = MeanSquaredError()
        self.mae_loss = MeanAbsoluteError()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, power, time_deltas, mask):
        # Use actual time deltas for temporal encoding (reshape for model)
        # Pass to transformer model
        output = self.model(power, time_deltas, mask)

        return output  # [batch, output_seq_len, 1]

    def do_step(self, batch, batch_idx, is_train=True):
        power, y, mask, time_deltas, ts = batch
        # Forward pass
        y_hat = self.forward(power, time_deltas, mask)

        # Ensure shapes match
        if y_hat.shape != y.shape:
            min_len = min(y_hat.size(1), y.size(1))
            y_hat = y_hat[:, :min_len]
            y = y[:, :min_len]

        # Calculate losses
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)

        # Combined loss with weights
        total_loss = mse_loss + 0.5 * mae_loss + 0.2 * huber_loss
        # Logging
        if is_train:
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("train_mse", mse_loss)
            self.log("train_mae", mae_loss)
        else:
            self.log("val_loss", total_loss, prog_bar=True)
            self.log("val_mse", mse_loss)
            self.log("val_mae", mae_loss)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr * 10,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
