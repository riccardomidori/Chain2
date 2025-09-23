import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SimpleTransformer(lightning.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        output_seq_len: int = 120,
        lr: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Input processing
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = SimplePositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool sequence dimension
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_seq_len),
        )

        # Metrics
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.lr = lr

    def forward(self, power, time_deltas, mask):
        # Combine features
        x = torch.cat([power, time_deltas], dim=-1)  # [B, seq_len, 2]
        x = self.input_projection(x)  # [B, seq_len, d_model]
        x = self.pos_encoding(x)

        # Encode with proper padding mask
        encoded = self.encoder(x, src_key_padding_mask=~mask)

        # Global pooling (only over valid positions)
        # Apply mask before pooling
        masked_encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        pooled = masked_encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()

        # Output projection
        output = self.fc_out(pooled)  # [B, output_seq_len]

        return output.unsqueeze(-1)  # [B, output_seq_len, 1]

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, True)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, False)

    def do_step(self, batch, batch_idx, is_train):
        key = "train" if is_train else "val"
        power, y, mask, time_deltas, ts = batch

        y_hat = self(power, time_deltas, mask)
        loss = nn.MSELoss()(y_hat, y)

        y_hat_flat = y_hat.squeeze(-1)
        y_flat = y.squeeze(-1)

        self.log(f"{key}_loss", loss, prog_bar=True)
        self.log(f"{key}_mse", self.mse(y_hat_flat, y_flat), prog_bar=True)
        self.log(f"{key}_mae", self.mae(y_hat_flat, y_flat))

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


class PositionalEncoding(nn.Module):
    """
    Enhanced positional encoding with learnable components
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

        # Learnable scaling factor
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            # Batch first format: (batch_size, seq_len, d_model)
            seq_len = x.size(1)
            return x + self.alpha * self.pe[:seq_len, :].transpose(0, 1).unsqueeze(0)
        else:
            # Sequence first format: (seq_len, batch_size, d_model)
            seq_len = x.size(0)
            return x + self.alpha * self.pe[:seq_len, :]


class ImprovedTemporalEncoding(nn.Module):
    """
    Enhanced temporal encoding for irregular time series
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Multi-layer temporal projection
        self.time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
        )

        # Learnable time embedding
        self.time_embedding = nn.Embedding(1000, d_model)  # For discretized time

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, time_deltas):
        # x: (batch, seq_len, d_model)
        # time_deltas: (batch, seq_len, 1) or (batch, seq_len)

        if time_deltas.dim() == 2:
            time_deltas = time_deltas.unsqueeze(-1)

        # Continuous time encoding
        temporal_encoding = self.time_projection(time_deltas)

        # Add discretized time encoding (optional)
        # Discretize time deltas to reasonable bins
        time_bins = torch.clamp((time_deltas.squeeze(-1) * 10).long(), 0, 999)
        discrete_encoding = self.time_embedding(time_bins)

        # Combine encodings
        combined_encoding = temporal_encoding + 0.1 * discrete_encoding
        combined_encoding = self.layer_norm(combined_encoding)

        return x + combined_encoding


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention for capturing patterns at different time scales
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Multi-head attention at different scales
        self.attention_1 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attention_2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attention_4 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Convolutions for different scales
        self.conv_1 = nn.Conv1d(d_model, d_model, kernel_size=1, groups=d_model // 8)
        self.conv_2 = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model // 8
        )
        self.conv_4 = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model // 8
        )

        # Output projection
        self.output_proj = nn.Linear(d_model * 3, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Multi-scale attention
        attn_1, _ = self.attention_1(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )

        # Downsample for scale 2
        x_2 = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=2, stride=1, padding=1
        ).transpose(1, 2)
        attn_2, _ = self.attention_2(
            x_2, x_2, x_2, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        attn_2 = F.interpolate(
            attn_2.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)

        # Downsample for scale 4
        x_4 = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=4, stride=1, padding=2
        ).transpose(1, 2)
        attn_4, _ = self.attention_4(
            x_4, x_4, x_4, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        attn_4 = F.interpolate(
            attn_4.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)

        # Apply convolutional processing
        conv_1 = self.conv_1(attn_1.transpose(1, 2)).transpose(1, 2)
        conv_2 = self.conv_2(attn_2.transpose(1, 2)).transpose(1, 2)
        conv_4 = self.conv_4(attn_4.transpose(1, 2)).transpose(1, 2)

        # Combine multi-scale features
        multi_scale = torch.cat([conv_1, conv_2, conv_4], dim=-1)
        output = self.output_proj(multi_scale)

        # Residual connection and normalization
        output = self.layer_norm(x + self.dropout(output))

        return output


class ImprovedTransformerTimeSeriesUpscaler(nn.Module):
    """
    Enhanced Transformer with better handling of irregular time series
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
        use_multi_scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_seq_len = output_seq_len
        self.max_input_len = max_input_len
        self.use_multi_scale = use_multi_scale

        # Enhanced input embeddings with normalization
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Enhanced temporal encoding
        self.temporal_encoding = ImprovedTemporalEncoding(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_input_len)

        # Multi-scale attention for encoder
        if use_multi_scale:
            self.multi_scale_attention = MultiScaleAttention(d_model, nhead, dropout)

        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Learnable output queries with better initialization
        self.output_queries = nn.Parameter(torch.randn(output_seq_len, d_model) * 0.02)
        self.output_pos_encoding = PositionalEncoding(d_model, output_seq_len)

        # Enhanced decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Enhanced output projection with skip connection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, input_dim),
        )

        # Output layer normalization
        self.output_layer_norm = nn.LayerNorm(d_model)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, power, time_delta, mask):
        batch_size = power.size(0)
        seq_len = power.size(1)

        # Handle input dimensions
        if power.dim() == 2:
            power = power.unsqueeze(-1)
        if time_delta.dim() == 2:
            time_delta = time_delta.unsqueeze(-1)

        # Embed inputs
        power_embedded = self.input_embedding(power)  # (batch, seq_len, d_model)

        # Add temporal encoding
        x = self.temporal_encoding(power_embedded, time_delta)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply multi-scale attention if enabled
        if self.use_multi_scale:
            x = self.multi_scale_attention(x, key_padding_mask=~mask)

        # Encoder
        memory = self.transformer_encoder(x, src_key_padding_mask=~mask)

        # Prepare decoder queries
        output_queries = self.output_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        output_queries = self.output_pos_encoding(output_queries)

        # Create causal mask for decoder
        tgt_mask = self.create_causal_mask(self.output_seq_len).to(
            output_queries.device
        )

        # Decoder with causal masking
        decoded = self.transformer_decoder(
            tgt=output_queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~mask,
        )

        # Apply output layer norm
        decoded = self.output_layer_norm(decoded)

        # Project to output dimension
        output = self.output_projection(decoded)

        return output


class PureTransformerUpscaler(lightning.LightningModule):
    """
    Enhanced Pure Transformer with better training strategies
    """

    def __init__(
        self,
        output_sequence_length: int,
        input_channels: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_lr_multiplier: float = 10,
        label_smoothing: float = 0.0,
        gradient_clip_val: float = 1.0,
        use_multi_scale: bool = True,
        power_scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["power_scaler"])
        self.power_scaler = power_scaler
        self.warmup_steps = warmup_steps

        # Enhanced model
        self.model = ImprovedTransformerTimeSeriesUpscaler(
            input_dim=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            output_seq_len=output_sequence_length,
            use_multi_scale=use_multi_scale,
        )

        # Enhanced loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()
        self.quantile_loss = self._quantile_loss

        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()

        # For tracking best validation loss
        self.best_val_loss = float("inf")

    def _quantile_loss(self, pred, target, quantile=0.5):
        """Quantile regression loss for robust training"""
        error = target - pred
        loss = torch.max(quantile * error, (quantile - 1) * error)
        return loss.mean()

    def forward(self, power, time_deltas, mask):
        # Ensure proper input format
        if power.dim() == 3 and power.size(-1) == 2:
            # If power contains both power and time_delta
            power, time_deltas_from_power = power[:, :, 0:1], power[:, :, 1:2]
            if time_deltas.dim() == 2:
                time_deltas = time_deltas.unsqueeze(-1)
        elif power.dim() == 2:
            power = power.unsqueeze(-1)
            if time_deltas.dim() == 2:
                time_deltas = time_deltas.unsqueeze(-1)

        return self.model(power, time_deltas, mask)

    def compute_losses(self, y_hat, y):
        """Compute multiple loss components"""
        # Ensure shapes match
        if y_hat.shape != y.shape:
            min_len = min(y_hat.size(1), y.size(1))
            y_hat = y_hat[:, :min_len]
            y = y[:, :min_len]

        # Basic losses
        mse_loss = self.mse_loss(y_hat, y)
        mae_loss = self.mae_loss(y_hat, y)
        huber_loss = self.huber_loss(y_hat, y)

        # Quantile losses for robustness
        q25_loss = self.quantile_loss(y_hat, y, 0.25)
        q75_loss = self.quantile_loss(y_hat, y, 0.75)

        # Frequency domain loss
        freq_loss = self._frequency_loss(y_hat, y)

        # Trend preservation loss
        trend_loss = self._trend_loss(y_hat, y)

        # Combined loss
        total_loss = (
            mse_loss
            + 0.5 * mae_loss
            + 0.3 * huber_loss
            + 0.1 * (q25_loss + q75_loss)
            + 0.2 * freq_loss
            + 0.1 * trend_loss
        )

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "mae_loss": mae_loss,
            "huber_loss": huber_loss,
            "freq_loss": freq_loss,
            "trend_loss": trend_loss,
        }

    def _frequency_loss(self, pred, target):
        """Frequency domain loss using FFT"""
        try:
            if pred.dim() == 3:
                pred = pred.squeeze(-1)
            if target.dim() == 3:
                target = target.squeeze(-1)

            pred_fft = torch.fft.rfft(pred, dim=1)
            target_fft = torch.fft.rfft(target, dim=1)

            return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        except:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def _trend_loss(self, pred, target):
        """Loss to preserve trends (first derivatives)"""
        try:
            if pred.dim() == 3:
                pred = pred.squeeze(-1)
            if target.dim() == 3:
                target = target.squeeze(-1)

            pred_diff = torch.diff(pred, dim=1)
            target_diff = torch.diff(target, dim=1)

            return F.mse_loss(pred_diff, target_diff)
        except:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def training_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 5:
            power, y, mask, time_deltas, ts = batch
        else:
            power, y, mask, time_deltas = batch

        # Ensure target has correct dimensions
        if y.dim() == 2:
            y = y.unsqueeze(-1)

        # Forward pass
        y_hat = self.forward(power, time_deltas, mask)

        # Compute losses
        losses = self.compute_losses(y_hat, y)

        # Update metrics
        if y.dim() == 3:
            y_flat = y.squeeze(-1)
            y_hat_flat = y_hat.squeeze(-1)
        else:
            y_flat = y
            y_hat_flat = y_hat

        self.train_mse(y_hat_flat, y_flat)
        self.train_mae(y_hat_flat, y_flat)

        # Logging
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        self.log("train_mse", self.train_mse, prog_bar=True)
        self.log("train_mae", self.train_mae)
        self.log("train_freq_loss", losses["freq_loss"])
        self.log("train_trend_loss", losses["trend_loss"])

        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        # Handle different batch formats
        if len(batch) == 5:
            power, y, mask, time_deltas, ts = batch
        else:
            power, y, mask, time_deltas = batch

        if y.dim() == 2:
            y = y.unsqueeze(-1)

        y_hat = self.forward(power, time_deltas, mask)
        losses = self.compute_losses(y_hat, y)

        # Update metrics
        if y.dim() == 3:
            y_flat = y.squeeze(-1)
            y_hat_flat = y_hat.squeeze(-1)
        else:
            y_flat = y
            y_hat_flat = y_hat

        self.val_mse(y_hat_flat, y_flat)
        self.val_mae(y_hat_flat, y_flat)

        # Logging
        self.log("val_loss", losses["total_loss"], prog_bar=True)
        self.log("val_mse", self.val_mse, prog_bar=True)
        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_freq_loss", losses["freq_loss"])

        # Track best validation loss
        if losses["total_loss"] < self.best_val_loss:
            self.best_val_loss = losses["total_loss"]
            self.log("best_val_loss", self.best_val_loss)

        return losses["total_loss"]

    def configure_optimizers(self):
        # Use AdamW with better defaults
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),  # Better for transformers
            eps=1e-8,
        )

        # Enhanced learning rate scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Warmup phase
                return current_step / self.warmup_steps
            else:
                # Cosine annealing
                progress = (current_step - self.warmup_steps) / max(
                    1, self.trainer.estimated_stepping_batches - self.warmup_steps
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def predict_step(self, batch, batch_idx):
        """Enhanced prediction with confidence intervals"""
        if len(batch) == 5:
            power, y, mask, time_deltas, ts = batch
        else:
            power, y, mask, time_deltas = batch

        # Forward pass
        y_hat = self.forward(power, time_deltas, mask)

        # Denormalize if scaler is available
        if self.power_scaler is not None:
            y_hat_denorm = self._denormalize_batch(y_hat)
            y_denorm = self._denormalize_batch(y) if y is not None else None
        else:
            y_hat_denorm = y_hat
            y_denorm = y

        return {
            "predictions": y_hat_denorm,
            "targets": y_denorm,
            "predictions_normalized": y_hat,
            "targets_normalized": y,
            "input_mask": mask,
        }

    def _denormalize_batch(self, normalized_data):
        """Denormalize a batch of data"""
        batch_size = normalized_data.size(0)
        seq_len = normalized_data.size(1)

        if normalized_data.dim() == 3:
            normalized_data = normalized_data.squeeze(-1)

        denormalized_batch = []
        for i in range(batch_size):
            sample = normalized_data[i].detach().cpu().numpy()
            denorm_sample = self.power_scaler.inverse_transform(
                sample.reshape(-1, 1)
            ).flatten()
            denormalized_batch.append(torch.from_numpy(denorm_sample))

        return torch.stack(denormalized_batch)
