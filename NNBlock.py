import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.interpolate import interp1d


class TemporalEncoding(nn.Module):
    """
    Custom temporal encoding that incorporates actual time deltas
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.time_projection = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
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

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal relationships in irregular time series
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[: x.size(0), :]


class AttentionBlock(nn.Module):
    """Self-attention for time series features"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()

        # Compute attention
        Q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        K = self.key(x).view(batch_size, -1, length)
        V = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)

        attention = torch.bmm(Q, K)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(attention, V).permute(0, 2, 1)
        out = out.view(batch_size, channels, length)

        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class UNetBlock(nn.Module):
    """
    A single down-sampling or up-sampling block in the U-Net.
    Composed of two 1D convolutional layers followed by a ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Interpolation:
    @staticmethod
    def spline_interpolation(power, time_delta, mask, target_length=1024):
        """Use spline interpolation instead of linear"""
        interpolated_list = []

        for i in range(power.shape[0]):
            valid_indices = torch.where(mask[i] == False)[0]

            if len(valid_indices) < 2:
                interpolated_list.append(
                    torch.zeros(target_length, device=power.device)
                )
                continue

            valid_power = power[i, valid_indices, 0].cpu().numpy()
            valid_time_deltas = time_delta[i, valid_indices, 0].cpu().numpy()

            # Create cumulative timestamps
            valid_timestamps = np.concatenate([[0], np.cumsum(valid_time_deltas[1:])])

            # Create spline interpolator
            if len(valid_timestamps) >= 3:
                f = interp1d(
                    valid_timestamps,
                    valid_power,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                f = interp1d(
                    valid_timestamps,
                    valid_power,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

            # Generate new timestamps
            new_timestamps = np.linspace(0, valid_timestamps.max(), target_length)
            interpolated_power = f(new_timestamps)

            interpolated_list.append(
                torch.tensor(interpolated_power, device=power.device, dtype=power.dtype)
            )

        return torch.stack(interpolated_list, dim=0).unsqueeze(1)

    @staticmethod
    def adaptive_interpolation(power, time_delta, mask, target_length=1024):
        """Adaptive interpolation based on time gaps"""
        interpolated_list = []

        for i in range(power.shape[0]):
            valid_indices = torch.where(mask[i] == False)[0]

            if len(valid_indices) == 0:
                interpolated_list.append(
                    torch.zeros(target_length, device=power.device)
                )
                continue

            valid_power = power[i, valid_indices, 0]
            valid_time_deltas = time_delta[i, valid_indices, 0]

            # Create timestamps
            valid_timestamps = torch.cat(
                [
                    torch.tensor([0.0], device=power.device),
                    torch.cumsum(valid_time_deltas, dim=0),
                ]
            )[:-1]

            # Detect large gaps (> median * 2)
            gaps = valid_time_deltas[1:]
            median_gap = torch.median(gaps)
            large_gap_threshold = median_gap * 2

            new_timestamps = torch.linspace(
                0, valid_timestamps.max(), target_length, device=power.device
            )

            # Use different interpolation for different regions
            indices = torch.searchsorted(valid_timestamps, new_timestamps)
            indices = torch.clamp(indices, 1, len(valid_timestamps) - 1)

            x0 = valid_timestamps[indices - 1]
            y0 = valid_power[indices - 1]
            x1 = valid_timestamps[indices]
            y1 = valid_power[indices]

            # Check for large gaps
            gap_sizes = x1 - x0
            large_gaps = gap_sizes > large_gap_threshold

            # Linear interpolation
            epsilon = 1e-8
            interpolated_power = y0 + (y1 - y0) * (new_timestamps - x0) / (
                x1 - x0 + epsilon
            )

            # For large gaps, use constant interpolation (hold last value)
            interpolated_power[large_gaps] = y0[large_gaps]

            interpolated_list.append(interpolated_power)

        return torch.stack(interpolated_list, dim=0).unsqueeze(1)
