import numpy as np
from typing import Literal

from matplotlib import pyplot as plt


class InterpolationBaseline:
    """
    A simple baseline model that uses various interpolation methods to upscale
    irregularly sampled time series data to a fixed length.
    """

    def __init__(
        self,
        output_seq_len: int,
        method: Literal["linear", "nearest", "constant"] = "linear",
    ):
        """
        Initializes the model with the target output sequence length and interpolation method.

        Args:
            output_seq_len (int): The number of data points in the output sequence.
            method (Literal["linear", "nearest", "constant"]): The interpolation method to use.
        """
        self.output_seq_len = output_seq_len
        if method not in ["linear", "nearest", "constant"]:
            raise ValueError(f"Unknown interpolation method: {method}")
        self.method = method

    @staticmethod
    def _interpolate_linear(
        timestamps: np.ndarray, power: np.ndarray, output_timestamps: np.ndarray
    ) -> np.ndarray:
        """Helper for linear interpolation using numpy.interp."""
        return np.interp(output_timestamps, timestamps, power)

    @staticmethod
    def _interpolate_nearest(
        timestamps: np.ndarray, power: np.ndarray, output_timestamps: np.ndarray
    ) -> np.ndarray:
        """Helper for nearest-neighbor interpolation."""
        # Find the index of the nearest timestamp for each output timestamp
        indices = np.searchsorted(timestamps, output_timestamps)
        # Handle edge cases (before the first timestamp, after the last)
        indices = np.clip(indices, 0, len(timestamps) - 1)
        # Get the power values at those indices
        return power[indices]

    @staticmethod
    def _interpolate_constant(
        timestamps: np.ndarray, power: np.ndarray, output_timestamps: np.ndarray
    ) -> np.ndarray:
        """Helper for piecewise constant interpolation (zero-order hold)."""
        indices = np.searchsorted(timestamps, output_timestamps)
        # Shift indices to the left to get the previous value
        indices = np.clip(indices - 1, 0, len(timestamps) - 1)
        return power[indices]

    def predict(self, power: np.ndarray, time_deltas: np.ndarray) -> np.ndarray:
        """
        Upscales the input time series using the chosen interpolation method.

        Args:
            power (np.ndarray): The low-frequency power data.
            time_deltas (np.ndarray): The time difference between each power sample.

        Returns:
            np.ndarray: The upscaled, fixed-length power sequence.
        """
        # Handle cases where the input data is too short
        if len(power) < 2:
            return np.full(self.output_seq_len, power[0] if len(power) > 0 else 0)

        # Create a cumulative time axis for the input data.
        timestamps = np.cumsum(time_deltas)

        # Create a new, evenly-spaced time axis for the output sequence.
        output_timestamps = np.linspace(
            timestamps[0], timestamps[-1], self.output_seq_len
        )

        # Perform the chosen interpolation
        if self.method == "linear":
            upscaled_power = self._interpolate_linear(
                timestamps, power, output_timestamps
            )
        elif self.method == "nearest":
            upscaled_power = self._interpolate_nearest(
                timestamps, power, output_timestamps
            )
        elif self.method == "constant":
            upscaled_power = self._interpolate_constant(
                timestamps, power, output_timestamps
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")

        return upscaled_power

    def __call__(self, power: np.ndarray, time_deltas: np.ndarray) -> np.ndarray:
        """
        Allows the class to be called directly, similar to a PyTorch model's forward method.
        """
        return self.predict(power, time_deltas)


# Example usage within a testing or evaluation script
if __name__ == "__main__":
    from DataPreparation import TimeSeriesPreparation, UpScalingDataset
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import polars as pl

    TARGET_FREQUENCY = 1
    TIME_WINDOW_MINUTES = 60
    SEQ_LEN = TIME_WINDOW_MINUTES * 60 // TARGET_FREQUENCY

    tsp = TimeSeriesPreparation(
        down_sample_to=TARGET_FREQUENCY, limit=2, n_days=1, to_normalize=False, show=True
    )
    chain2, ned_d = tsp.load_chain_2()
    train_dataset = UpScalingDataset(
        ned_d,
        chain2,
        sequence_len=SEQ_LEN,
        max_input_len=SEQ_LEN,
        min_input_len=10,
        overlap_ratio=0.8,
        normalize=False,  # Already normalized
        phase="train",
        split_by_time=True,
        show=True,
        to_interpolate=True,
        only_spike=False,
        split_ratio=0.7,
    )
    linear_baseline = InterpolationBaseline(SEQ_LEN, method="linear")
    nearest_baseline = InterpolationBaseline(SEQ_LEN, method="nearest")
    constant_baseline = InterpolationBaseline(SEQ_LEN, method="constant")
    errors = []
    for chunk in train_dataset:
        x, y, mask, time_deltas, ts = chunk
        power = x[mask == True].squeeze().numpy()
        time_deltas = time_deltas[mask == True].squeeze().numpy()
        upscaled_linear = linear_baseline(power, time_deltas)
        fig, ax = plt.subplots()
        ax.plot(y.squeeze(), label="True", linewidth=3)
        # ax.plot(x.squeeze(), label="Input")
        ax.plot(y.squeeze() - upscaled_linear, label="Residual")
        ax.plot(upscaled_linear, label="Linear")
        plt.legend()
        plt.show()
    df = pl.DataFrame(errors)
    print(df.mean())
