import numpy as np
from scipy.interpolate import interp1d
from typing import Literal


class InterpolationBaseline:
    """
    Robust interpolation using Scipy.
    Supports 'previous' (Step) interpolation which is best for Chain2 sensors.
    """

    def __init__(
        self,
        method: Literal["linear", "nearest", "previous", "next"] = "previous",
        fill_value: Literal["extrapolate", "nan"] = "extrapolate",
        output_seq_len=86400
    ):
        """
        Args:
            method: 'previous' is recommended for event-based sensors (Chain2).
                    'linear' is standard but may hallucinate ramps.
            fill_value: 'extrapolate' keeps the last known value for edges.
        """
        self.method = method
        self.fill_value = fill_value
        self.output_seq_len = output_seq_len

    @staticmethod
    def _interpolate_linear(
        timestamps: np.ndarray, power: np.ndarray, output_timestamps: np.ndarray
    ) -> np.ndarray:
        """Helper for linear interpolation using numpy.interp."""
        return np.interp(output_timestamps, timestamps, power)

    def predict(
        self,
        input_timestamps: np.ndarray,
        power: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolates irregularly sampled data onto a fixed target grid.

        Args:
            input_timestamps (np.ndarray): Absolute timestamps of the sensor data.
            power (np.ndarray): Power values corresponding to input_timestamps.
            target_timestamps (np.ndarray): The grid you want (e.g., 0, 1, 2... 29).

        Returns:
            np.ndarray: Interpolated values on the target grid.
        """
        # 1. Input Validation
        if len(input_timestamps) < 2:
            # Fallback for insufficient data: return constant array of the single value or 0
            val = power[0] if len(power) > 0 else 0.0
            return np.full_like(target_timestamps, val, dtype=float)

        # 2. Sort inputs (Scipy requires sorted x)
        sort_idx = np.argsort(input_timestamps)
        t_in = input_timestamps[sort_idx]
        v_in = power[sort_idx]
        if self.method == "linear":
            timestamps = np.cumsum(input_timestamps)

            # Create a new, evenly-spaced time axis for the output sequence.
            output_timestamps = np.linspace(
                timestamps[0], timestamps[-1], self.output_seq_len
            )
            return self._interpolate_linear(timestamps, power, output_timestamps)
        else:
            # 3. Create Interpolator
            # kind='previous' implements Zero-Order Hold (maintains value until next change)
            f = interp1d(
                t_in,
                v_in,
                kind=self.method,
                fill_value=self.fill_value,
                bounds_error=False,
                assume_sorted=True,
            )

            # 4. Interpolate onto target grid
            return f(target_timestamps)

    def __call__(self, t_in, v_in, t_out):
        return self.predict(t_in, v_in, t_out)
