import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt
from InterpolationModel import InterpolationBaseline
from UNet import UNetUpscaler
from DataPreparation import TimeSeriesPreparation

class InferenceManager:
    def __init__(self, model, device="cuda", seq_len=1800, overlap=0.5):
        self.model = model.to(device)
        self.model.eval()  # CRITICAL: Disables Dropout/BatchNorm updating
        self.device = device
        self.seq_len = seq_len
        self.stride = int(seq_len * (1 - overlap))
        self.interpolator = InterpolationBaseline(method="previous")

    def preprocess(self, chain2_df, ned_df):
        """
        Prepares raw dataframes into full-day arrays.
        """
        # 1. Extract Timestamps and Power
        # Assuming timestamps are in standard units (e.g. seconds or unix)
        # We align everything to the NED (target) grid.
        ned_ts = (
            ned_df["timestamp"].cast(pl.Int64).to_numpy() / 1e6
        )  # adjust divisor based on your data precision
        ned_power = ned_df["power"].to_numpy().astype(np.float32)

        chain_ts = chain2_df["timestamp"].cast(pl.Int64).to_numpy() / 1e6
        chain_power = chain2_df["power"].to_numpy().astype(np.float32)

        # 2. Interpolate (Baseline)
        interpolated_full = self.interpolator.predict(
            input_timestamps=chain_ts,
            power=chain_power,
            target_timestamps=ned_ts,
        ).astype(np.float32)

        # 3. Generate Mask
        indices = np.searchsorted(ned_ts, chain_ts)
        indices = np.clip(indices, 0, len(ned_ts) - 1)
        mask_full = np.zeros(len(ned_ts), dtype=np.float32)
        mask_full[indices] = 1.0

        return interpolated_full, mask_full, ned_power, ned_ts

    def predict_full(self, interpolated, mask):
        """
        Performs Sliding Window Inference.
        Returns the reconstructed (stitched) residual and final prediction.
        """
        total_len = len(interpolated)

        # Buffers to store the sum of predictions and the count of overlaps
        prediction_sum = np.zeros(total_len, dtype=np.float32)
        overlap_counts = np.zeros(total_len, dtype=np.float32)

        with torch.no_grad():  # CRITICAL: Saves memory, speeds up inference
            for start_idx in range(0, total_len - self.seq_len + 1, self.stride):
                end_idx = start_idx + self.seq_len

                # 1. Extract Window
                win_interp = interpolated[start_idx:end_idx]
                win_mask = mask[start_idx:end_idx]

                # 2. Prepare Tensor [1, 2, Seq_Len]
                # Note: We manually handle the batch dim (1) and channel dim
                x_interp = torch.from_numpy(win_interp).float().to(self.device)
                x_mask = torch.from_numpy(win_mask).float().to(self.device)

                # Stack to create [2, Seq_Len] -> Add Batch [1, 2, Seq_Len]
                x = torch.stack([x_interp, x_mask], dim=0).unsqueeze(0)

                # 3. Model Prediction
                # Output is [1, 1, Seq_Len] -> Squeeze to [Seq_Len]
                residual_pred = self.model(x).squeeze().cpu().numpy()

                # 4. Accumulate
                prediction_sum[start_idx:end_idx] += residual_pred
                overlap_counts[start_idx:end_idx] += 1.0

        # Handle edges where counts might be 0 (if stride logic skips end)
        # (Though logic above stops before end, simple handling:)
        overlap_counts[overlap_counts == 0] = 1.0

        # Average the predictions
        avg_residual = prediction_sum / overlap_counts

        # Final Signal = Baseline Interpolation + Predicted Residual
        final_prediction = interpolated + avg_residual

        return final_prediction, avg_residual

    def evaluate(self, house_id, chain2_df, ned_df):
        print(f"--- Inference for House {house_id} ---")

        # 1. Prepare
        interp, mask, truth, ts = self.preprocess(chain2_df, ned_df)

        # 2. Predict
        pred, residual = self.predict_full(interp, mask)

        # 3. Calculate Metrics (on full 24h)
        # Note: Depending on your goal, you might want to exclude the
        # very first/last few seconds if they weren't covered by windows
        valid_idx = np.where(pred != 0)[0]  # simple check

        mae_baseline = np.mean(np.abs(truth[valid_idx] - interp[valid_idx]))
        mae_model = np.mean(np.abs(truth[valid_idx] - pred[valid_idx]))

        print(f"Baseline MAE: {mae_baseline:.4f} W")
        print(f"Model MAE:    {mae_model:.4f} W")
        print(f"Improvement:  {mae_baseline - mae_model:.4f} W")

        # 4. Visualize
        self.plot_results(ts, truth, interp, pred, mask)

    @staticmethod
    def plot_results(ts, truth, interp, pred, mask):
        # Convert TS to datetime for nicer plotting if needed, or use index
        plt.figure(figsize=(15, 6))

        # Plot a subset (e.g., a zoom in on a spike) or full day
        # Let's verify the whole day first
        plt.plot(truth, label="Ground Truth (NED)", color="black", alpha=0.3)
        plt.plot(interp, label="Baseline (Interp)", color="blue", linestyle="--")
        plt.plot(pred, label="U-Net Prediction", color="red", alpha=0.8)

        # Optional: Plot dots where Chain2 data actually existed
        chain2_indices = np.where(mask == 1)[0]
        plt.scatter(
            chain2_indices,
            interp[chain2_indices],
            color="green",
            s=10,
            label="Chain2 Points",
            zorder=5,
        )

        plt.title("24-Hour Super-Resolution Reconstruction")
        plt.ylabel("Power (W)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_inference_test():
    # 1. Load Model
    # Point this to your best .ckpt file from lightning_logs
    checkpoint_path = "checkpoints/UNet_SEQ=1800_Freq=1.ckpt"
    model = UNetUpscaler.load_from_checkpoint(checkpoint_path)

    # 2. Setup Inference Manager
    # Overlap 0.5 means we stride by half the sequence length (smooth transitions)
    inference = InferenceManager(model, device="cuda", seq_len=1800, overlap=0.5)

    # 3. Get Data (Example for one house)
    tsp = TimeSeriesPreparation(...)  # Your existing config
    # Force loading a specific house or use existing logic
    # Let's say you have a method to get raw DFs for a specific house:
    house_id = 12345
    chain2_df, ned_df = tsp.chain2(
        house_id,
        n=1
    )  # You might need to expose this in DataPrep
    # 4. Run
    inference.evaluate(house_id, chain2_df, ned_df)


if __name__ == "__main__":
    run_inference_test()
