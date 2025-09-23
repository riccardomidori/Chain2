import pprint
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt

from config.DatabaseManager import DatabaseConnector
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pandas import Timedelta
import seaborn as sns


class TimeSeriesPreparation:
    def __init__(
        self,
        to_normalize=True,
        down_sample_to=30,  # seconds to downsample the ned dataset
        limit=2,
        n_days=1,
        normalization_method="robust",
    ):
        self.to_normalize = to_normalize
        self.normalization_method = normalization_method
        self.database = DatabaseConnector()
        self.connection_string = self.database.connection_string
        self.down_sample_to = down_sample_to
        self.limit = limit
        self.n_days = n_days

        self.power_scaler = self.get_scaler()
        self.time_delta_scaler = StandardScaler()

    @staticmethod
    def power_crossings(
        df: pl.DataFrame, step=300, time_col="timestamp", power_col="p"
    ) -> pl.DataFrame:
        df = df.with_columns([pl.col(power_col).shift(1).alias("prev_power")])
        df = df.with_columns(
            [
                (pl.col(power_col) // step).alias("bucket"),
                (pl.col("prev_power") // step).alias("prev_bucket"),
            ]
        )
        return df.filter(pl.col("bucket") != pl.col("prev_bucket")).select(
            [time_col, power_col]
        )

    def get_scaler(self):
        """Get the appropriate scaler based on method"""
        if self.normalization_method == "standard":
            return StandardScaler()
        elif self.normalization_method == "minmax":
            return MinMaxScaler()
        elif self.normalization_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}"
            )

    def denormalize_power(self, normalized_power):
        """Denormalize power values back to original scale"""
        if hasattr(normalized_power, "numpy"):
            power_array = normalized_power.detach().cpu().numpy()
        else:
            power_array = np.array(normalized_power)

        if power_array.ndim == 1:
            power_array = power_array.reshape(-1, 1)

        return self.power_scaler.inverse_transform(power_array).flatten()

    def normalize_data_simple(self, chain2_df: pl.DataFrame, ned_df: pl.DataFrame):
        if self.to_normalize:
            chain2_df = chain2_df.with_columns(
                original_power=pl.col("power"),
                original_time_delta=pl.col("time_delta"),
                power=(pl.col("power") - pl.col("power").mean())
                / pl.col("power").std(),
                time_delta=(pl.col("time_delta") - pl.col("time_delta").mean())
                / pl.col("time_delta").std(),
            )
            ned_df = ned_df.with_columns(
                original_power=pl.col("power"),
                power=(pl.col("power") - pl.col("power").mean())
                / pl.col("power").std(),
            )
        else:
            chain2_df = chain2_df.with_columns(
                original_power=pl.col("power"),
                original_time_delta=pl.col("time_delta"),
                power=pl.col("power"),
                time_delta=pl.col("time_delta"),
            )
            ned_df = ned_df.with_columns(
                original_power=pl.col("power"),
                power=pl.col("power"),
            )
        return chain2_df, ned_df

    def normalize_data(
        self, chain2_df: pl.DataFrame, ned_df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Improved normalization strategy
        """
        # Collect all power values for consistent normalization
        all_power_chain2 = chain2_df["power"].to_numpy()
        all_power_ned = ned_df["power"].to_numpy()
        all_power = np.concatenate([all_power_chain2, all_power_ned])

        # Fit power scaler on combined data
        self.power_scaler.fit(all_power.reshape(-1, 1))

        # Normalize power values
        chain2_power_norm = self.power_scaler.transform(
            all_power_chain2.reshape(-1, 1)
        ).flatten()

        ned_power_norm = self.power_scaler.transform(
            all_power_ned.reshape(-1, 1)
        ).flatten()

        # Handle time deltas
        time_deltas = chain2_df["time_delta"].to_numpy()
        time_deltas_filtered = time_deltas[time_deltas > 0]  # Remove zeros

        if len(time_deltas_filtered) > 0:
            self.time_delta_scaler.fit(time_deltas_filtered.reshape(-1, 1))
            time_deltas_norm = np.zeros_like(time_deltas)

            # Only normalize non-zero time deltas
            non_zero_mask = time_deltas > 0
            time_deltas_norm[non_zero_mask] = self.time_delta_scaler.transform(
                time_deltas[non_zero_mask].reshape(-1, 1)
            ).flatten()
        else:
            time_deltas_norm = time_deltas

        # Update dataframes with normalized values
        chain2_df = chain2_df.with_columns(
            [
                pl.Series("power", chain2_power_norm),
                pl.Series("time_delta", time_deltas_norm),
                pl.Series("original_power", all_power_chain2),
                pl.Series("original_time_delta", time_deltas),
            ]
        )

        ned_df = ned_df.with_columns(
            [
                pl.Series("power", ned_power_norm),
                pl.Series("original_power", all_power_ned),
            ],
        )

        return chain2_df, ned_df

    def calculate_statistics(self, chain2_df: pl.DataFrame, ned_df: pl.DataFrame):
        n_houses = len(chain2_df["house_id"].unique())
        """Calculate and store dataset statistics"""
        stats = {
            "n_houses_processed": n_houses,
            "chain2_samples": len(chain2_df),
            "ned_samples": len(ned_df),
            "chain2_power_range (W)": (
                chain2_df["original_power"].min(),
                chain2_df["original_power"].max(),
            ),
            "chain2_time_range (s)": (
                chain2_df["original_time_delta"].min(),
                chain2_df["original_time_delta"].max(),
            ),
            "ned_power_range (W)": (
                ned_df["original_power"].min(),
                ned_df["original_power"].max(),
            ),
            "avg_time_delta (s)": chain2_df["original_time_delta"].mean(),
            "time_delta_std (s)": chain2_df["original_time_delta"].std(),
            "normalization_method": self.normalization_method,
            "normalised_chain2_power_range (W)": (
                chain2_df["power"].min(),
                chain2_df["power"].max(),
            ),
            "normalised_ned_power_range (W)": (
                ned_df["power"].min(),
                ned_df["power"].max(),
            ),
        }

        print(f"\n=== Dataset Statistics ===")
        print(f"Houses processed: {n_houses}")
        print(f"Chain2 samples: {len(chain2_df):,}")
        print(f"NED samples: {len(ned_df):,}")
        print("\n")
        pprint.pprint(stats)
        print("\n")
        print(f"Normalization: {self.normalization_method}")

    def chain2(self, house_id, show=False, every="60s", n=1):
        query = (
            "select from_unixtime(t) as timestamp, "
            "abs(p1) as power "
            f"from ned_data_{house_id} "
            f"where t>=unix_timestamp(curdate()) - {n}*86400 "
            f"and t<unix_timestamp(curdate()) "
        )

        df = pl.read_database_uri(query, self.connection_string).with_columns(
            timestamp=pl.col("timestamp").dt.cast_time_unit(time_unit="ms")
        )
        df_grouped = df.group_by_dynamic(
            index_column="timestamp", every=every, period="1s", closed="left"
        ).agg(power=pl.col("power").first())
        df_cross = self.power_crossings(df, power_col="power")
        df_down_sampled = (
            pl.concat([df_grouped, df_cross])
            .unique(subset=["timestamp"])
            .sort("timestamp")
        )
        if self.down_sample_to > 1:
            df = df.group_by_dynamic(
                "timestamp", every=f"{self.down_sample_to}s", period="1s", closed="left"
            ).agg(power=pl.col("power").first())
        if show:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, sharex=True)
            df.to_pandas().set_index("timestamp", drop=True).plot(ax=ax[0])
            df_down_sampled.to_pandas().set_index("timestamp", drop=True).plot(ax=ax[1])
            plt.show()

        return (
            df_down_sampled.with_columns(house_id=pl.lit(house_id)),
            df.with_columns(house_id=pl.lit(house_id)),
        )

    def load_chain_2(self):
        p = Path(f"data/{self.down_sample_to}")
        if not p.exists() and not (p / "chain2.csv").exists():
            users_query = (
                "select id_abitazione as id "
                "from tab_hourly_consumption "
                "where timestamp>unix_timestamp(curdate()) "
                "and id_abitazione > 2 "
                "group by id_abitazione "
                "order by rand(69) "
                f"limit {self.limit}"
            )

            users = pl.read_database_uri(users_query, self.connection_string)
            input_dfs, target_dfs = [], []
            processed_houses = 0
            for row in users.iter_rows(named=True):
                house_id = row["id"]
                df_input, df_target = self.chain2(house_id, n=self.n_days)
                print(
                    f"HouseID@{house_id} - {len(df_input)} Chain2 samples - {len(df_target)} NED_D samples"
                )
                if df_input is not None and df_target is not None:
                    input_dfs.append(df_input)
                    target_dfs.append(df_target)
                    processed_houses += 1
            chain2_df, ned_df = (
                pl.concat(input_dfs, how="vertical").sort(by=["house_id", "timestamp"]),
                pl.concat(target_dfs, how="vertical").sort(
                    by=["house_id", "timestamp"]
                ),
            )
            p.mkdir(parents=True)
            chain2_df.write_csv(f"data/{self.down_sample_to}/chain2.csv")
            ned_df.write_csv(f"data/{self.down_sample_to}/ned.csv")
        else:
            chain2_df = pl.read_csv(
                f"data/{self.down_sample_to}/chain2.csv"
            ).with_columns(
                timestamp=pl.col("timestamp").str.strptime(
                    pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f"
                )
            )
            ned_df = pl.read_csv(f"data/{self.down_sample_to}/ned.csv").with_columns(
                timestamp=pl.col("timestamp").str.strptime(
                    pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f"
                )
            )

        chain2_df = chain2_df.with_columns(
            time_delta=pl.col("timestamp")
            .over("house_id")
            .diff()
            .dt.total_seconds()
            .fill_null(0)
        ).filter(pl.col("time_delta").abs() <= 60)

        chain2_df, ned_df = self.normalize_data_simple(chain2_df, ned_df)

        self.calculate_statistics(chain2_df, ned_df)
        scaling_obj = self.get_scaling()

        return (
            chain2_df,
            ned_df,
            scaling_obj["power_scaler"],
            scaling_obj["time_delta_scaler"],
        )

    def get_scaling(self) -> Dict:
        """Return fitted scalers for use in models"""
        return {
            "power_scaler": self.power_scaler,
            "time_delta_scaler": self.time_delta_scaler,
        }


class UpScalingDataset(Dataset):
    def __init__(
        self,
        ned_d: pl.DataFrame,
        chain2: pl.DataFrame,
        sequence_len=30,
        max_len=30,
        step=30,
        normalize: bool = True,
        max_input_len: int = 50,  # Maximum input sequence length (low-freq)
        min_input_len: int = 5,  # Minimum input sequence length
        overlap_ratio: float = 0.5,  # Overlap between sequences (0.0 - 1.0)
        power_threshold: float = 0.01,  # Minimum power to consider valid
        time_window_hours: float = 2.0,  # Time window for matching sequences
        split_by_time: bool = True,  # Whether to split by time (avoid data leakage)
        split_ratio: float = 0.8,  # Train/val split ratio
        phase: str = "train",  # 'train', 'val', or 'test'
        show=False,
    ):
        self.ned_d = ned_d
        self.chain2 = chain2
        self.sequence_len = sequence_len
        self.max_len = max_len
        self.step = step
        self.dataset: list[dict] = []
        self.normalize = normalize
        self.max_input_len = max_input_len
        self.min_input_len = min_input_len
        self.overlap_ratio = overlap_ratio
        self.power_threshold = power_threshold
        self.time_window_hours = time_window_hours
        self.split_by_time = split_by_time
        self.split_ratio = split_ratio
        self.phase = phase
        self.show = show

        self.power_scaler = StandardScaler()
        self.time_delta_scaler = StandardScaler()

        self.validate_inputs()

        self.create_dataset_simple()

    def get_time_splits(
        self, current_target: pl.DataFrame, timestamp_col="timestamp"
    ) -> Tuple[pl.Series, pl.Series]:
        """Split timestamps for train/val to avoid data leakage"""
        if not self.split_by_time:
            # Random split (less recommended)
            n_train = int(len(current_target) * self.split_ratio)
            indices = np.random.permutation(len(current_target))
            return (
                current_target[indices[:n_train]][timestamp_col],
                current_target[indices[n_train:]][timestamp_col],
            )

        # Time-based split (recommended)
        split_time = np.quantile(current_target[timestamp_col], self.split_ratio)
        mask = current_target.with_columns(
            train_mask=pl.col(timestamp_col) <= split_time,
            val_mask=pl.col(timestamp_col) >= split_time,
        )
        return (
            mask.filter(pl.col("train_mask"))[timestamp_col],
            mask.filter(pl.col("val_mask"))[timestamp_col],
        )

    @staticmethod
    def calculate_time_deltas(timestamps: np.ndarray) -> np.ndarray:
        """Calculate time deltas in seconds"""
        if len(timestamps) <= 1:
            return np.array([0.0])

        # Convert to seconds and calculate deltas
        time_deltas = np.diff(timestamps.astype("datetime64[s]").astype(float))
        # Add zero for first element to match sequence length
        time_deltas = np.concatenate([[0.0], time_deltas])

        return time_deltas

    def validate_inputs(self):
        """Validate input parameters and data"""
        assert self.sequence_len > 0, "sequence_len must be positive"
        assert (
            self.max_input_len >= self.min_input_len
        ), "max_input_len must be >= min_input_len"
        assert 0 <= self.overlap_ratio < 1, "overlap_ratio must be in [0, 1)"
        assert self.phase in [
            "train",
            "val",
            "test",
        ], "phase must be 'train', 'val', or 'test'"

        # Check required columns
        required_ned_cols = ["house_id", "timestamp", "power"]
        required_chain2_cols = ["house_id", "timestamp", "power", "time_delta"]

        for col in required_ned_cols:
            assert col in self.ned_d.columns, f"Missing column {col} in ned_d"
        for col in required_chain2_cols:
            assert col in self.chain2.columns, f"Missing column {col} in chain2"

    def create_dataset(self):
        house_ids = self.ned_d["house_id"].unique()
        for house_id in house_ids:
            house_ned = self.ned_d.filter(pl.col("house_id").eq(house_id))
            house_chain = self.chain2.filter(pl.col("house_id").eq(house_id))

            for i in range(0, len(house_ned) - self.sequence_len, self.sequence_len):
                curr_ned = house_ned[i : i + self.sequence_len]
                start_time = curr_ned[0]["timestamp"]
                end_time = curr_ned[-1]["timestamp"]
                curr_chain = house_chain.filter(
                    (pl.col("timestamp") >= start_time)
                    & (pl.col("timestamp") <= end_time)
                )
                if len(curr_chain) == 0:
                    continue  # Skip windows with no Chain2 data
                target = curr_ned.select(["power"]).to_numpy()

                curr_chain_np = (
                    curr_chain.select(["power", "time_delta"])
                    .to_numpy()
                    .astype(np.float32)
                )

                if len(curr_chain_np) > self.max_len:
                    # Truncate the sequence if it's too long
                    # Keep the most recent data points, as they are often more relevant.
                    curr_chain_np = curr_chain_np[-self.max_len :]

                current_len = len(curr_chain_np)

                # Pad the input sequence to the max_chain2_len
                padded_input = np.zeros(
                    (self.max_len, curr_chain_np.shape[1]), dtype=np.float32
                )
                padded_input[:current_len, :] = curr_chain_np

                # Create the corresponding mask
                mask = np.zeros(self.max_len, dtype=bool)
                mask[:current_len] = True
                ts = (
                    curr_ned.with_columns(ts_int=pl.col("timestamp").dt.timestamp())
                    .select("ts_int")
                    .to_numpy()
                )
                to_add = {
                    "ts": ts,
                    "input": torch.from_numpy(padded_input).float(),
                    "target": torch.from_numpy(target)
                    .squeeze(-1)
                    .float(),  # Squeeze to remove trailing dim
                    "mask": torch.from_numpy(
                        ~mask
                    ).float(),  # Invert the mask for Transformer,
                }
                # to_add["input_shape"] = to_add["input"].shape
                # to_add["target_shape"] = to_add["target"].shape
                # to_add["mask_shape"] = to_add["mask"].shape
                self.dataset.append(to_add)

    def create_dataset_2(self):
        print(f"Creation dataset for {self.phase}")

        house_ids = self.ned_d["house_id"].unique()
        self.dataset: list[dict] = []

        # Second pass: create sequences
        samples = 0
        for house_id in house_ids:
            house_ned = self.ned_d.filter(pl.col("house_id").eq(house_id)).sort(
                "timestamp"
            )
            house_chain = self.chain2.filter(pl.col("house_id").eq(house_id)).sort(
                "timestamp"
            )

            if len(house_ned) < self.sequence_len:
                continue

            train_times, val_times = self.get_time_splits(house_ned)
            valid_times_set = set(
                train_times.to_list() if self.phase == "train" else val_times.to_list()
            )

            # Create sequences
            step_size = max(1, int(self.sequence_len * (1 - self.overlap_ratio)))

            for i in range(0, len(house_ned) - self.sequence_len + 1, step_size):
                curr_ned = house_ned[i : i + self.sequence_len]
                start_time = curr_ned[0]["timestamp"].item()
                end_time = curr_ned[-1]["timestamp"].item()

                # Time-based filtering
                if self.split_by_time and start_time not in valid_times_set:
                    continue

                # Add time window buffer
                time_buffer_seconds = Timedelta(hours=self.time_window_hours)
                search_start = start_time - time_buffer_seconds
                search_end = end_time + time_buffer_seconds

                curr_chain = house_chain.filter(
                    (pl.col("timestamp") >= search_start)
                    & (pl.col("timestamp") <= search_end)
                )

                if len(curr_chain) < self.min_input_len:
                    continue

                target_power = curr_ned["power"].to_numpy().astype(np.float32)
                chain_power = curr_chain["power"].to_numpy().astype(np.float32)
                chain_time_deltas = (
                    curr_chain["time_delta"].to_numpy().astype(np.float32)
                )

                # Truncate if too long (keep most recent data)
                if len(chain_power) > self.max_input_len:
                    chain_power = chain_power[-self.max_input_len :]
                    chain_time_deltas = chain_time_deltas[-self.max_input_len :]

                current_len = len(chain_power)

                # Create padded input sequence
                padded_input_power = np.zeros(self.max_input_len, dtype=np.float32)
                padded_input_power[:current_len] = chain_power

                padded_input_td = np.zeros(self.max_input_len, dtype=np.float32)
                padded_input_td[:current_len] = chain_time_deltas

                # Create attention mask
                attention_mask = np.zeros(self.max_input_len, dtype=bool)
                attention_mask[:current_len] = True
                ts = (
                    curr_ned.with_columns(ts_int=pl.col("timestamp").dt.timestamp())
                    .select("ts_int")
                    .to_numpy()
                )
                sample = {
                    "power": torch.from_numpy(padded_input_power).float().unsqueeze(-1),
                    "time_delta": torch.from_numpy(padded_input_td)
                    .float()
                    .unsqueeze(-1),
                    "mask": torch.from_numpy(attention_mask).bool(),
                    "target": torch.from_numpy(target_power).float().unsqueeze(-1),
                    "ts": ts,
                }
                self.dataset.append(sample)
                samples += 1

            print(f"\t\tHouse@{house_id} added {samples} samples")
            samples = 0
        print(f"Total samples added: {len(self.dataset)}")
        if len(self.dataset) > 0:
            print(
                f"\tSample example:",
                {
                    k: v.shape if hasattr(v, "shape") else v
                    for k, v in self.dataset[0].items()
                },
            )

    def create_dataset_simple(self):
        """
        Simpler dataset creation:
        - Align each target (ned_d) window with chain2 in the same time range.
        - No time buffer, no overlap complexity.
        - Input = (power, time_delta) from chain2
        - Target = ned_d sequence
        """

        print(f"[create_dataset_simple] Building dataset for {self.phase}")
        self.dataset = []

        house_ids = self.ned_d["house_id"].unique()
        step_size = max(1, int(self.sequence_len * (1 - self.overlap_ratio)))

        for house_id in house_ids:
            house_ned = self.ned_d.filter(pl.col("house_id") == house_id).sort(
                "timestamp"
            )
            house_chain = self.chain2.filter(pl.col("house_id") == house_id).sort(
                "timestamp"
            )

            if len(house_ned) < self.sequence_len:
                continue

            # Train/val split
            train_times, val_times = self.get_time_splits(house_ned)
            valid_times_set = set(
                train_times.to_list() if self.phase == "train" else val_times.to_list()
            )

            for i in range(0, len(house_ned) - self.sequence_len + 1, step_size):
                curr_ned = house_ned[i : i + self.sequence_len]
                start_time = curr_ned[0]["timestamp"].item()
                end_time = curr_ned[-1]["timestamp"].item()

                # Enforce time-based split
                if self.split_by_time and start_time not in valid_times_set:
                    continue

                # Take only chain2 points strictly within this window
                curr_chain = house_chain.filter(
                    (pl.col("timestamp") >= start_time)
                    & (pl.col("timestamp") <= end_time)
                )

                if len(curr_chain) > self.max_input_len:
                    curr_chain = curr_chain[-self.max_input_len :]

                if self.show:
                    print(curr_ned)
                    print(curr_chain)
                    fig, ax = plt.subplots()
                    curr_ned.to_pandas().set_index("timestamp")["power"].plot(
                        ax=ax, color="red"
                    )
                    curr_chain.to_pandas().set_index("timestamp")["power"].plot(
                        ax=ax, color="blue"
                    )
                    plt.show()

                if len(curr_chain) < self.min_input_len:
                    continue

                # Extract arrays
                target_power = curr_ned["power"].to_numpy().astype(np.float32)
                chain_power = curr_chain["power"].to_numpy().astype(np.float32)
                chain_time_deltas = (
                    curr_chain["time_delta"].to_numpy().astype(np.float32)
                )

                current_len = len(chain_power)

                # Pad inputs
                padded_power = np.zeros(self.max_input_len, dtype=np.float32)
                padded_power[:current_len] = chain_power

                padded_td = np.zeros(self.max_input_len, dtype=np.float32)
                padded_td[:current_len] = chain_time_deltas

                # Attention mask
                mask = np.zeros(self.max_input_len, dtype=bool)
                mask[:current_len] = True

                # Convert timestamps to numpy
                ts = (
                    curr_ned.with_columns(ts_int=pl.col("timestamp").dt.timestamp())
                    .select("ts_int")
                    .to_numpy()
                )

                # Build sample
                sample = {
                    "power": torch.from_numpy(padded_power).float().unsqueeze(-1),
                    "time_delta": torch.from_numpy(padded_td).float().unsqueeze(-1),
                    "mask": torch.from_numpy(mask).bool(),
                    "target": torch.from_numpy(target_power).float().unsqueeze(-1),
                    "ts": ts,
                }
                self.dataset.append(sample)

        print(f"[create_dataset_simple] Total samples: {len(self.dataset)}")
        if len(self.dataset) > 0:
            print(
                "Sample example:",
                {k: v.shape for k, v in self.dataset[0].items() if hasattr(v, "shape")},
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input features [max_input_len, 2] (power, time_delta)
            y: Target sequence [sequence_len]
            mask: Attention mask [max_input_len] (True for valid positions)
            time_deltas: Actual time deltas [max_input_len] (for temporal encoding)
            ts: timestamp for plotting
        """
        sample = self.dataset[idx]

        # For compatibility with your existing code
        power = sample["power"]  # [max_input_len, 1]
        time_deltas = sample["time_delta"]  # [max_input_len]
        y = sample["target"]  # [sequence_len]
        mask = sample["mask"]  # [max_input_len]
        ts = sample["ts"]
        return power, y, mask, time_deltas, ts


if __name__ == "__main__":
    TARGET_FREQ = 10
    TIME_WINDOW_HOURS = 2
    SEQ_LEN = TIME_WINDOW_HOURS * 3600 // TARGET_FREQ
    tsp = TimeSeriesPreparation(
        down_sample_to=TARGET_FREQ, limit=2, n_days=1, to_normalize=False
    )
    chain2, ned_d, power_scaling, time_delta_scaling = tsp.load_chain_2()
    train_dataset = UpScalingDataset(
        ned_d=ned_d,
        chain2=chain2,
        sequence_len=SEQ_LEN,  # Target sequence length
        max_input_len=SEQ_LEN,  # Max irregular input length
        min_input_len=10,  # Min input length
        overlap_ratio=0.3,  # 30% overlap
        normalize=False,  # Already normalized
        phase="train",
        split_by_time=True,
        time_window_hours=TIME_WINDOW_HOURS,
        show=False,
    )
