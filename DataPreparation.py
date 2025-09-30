import pprint
from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt

from config.DatabaseManager import DatabaseConnector
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from InterpolationModel import InterpolationBaseline


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

    def normalize_data_simple(self, chain2_df: pl.DataFrame, ned_df: pl.DataFrame):
        if self.to_normalize:
            min_, max_ = ned_df["power"].min(), ned_df["power"].quantile(quantile=0.98)
            chain2_df = chain2_df.with_columns(
                original_power=pl.col("power"),
                original_time_delta=pl.col("time_delta"),
                power=pl.col("power").clip(0, max_) / max_,
                time_delta=pl.col("time_delta"),
            )
            ned_df = ned_df.with_columns(
                original_power=pl.col("power"),
                power=pl.col("power").clip(0, max_) / max_,
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
        print(f"Normalization: {self.normalization_method}" f"\n")

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

    def load_chain_2(self, limit=-1, ratio=-1.0):
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
            if ratio > 0:
                print(
                    f"Loaded {ratio*100}% of original: {int(len(ned_df)*ratio)}/{len(ned_df)} rows"
                )
                chain2_df = chain2_df.limit(int(len(chain2_df) * ratio))
                ned_df = ned_df.limit(int(len(ned_df) * ratio))
            elif limit > 0:
                print(
                    f"Loaded {limit} rows of original: {limit}/{len(ned_df)} rows ({limit / len(ned_df)*100}%)"
                )
                chain2_df = chain2_df.limit(limit)
                ned_df = ned_df.limit(limit)

        chain2_df = chain2_df.with_columns(
            time_delta=pl.col("timestamp")
            .over("house_id")
            .diff()
            .dt.total_seconds()
            .fill_null(0),
        ).filter(pl.col("time_delta").abs() <= 60)

        chain2_df, ned_df = self.normalize_data_simple(chain2_df, ned_df)
        self.calculate_statistics(chain2_df, ned_df)

        return (
            chain2_df,
            ned_df,
        )


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
        only_spike=False,
        to_interpolate=True,
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
        self.only_spike = only_spike
        self.to_interpolate = to_interpolate
        self.interpolate_model = InterpolationBaseline(self.sequence_len, "linear")

        self.validate_inputs()
        self.create_dataset()

    def get_time_splits(
        self, current_target: pl.DataFrame, timestamp_col="timestamp"
    ) -> Tuple[pl.Series, pl.Series, int]:
        """Split timestamps for train/val to avoid data leakage"""
        if not self.split_by_time:
            # Random split (less recommended)
            n_train = int(len(current_target) * self.split_ratio)
            indices = np.random.permutation(len(current_target))
            return (
                current_target[indices[:n_train]][timestamp_col],
                current_target[indices[n_train:]][timestamp_col],
                0
            )

        # Time-based split (recommended)
        split_time = np.quantile(current_target[timestamp_col], self.split_ratio)
        mask = current_target.with_columns(
            train_mask=pl.col(timestamp_col) <= split_time,
            val_mask=pl.col(timestamp_col) >= split_time,
        )
        idx = mask.with_row_index().filter(pl.col("train_mask"))[-1, 0]
        return (
            mask.filter(pl.col("train_mask"))[timestamp_col],
            mask.filter(pl.col("val_mask"))[timestamp_col],
            idx
        )

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
        """
        Simpler dataset creation:
        - Align each target (ned_d) window with chain2 in the same time range.
        - No time buffer, no overlap complexity.
        - Input = (power, time_delta) from chain2
        - Target = ned_d sequence
        """

        self.dataset = []

        house_ids = self.ned_d["house_id"].unique()
        step_size = max(1, int(self.sequence_len * (1 - self.overlap_ratio)))
        print(
            f"[create_dataset] Building dataset for {self.phase} step size {step_size}"
        )

        for house_id in house_ids:
            house_ned = self.ned_d.filter(pl.col("house_id") == house_id).sort(
                "timestamp"
            )
            if len(house_ned) < self.sequence_len:
                continue
            house_chain = self.chain2.filter(pl.col("house_id") == house_id).sort(
                "timestamp"
            )
            # Train/val split
            train_times, val_times, idx_split = self.get_time_splits(house_ned)
            loop_range = (
                range(0, len(train_times) - self.sequence_len + 1, step_size)
                if self.phase == "train"
                else range(idx_split, len(house_ned) - self.sequence_len + 1, step_size)
            )

            for i in loop_range:
                curr_ned = house_ned[i : i + self.sequence_len]
                start_time = curr_ned[0]["timestamp"].item()
                end_time = curr_ned[-1]["timestamp"].item()

                # Take only chain2 points strictly within this window
                curr_chain = house_chain.filter(
                    (pl.col("timestamp") >= start_time)
                    & (pl.col("timestamp") <= end_time)
                )
                has_spike = not curr_chain.filter(
                    pl.col("original_power").abs().gt(300)
                ).is_empty()

                if len(curr_chain) > self.max_input_len:
                    curr_chain = curr_chain[-self.max_input_len :]

                if len(curr_chain) < self.min_input_len:
                    continue

                if self.only_spike and not has_spike:
                    continue

                if self.show:
                    fig, ax = plt.subplots()
                    curr_ned.to_pandas().set_index("timestamp", drop=True)["original_power"].plot(ax=ax, color="red")
                    curr_chain.to_pandas().set_index("timestamp", drop=True)["original_power"].plot(ax=ax, color="blue")
                    plt.show()

                # Extract arrays
                target_power = curr_ned["power"].to_numpy().astype(np.float32)
                chain_power = curr_chain["power"].to_numpy().astype(np.float32)
                chain_time_deltas = (
                    curr_chain["time_delta"].to_numpy().astype(np.float32)
                )

                current_len = len(chain_power)

                if self.to_interpolate:
                    chain_power = self.interpolate_model.predict(
                        chain_power, chain_time_deltas
                    )
                    sample = {
                        "power": torch.from_numpy(chain_power).float().unsqueeze(-1),
                        "target": torch.from_numpy(target_power).float().unsqueeze(-1),
                    }
                else:
                    # Pad inputs
                    padded_power = np.zeros(self.max_input_len, dtype=np.float32)
                    padded_power[:current_len] = chain_power

                    padded_td = np.zeros(self.max_input_len, dtype=np.float32)
                    padded_td[:current_len] = chain_time_deltas

                    # Attention mask
                    mask = np.zeros(self.max_input_len, dtype=bool)
                    mask[:current_len] = True

                    # Build sample
                    sample = {
                        "power": torch.from_numpy(padded_power).float().unsqueeze(-1),
                        "time_delta": torch.from_numpy(padded_td).float().unsqueeze(-1),
                        "mask": torch.from_numpy(mask).bool(),
                        "target": torch.from_numpy(target_power).float().unsqueeze(-1),
                    }
                self.dataset.append(sample)

        print(f"[create_dataset] Total samples: {len(self.dataset)}")
        if len(self.dataset) > 0:
            print(
                "Sample example:",
                {k: v.shape for k, v in self.dataset[0].items() if hasattr(v, "shape")},
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns:
            x: Input features [max_input_len, 1] (power) if interpolate else [max_input_len, 2] (power, time_delta)
            y: Target sequence [sequence_len]
            mask: Attention mask [max_input_len] (True for valid positions) if not_interpolate
            time_deltas: Actual time deltas [max_input_len] (for temporal encoding) if not_interpolate
            ts: timestamp for plotting
        """
        sample = self.dataset[idx]
        if self.to_interpolate:
            return sample["power"], sample["target"]
        else:
            return (
                sample["power"],
                sample["target"],
                sample["mask"],
                sample["time_delta"],
            )


if __name__ == "__main__":
    TARGET_FREQ = 5
    TIME_WINDOW_MINUTES = 30
    SEQ_LEN = TIME_WINDOW_MINUTES * 60 // TARGET_FREQ
    tsp = TimeSeriesPreparation(
        down_sample_to=TARGET_FREQ, limit=2, n_days=1, to_normalize=False
    )
    chain2, ned_d = tsp.load_chain_2(ratio=0.01)
    train_dataset = UpScalingDataset(
        ned_d=ned_d,
        chain2=chain2,
        sequence_len=SEQ_LEN,  # Target sequence length
        max_input_len=SEQ_LEN,  # Max irregular input length
        min_input_len=min(10, SEQ_LEN),  # Min input length
        overlap_ratio=0.9,  # 30% overlap
        normalize=False,  # Already normalized
        phase="train",
        split_by_time=True,
        show=True,
    )
