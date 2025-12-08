import datetime
import json
from typing import Literal

import matplotlib.pyplot as plt
import pytz

from config.DatabaseManager import DatabaseConnector
import polars as pl
from pathlib import Path
from DataPreparation import TimeSeriesPreparation
from multiprocessing import Pool
from InterpolationModel import InterpolationBaseline


class MAC:
    def __init__(self, n_jobs=1, run_mode=Literal["daily", "real-time"]):
        self.n_jobs = n_jobs
        self.log_path = Path("data/mac/mac.json")
        self.database_connector = DatabaseConnector()
        self.interpolator = InterpolationBaseline(method="previous")
        self.run_mode = run_mode

    def parse(self) -> pl.DataFrame:
        data = []
        with open(self.log_path, "r") as f:
            for i, line in enumerate(f.readlines()):
                try:
                    json_data: dict = json.loads(line)
                    message_id = list(json_data.keys())[0]
                    device_id = json_data[message_id]["Id"]
                    if message_id == "Chain2Data":
                        message_type = json_data[message_id]["Type"]
                        if message_type in ["CF51"]:
                            ts = datetime.datetime.strptime(
                                json_data[message_id]["Ts"], "%Y/%m/%d %H:%M:%S"
                            )
                            power = json_data[message_id]["Payload"]["InstantPower"]
                            obj = {
                                "device_id": device_id,
                                "timestamp": ts,
                                "power": power,
                            }
                            data.append(obj)
                except:
                    print(i)
                    print(line)
        df = pl.from_dicts(data)
        print(df)
        return df

    def parse_power(self) -> pl.DataFrame:
        """
        Optimized function using Polars' Lazy NDJSON scanning and expression engine.
        """
        log_file_path = Path(self.log_path)

        df = (
            pl.scan_ndjson(log_file_path, ignore_errors=True)
            .filter(pl.col("Chain2Data").is_not_null())
            .with_columns(
                device_id=pl.col("Chain2Data").struct.field("Id"),
                message_type=pl.col("Chain2Data").struct.field("Type"),
                timestamp_str=pl.col("Chain2Data").struct.field("Ts"),
                power=pl.col("Chain2Data")
                .struct.field("Payload")
                .struct.field("InstantPower"),
            )
            .with_columns(
                timestamp=pl.col("timestamp_str")
                .str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S")
                .dt.convert_time_zone("Europe/Rome"),
                power=pl.col("power").cast(
                    pl.Float64
                ),  # Assuming power is a float/numeric
            )
            .select(
                "device_id",
                "timestamp",
                "power",
            )
            .collect()
        )
        return df.filter(
            pl.col("timestamp").gt(
                datetime.datetime(2025, 1, 1, tzinfo=pytz.timezone("Europe/Rome"))
            )
        )

    def parse_energy(self):
        """
        Optimized function using Polars' Lazy NDJSON scanning and expression engine.
        """
        log_file_path = Path(self.log_path)

        # 1. Use scan_ndjson for memory-efficient, parallel reading and lazy execution.
        # We read all columns as strings initially to handle the nested/dynamic structure.
        df = (
            pl.scan_ndjson(log_file_path, ignore_errors=True)
            .filter(pl.col("Chain2Data").is_not_null())
            .with_columns(
                # 3. Extract nested fields and cast/parse them to the correct types
                device_id=pl.col("Chain2Data").struct.field("Id"),
                message_type=pl.col("Chain2Data").struct.field("Type"),
                timestamp_str=pl.col("Chain2Data").struct.field("Ts"),
            )
            .filter(pl.col("message_type") == "CF1")
            .with_columns(
                # 5. Fast timestamp parsing and final type casting
                timestamp=pl.col("timestamp_str")
                .str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S")
                .dt.convert_time_zone("Europe/Rome"),
                energy=pl.col("Chain2Data")
                .struct.field("Payload")
                .struct.field("CurrQuartActEnergy"),
            )
            .select(
                # 6. Select the final desired columns
                "device_id",
                "timestamp",
                "energy",
            )
            # 7. Trigger the computation and load the resulting DataFrame eagerly
            .collect()
        )
        return df

    def compare_energy(self):
        house_id = 1563
        df = self.parse_energy()
        df_chain = (
            df.filter(pl.col("device_id").eq("c2g-815846480"))
            .sort("timestamp")
            .group_by_dynamic("timestamp", every="1d")
            .agg(energy=pl.col("energy").sum(), count=pl.len())
        )
        start = df_chain["timestamp"][0]

        dc = DatabaseConnector()
        query = (
            "select from_unixtime(date) as timestamp, "
            "aggregate_energy as energy "
            "from tab_rt_dailyresults "
            f"where id_abitazione={house_id} "
            f"and date>={start.timestamp()}"
        )
        df_ned = pl.read_database_uri(query, dc.connection_string).with_columns(
            timestamp=pl.col("timestamp").dt.replace_time_zone("Europe/Rome")
        )
        print(df_chain)
        print(df_ned)
        fig, ax = plt.subplots()
        ax.plot(df_ned["timestamp"], df_ned["energy"], label="NED")
        ax.plot(df_chain["timestamp"], df_chain["energy"], label="MAC")
        plt.legend()
        plt.show()

    def compare(self):
        house_id = 248454
        df = self.parse_power()
        start, end = datetime.datetime(
            2025, 10, 13, tzinfo=pytz.timezone("Europe/Rome")
        ), datetime.datetime(2025, 10, 15, tzinfo=pytz.timezone("Europe/Rome"))
        tsp = TimeSeriesPreparation()
        chain2, _ = tsp.chain2(house_id, n=2)
        chain2 = chain2.with_columns(
            timestamp=pl.col("timestamp").dt.replace_time_zone("Europe/Rome")
        )
        dc = DatabaseConnector()
        df_chain = df.filter(
            (pl.col("device_id").eq("c2g-815846480"))
            & (pl.col("timestamp").is_between(start, end))
        ).sort("timestamp")

        query = (
            "select from_unixtime(t) as timestamp, "
            "p1 as power "
            f"from ned_data_{house_id} "
            f"where t>={start.timestamp()} "
            f"and t<={end.timestamp()}"
        )
        df_ned = pl.read_database_uri(query, dc.connection_string).with_columns(
            timestamp=pl.col("timestamp").dt.replace_time_zone("Europe/Rome")
        )
        print(df_ned)
        fig, ax = plt.subplots()
        ax.plot(df_ned["timestamp"], df_ned["power"], label="NED")
        ax.plot(df_chain["timestamp"], df_chain["power"], label="MAC")
        ax.plot(chain2["timestamp"], chain2["power"], label="CHAIN2")
        plt.legend()
        plt.show()

    def run_interpolation(
        self,
        code_list: list[str],
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        save_db=False,
    ):
        if start_date is None:
            start_date = datetime.datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = datetime.datetime.now()
        s, e = start_date.timestamp(), end_date.timestamp()
        query_code_list = "','".join(code_list)
        query = (
            "select id, "
            "from_unixtime(timestamp) as timestamp, "
            "power as power "
            "from tab_chain2_data "
            f"where id in ('{query_code_list}') "
            f"and timestamp >= {s} "
            f"and timestamp < {e}"
        )
        df_chain2 = pl.read_database_uri(
            query, self.database_connector.connection_string
        )
        df_partitioned = df_chain2.partition_by("id", as_dict=True)
        return_batch = {}
        for code_tuple in df_partitioned:
            code: str = code_tuple[0]
            current_chain2 = df_partitioned[code_tuple]
            current_interpolated = self.interpolate_data(code, current_chain2)
            return_batch[code] = current_interpolated
        return return_batch

    def interpolate_data(
        self,
        code: str,
        chain2_df: pl.DataFrame,
    ):
        timestamp_second = pl.datetime_range(
            start=chain2_df["timestamp"].min(),
            end=chain2_df["timestamp"].max(),
            interval="1s",
            closed="right",
            eager=True,
        )
        ned_ts = timestamp_second.cast(pl.Int64).to_numpy() / 1e6
        chain_ts = chain2_df["timestamp"].cast(pl.Int64).to_numpy() / 1e6
        chain_power = chain2_df["power"].to_numpy().astype(float)

        interpolated_full = self.interpolator.predict(
            input_timestamps=chain_ts,
            power=chain_power,
            target_timestamps=ned_ts,
        ).astype(float)
        interpolated_full = (
            pl.from_numpy(interpolated_full)
            .with_columns(
                id=pl.lit(code), timestamp=timestamp_second, timestamp_second=ned_ts
            )
            .rename({"column_0": "power"})
        )
        return interpolated_full

    def run(self):
        query_chain_houses = (
            "select tc.id, "
            "tc.codice_luce as light_code, "
            "tc.codice_dispositivo as device_code "
            "from tab_chain2 tc "
            "inner join tab_abitazione ta on ta.id=tc.id_abitazione "
            "where ta.stato_attivazione_ned=1"
        )
        df_users = pl.read_database_uri(
            query_chain_houses, self.database_connector.connection_string
        )
        print(df_users)
        batch_size = 100
        now = datetime.datetime.now()
        if self.run_mode == "real-time":
            start, end = now - datetime.timedelta(minutes=30), now
        else:
            start, end = (
                (now - datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                now.replace(hour=0, minute=0, second=0, microsecond=0),
            )
        batches = [
            (
                df_users[i : i + batch_size]["device_code"].to_list(),
                start,
                end,
            )
            for i in range(0, len(df_users), batch_size)
        ]
        results = []
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as p:
                results = p.map(self.run_interpolation, batches)
        else:
            results = [self.run_interpolation(*batch) for batch in batches]


if __name__ == "__main__":
    pl.Config.set_tbl_rows(30)
    MAC().run()
