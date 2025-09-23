from DataPreparation import TimeSeriesPreparation, UpScalingDataset, DataLoader
from Utils import ModelVisualizer, VisualizationCallback, ModelTrainingTesting
from CNN import CNNUpscaler

def train():
    batch_size = 128

    target_frequency = 30
    time_window_hours = 1
    seq_len = time_window_hours * 3600 // target_frequency
    seq_len = 60

    n_jobs = 12
    house_limit = 100
    days = 10

    print("Starting transformer-based time series upscaling...")
    tsp = TimeSeriesPreparation(
        to_normalize=True,
        limit=house_limit,
        n_days=days,
        down_sample_to=target_frequency,
        normalization_method="standard"
    )
    chain2_data, ned_data, power_scaling, time_delta_scaling = tsp.load_chain_2()

    train_dataset = UpScalingDataset(
        ned_data,
        chain2_data,
        sequence_len=seq_len,
        max_input_len=seq_len,  # Max irregular input length
        min_input_len=10,  # Min input length
        overlap_ratio=0.3,  # 30% overlap
        normalize=False,  # Already normalized
        phase="train",
        split_by_time=True,
        time_window_hours=time_window_hours,
    )
    val_dataset = UpScalingDataset(
        ned_data,
        chain2_data,
        sequence_len=seq_len,
        max_input_len=seq_len,  # Max irregular input length
        min_input_len=10,  # Min input length
        overlap_ratio=0.3,  # 30% overlap
        normalize=False,  # Already normalized
        phase="val",
        split_by_time=True,
        time_window_hours=time_window_hours,
    )
    print(f"Train Dataset: Samples={len(train_dataset.dataset)}")
    print(f"Val Dataset: Samples={len(val_dataset.dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_jobs,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Ensures consistent batch sizes
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    model = CNNUpscaler(
        input_dim=2,
        output_seq_len=seq_len,
        hidden_dim=batch_size
    )
    visualizer = ModelVisualizer(model)
    visualization_callback = VisualizationCallback(
        val_loader=val_loader,
        model_visualizer=visualizer,
        log_every_n_epochs=10,
    )
    mt = ModelTrainingTesting(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=500,
        method="regression",
        model_name=f"UpScalingTimeSeries",
        callbacks=[visualization_callback],
        monitor="val_recon_loss",
    )
    mt.train()


if __name__ == "__main__":
    train()
