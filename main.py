from DataPreparation import TimeSeriesPreparation, UpScalingDataset, DataLoader
from Utils import ModelVisualizer, VisualizationCallback, ModelTrainingTesting
import torch
from ResidualUpscaler import ResidualUpscaler

torch.set_float32_matmul_precision("medium")

TARGET_FREQUENCY = 5
TIME_WINDOW_MINUTES = 30
SEQ_LEN = TIME_WINDOW_MINUTES * 60 // TARGET_FREQUENCY
BATCH_SIZE = 128
N_JOBS = 1
HOUSE_LIMIT = 100
DAYS = 10
LOADING_RATIO = 0.4

def train():
    print("Starting time series up-scaling")
    tsp = TimeSeriesPreparation(
        to_normalize=True,
        limit=HOUSE_LIMIT,
        n_days=DAYS,
        down_sample_to=TARGET_FREQUENCY,
        normalization_method="standard",
    )
    chain2_data, ned_data = tsp.load_chain_2(ratio=LOADING_RATIO)

    train_dataset = UpScalingDataset(
        ned_data,
        chain2_data,
        sequence_len=SEQ_LEN,
        max_input_len=SEQ_LEN,
        min_input_len=10,
        overlap_ratio=0.8,
        normalize=False,  # Already normalized
        phase="train",
        split_by_time=True,
        show=False,
        to_interpolate=True,
        only_spike=False,
        split_ratio=0.7
    )
    val_dataset = UpScalingDataset(
        ned_data,
        chain2_data,
        sequence_len=SEQ_LEN,
        max_input_len=SEQ_LEN,
        min_input_len=10,
        overlap_ratio=0.7,
        normalize=False,  # Already normalized
        phase="val",
        split_by_time=True,
        show=False,
        to_interpolate=True,
        only_spike=True,
        split_ratio=0.7
    )
    print(f"Train Dataset: Samples={len(train_dataset.dataset)}")
    print(f"Val Dataset: Samples={len(val_dataset.dataset)}")

    if any([len(train_dataset) < 1, len(val_dataset) < 1]):
        print("Train or validation datasets are empty")
        exit()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_JOBS,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Ensures consistent batch sizes
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=N_JOBS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    model = ResidualUpscaler(
        input_dim=1,
        hidden_dim=BATCH_SIZE,
        num_blocks=4
    )
    visualizer = ModelVisualizer(model)
    visualization_callback = VisualizationCallback(
        val_loader=val_loader,
        model_visualizer=visualizer,
        log_every_n_epochs=5,
    )
    callbacks = []
    mt = ModelTrainingTesting(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=500,
        method="regression",
        model_name=f"ResidualUpscaler_{SEQ_LEN}",
        callbacks=callbacks,
        monitor="val_loss",
    )
    mt.train()


if __name__ == "__main__":
    train()
