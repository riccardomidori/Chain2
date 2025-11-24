import datetime
import random
import seaborn as sns
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import lightning
from config.Logger import XMLLogger
from pathlib import Path
import torch

class ModelVisualizer:
    """
    A class to visualize the model's predictions against the original data.
    """

    def __init__(self, model):
        """
        Initializes the visualizer with the trained model.

        Args:
            model (lightning.LightningModule): The trained PyTorch Lightning model.
        """
        self.model = model
        sns.set_style("whitegrid")
        np.random.seed(69)

    def plot_predictions(self, power, time_delta, mask, target):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(power, mask, target)
        power_cpu = power.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        prediction_cpu = prediction.detach().cpu().numpy()

        sample_idx = random.choices(range(0, power_cpu.shape[0]), k=5)

        target_cpu = target_cpu.squeeze(-1)
        prediction_cpu = prediction_cpu.squeeze(1)
        power_cpu = power_cpu.squeeze(-1)
        fig, ax = plt.subplots(5)
        for i, idx in enumerate(sample_idx):
            valid_power = power_cpu[idx]
            valid_prediction = prediction_cpu[idx]
            valid_prediction = valid_prediction + valid_power

            # Plot the data on the respective subplots.
            # ax[0].plot(valid_power, "o-", color="skyblue", label="Chain2 Input Power")
            ax[i].plot(target_cpu[idx], "ro--", label="NED_D Target")
            ax[i].plot(valid_prediction, "go-", label="Model Prediction")
            ax[i].legend()

        ax[0].set_title("Model Predictions vs Target Data")
        ax[0].set_ylabel("Power (kW)")
        fig.tight_layout()
        plt.show()


class VisualizationCallback(Callback):
    def __init__(
        self,
        val_loader: DataLoader,
        model_visualizer: ModelVisualizer,
        log_every_n_epochs: int = 1,
    ):
        """
        Args:
            val_loader (DataLoader): The DataLoader for validation data.
            model_visualizer (ModelVisualizer): An instance of the ModelVisualizer class.
            log_every_n_epochs (int): The frequency of visualization.
        """
        self.val_loader = val_loader
        self.visualizer = model_visualizer
        self.log_every_n_epochs = log_every_n_epochs
        # Use an iterator to avoid re-initializing the loader every time
        self.val_iter = iter(val_loader)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            trainer.current_epoch % self.log_every_n_epochs == 0
            and trainer.current_epoch > 0
        ):
            try:
                # Get a validation batch
                batch = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                batch = next(self.val_iter)

            power, target, mask = batch
            power = power.to(pl_module.device)
            target = target.to(pl_module.device)
            mask = mask.to(pl_module.device)
            # Create visualization for the first sample in the batch
            # We pass the full batch and let the visualizer handle it
            self.visualizer.plot_predictions(
                power,
                None,
                mask,
                target,
            )


class ModelTrainingTesting:
    def __init__(
        self,
        model: lightning.LightningModule,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        seq_len=4,
        horizon=1,
        batch_size=16,
        epochs=150,
        callbacks: list = None,
        normalize=True,
        show=False,
        method="regression",
        limit=None,
        model_name="nn",
        min_delta=1e-4,
        patience=10,
        monitor="val_loss",
    ):
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model_name = model_name
        self.limit = limit
        self.method = method
        self.model = model
        self.normalize = normalize
        self.show = show

        self.logger = XMLLogger("ModelTrainingTesting").logger
        self.model_checkpoint = ModelCheckpoint(
            dirpath="checkpoints",
            filename=model_name + "-{val_loss:.4f}",
            save_top_k=1,
            monitor=monitor,
            mode="min"
            if self.model.hparams.method in ["regression", "forecasting"]
            else "max",
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=5,
            verbose=True
        )
        if callbacks is None:
            callbacks = [
                self.model_checkpoint,
                early_stopping
            ]
        else:
            callbacks = callbacks + [self.model_checkpoint, early_stopping]
        self.callbacks = callbacks
        self.epochs = epochs
        self.df_scaled = None
        self.batch_size = batch_size
        self.horizon = horizon
        self.history = seq_len

        self.trainer = lightning.Trainer(
            max_epochs=self.epochs,
            callbacks=self.callbacks,
        )

    def train(self):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.test_dataloader,
        )

    def visualize_importance(self, feature_names=None):
        from captum.attr import IntegratedGradients

        x_val, y_val = next(iter(self.test_dataloader))
        model = self.model.eval()
        x_val = x_val.to(model.device)
        x_val = x_val[: self.batch_size]

        ig = IntegratedGradients(model)
        if len(y_val.shape) > 1:
            attr, delta = ig.attribute(
                inputs=x_val, target=0, return_convergence_delta=True
            )
        else:
            attr, delta = ig.attribute(inputs=x_val, return_convergence_delta=True)
        avg_attr = (
            attr.mean(dim=(0, 1)).detach().cpu().numpy()
        )  # mean over samples & time

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(avg_attr)), avg_attr, color="skyblue")
        if feature_names is not None:
            plt.yticks(np.arange(len(avg_attr)), feature_names)
        plt.xlabel("Average Feature Attribution")
        plt.title("Feature Importance via Integrated Gradients")
        plt.gca().invert_yaxis()
        plt.grid(True, axis="x")
        plt.tight_layout()
        plt.show()

    def test(self, class_model, name=None):
        if name is None:
            check_points = [
                int(p.name.split("-v")[1].split(".")[0]) if "-v" in p.name else 0
                for p in Path("checkpoints").glob("*.ckpt")
            ]
            v = max(check_points)
            name = (
                f"{self.model_name}-v{v}.ckpt" if v > 0 else f"{self.model_name}.ckpt"
            )
        trained_model = class_model.load_from_checkpoint(
            f"checkpoints/{name}",
            method=self.model.method,
        )
        with torch.no_grad():
            for item in self.test_dataloader.dataset:
                x, y = item
                x_input = x.reshape(1, 1, -1)
                y_hat = trained_model(x_input)
                print(y_hat.item(), y)
