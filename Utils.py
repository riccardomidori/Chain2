from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import Tensor, no_grad
from random import sample
import numpy as np
import lightning
from config.Logger import XMLLogger
from pathlib import Path


class ModelVisualizer:
    """
    A class to visualize the model's predictions against the original data.
    """

    def __init__(self, model, time_interval=30):
        """
        Initializes the visualizer with the trained model.

        Args:
            model (lightning.LightningModule): The trained PyTorch Lightning model.
        """
        self.time_interval = time_interval
        self.model = model

    def plot_predictions(
        self,
        ts,
        power: Tensor,
        time_delta: Tensor,
        mask: Tensor,
        target: Tensor,
        plot_file_path: str = "predictions.png",
    ):
        """
        Generates and saves a plot showing the model's predictions.

        The plot includes the raw Chain2 input power, the NED_D target power,
        and the model's predicted NED_D power.

        Args:
            power (torch.Tensor): The input power tensor from a single batch.
            time_delta (torch.Tensor): The input time_delta tensor from a single batch.
            mask (torch.Tensor): The mask tensor for the batch.
            target (torch.Tensor): The true NED_D power values for the batch.
            plot_file_path (str): The file path to save the plot.
        """
        self.model.eval()
        with no_grad():
            # Get the model's predictions
            prediction = self.model(power, time_delta, mask)

        # Move tensors to CPU for plotting
        ts_cpu = Tensor.cpu(ts)
        power_cpu = Tensor.cpu(power)
        time_delta_cpu = Tensor.cpu(time_delta)
        target_cpu = Tensor.cpu(target)
        prediction_cpu = Tensor.cpu(prediction)
        mask_cpu = Tensor.cpu(mask)
        # The input data (Chain2) is irregularly sampled, while the output is regular.
        # We need to compute the cumulative time for plotting the Chain2 data.
        # Note: The time_delta here is the *input* to the model, which is irregular.
        # The output target and prediction are regularly sampled.

        # We will plot the first sample in the batch for clarity.
        sample_idx = sample(range(ts_cpu.shape[0]), 1)[0]

        # Handle the case where the output is a single value (seq_len=1)
        if target_cpu.ndim == 1:
            target_cpu = np.expand_dims(target_cpu, axis=1)
            prediction_cpu = np.expand_dims(prediction_cpu, axis=1)

        # Create time axis for the irregular Chain2 data
        # We filter out the padded values
        valid_indices = mask_cpu[sample_idx]
        valid_power = power_cpu[sample_idx]
        valid_time_deltas = time_delta_cpu[sample_idx]

        # Calculate cumulative time for the Chain2 data
        cumulative_time_chain2 = np.cumsum(valid_time_deltas)
        # Shift to start at 0
        cumulative_time_chain2 = np.insert(cumulative_time_chain2, 0, 0)[:-1]

        # Create time axis for the regular NED_D data
        # Assumes a consistent 30s interval as per your setup
        ned_d_interval = self.time_interval
        cumulative_time_nedd = np.arange(target_cpu.shape[1]) * ned_d_interval

        # Create the plot
        plt.figure(figsize=(12, 7))
        plt.style.use("seaborn-v0_8-whitegrid")

        # Plot the original Chain2 input power
        plt.plot(
            ts_cpu[sample_idx],
            valid_power,
            "o-",
            color="skyblue",
            label="Chain2 Input Power",
        )

        # Plot the true NED_D target power
        plt.plot(
            ts_cpu[sample_idx],
            target_cpu[sample_idx],
            "ro--",
            markersize=5,
            label="NED_D Target Power",
        )

        # Plot the model's prediction
        plt.plot(
            ts_cpu[sample_idx],
            prediction_cpu[sample_idx],
            "go-",
            markersize=5,
            label="Model Prediction",
        )

        plt.title("Model Predictions vs. Target Data")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Power (kW)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        # plt.savefig(plot_file_path)
        # print(f"Plot saved to {plot_file_path}")
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

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            trainer.current_epoch % self.log_every_n_epochs == 0
            and trainer.current_epoch > 0
        ):
            try:
                # Get a validation batch
                batch = next(self.val_iter)
            except StopIteration:
                # Re-initialize the iterator if we've gone through the whole loader
                self.val_iter = iter(self.val_loader)
                batch = next(self.val_iter)

                # Unpack the batch tuple. Make sure the order matches your Dataset.__getitem__
            power, target, mask, time_delta, ts = batch
            ts = ts.to(pl_module.device)
            power = power.to(pl_module.device)
            time_delta = time_delta.to(pl_module.device)
            mask = mask.to(pl_module.device)
            target = target.to(pl_module.device)

            # Create visualization for the first sample in the batch
            # We pass the full batch and let the visualizer handle it
            self.visualizer.plot_predictions(
                ts,
                power,
                time_delta,
                mask,
                target,
                plot_file_path=f"epoch_{trainer.current_epoch}_predictions.png",
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
        # self.model_checkpoint = ModelCheckpoint(
        #     dirpath="checkpoints",
        #     filename=model_name + f"{monitor}:.2f",
        #     save_top_k=1,
        #     monitor=monitor,
        #     mode="min"
        #     if self.model.hparams.method in ["regression", "forecasting"]
        #     else "max",
        # )

        if callbacks is None:
            callbacks = []
            # if self.model.hparams.method in ["regression", "forecasting"]:
            #     callbacks = [
            #         self.model_checkpoint,
            #     ]
            # else:
            #     callbacks = [
            #         self.model_checkpoint,
            #     ]
        # else:
        #     callbacks = callbacks + [self.model_checkpoint]
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
        with no_grad():
            for item in self.test_dataloader.dataset:
                x, y = item
                x_input = x.reshape(1, 1, -1)
                y_hat = trained_model(x_input)
                print(y_hat.item(), y)
