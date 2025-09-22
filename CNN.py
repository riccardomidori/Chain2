import torch
import torch.nn as nn
import lightning
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class CNNUpscaler(lightning.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,  # (power, time_delta)
        hidden_dim: int = 64,
        num_layers: int = 5,
        kernel_size: int = 3,
        output_seq_len: int = 120,
        lr: float = 1e-3,
        method="regression",
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding="same")
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim

        self.conv_net = nn.Sequential(*layers)

        # Map from hidden channels → output length
        self.fc_out = nn.Linear(hidden_dim, output_seq_len)

        # Metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, power, time_deltas, mask):
        """
        power: [B, seq_len, 1]
        time_deltas: [B, seq_len, 1]
        mask: [B, seq_len]  (ignored here, because CNN expects fixed-length input)
        """
        # Concatenate features along channel dimension
        x = torch.cat([power, time_deltas], dim=-1)  # [B, seq_len, 2]
        x = x.permute(0, 2, 1)  # [B, channels=2, seq_len]

        h = self.conv_net(x)  # [B, hidden_dim, seq_len]
        h = h.mean(dim=-1)  # global average pooling → [B, hidden_dim]

        out = self.fc_out(h)  # [B, output_seq_len]
        return out.unsqueeze(-1)  # [B, output_seq_len, 1]

    def do_step(self, batch, batch_idx, is_train=True):
        power, y, mask, time_deltas, ts = batch
        y_hat = self(power, time_deltas, mask)
        loss = nn.MSELoss()(y_hat, y)
        if is_train:
            self.train_mse(y_hat.squeeze(-1), y.squeeze(-1))
            self.train_mae(y_hat.squeeze(-1), y.squeeze(-1))
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_mse", self.train_mse, prog_bar=True)
            self.log("train_mae", self.train_mae)
        else:
            self.val_mse(y_hat.squeeze(-1), y.squeeze(-1))
            self.val_mae(y_hat.squeeze(-1), y.squeeze(-1))
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_mse", self.val_mse, prog_bar=True)
            self.log("val_mae", self.val_mae)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, batch_idx, is_train=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
