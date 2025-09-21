import torch
import torch.nn as nn
import lightning as pl
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


class LSTMUpscaler(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 2,       # (power, time_delta)
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_seq_len: int = 120,
        lr: float = 1e-3,
        method="regression"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=1,   # feeding last prediction (or teacher forcing)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.lr = lr
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, power, time_deltas, target=None, teacher_forcing_ratio=0.5):
        """
        x: [batch, seq_len, input_dim] (power, time_delta)
        target: [batch, out_seq_len, 1]
        """
        x = torch.cat([power, time_deltas], dim=-1)
        batch_size = x.size(0)
        out_seq_len = self.hparams.output_seq_len

        # Encode irregular inputs
        _, (h, c) = self.encoder(x)

        # Prepare decoder input (start token = 0)
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)

        outputs = []
        for t in range(out_seq_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc_out(out)  # [batch, 1, 1]
            outputs.append(pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1]  # use true value
            else:
                decoder_input = pred.detach()

        return torch.cat(outputs, dim=1)  # [batch, out_seq_len, 1]

    def training_step(self, batch, batch_idx):
        power, y, mask, time_deltas, ts = batch
        y_hat = self(power, time_deltas, y)
        loss = nn.MSELoss()(y_hat, y)
        self.train_mse(y_hat.squeeze(-1), y.squeeze(-1))
        self.train_mae(y_hat.squeeze(-1), y.squeeze(-1))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", self.train_mse, prog_bar=True)
        self.log("train_mae", self.train_mae)
        return loss

    def validation_step(self, batch, batch_idx):
        power, y, mask, time_deltas, ts = batch
        y_hat = self(power, time_deltas)
        loss = nn.MSELoss()(y_hat, y)
        self.val_mse(y_hat.squeeze(-1), y.squeeze(-1))
        self.val_mae(y_hat.squeeze(-1), y.squeeze(-1))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", self.val_mse, prog_bar=True)
        self.log("val_mae", self.val_mae)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
