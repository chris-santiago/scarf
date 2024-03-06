from typing import Optional

import pytorch_metric_learning.losses as losses
import torch
import torch.nn as nn

from scarf.models.base import BaseModule


class MaskGenerator(nn.Module):
    """Module for generating Bernoulli mask."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor):
        """Generate Bernoulli mask."""
        p_mat = torch.ones_like(x) * self.p
        return torch.bernoulli(p_mat)


class PretextGenerator(nn.Module):
    """Module for generating training pretext."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def shuffle(x: torch.tensor):
        """Shuffle each column in a tensor."""
        m, n = x.shape
        x_bar = torch.zeros_like(x)
        for i in range(n):
            idx = torch.randperm(m)
            x_bar[:, i] += x[idx, i]
        return x_bar

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """Generate corrupted features and corresponding mask."""
        shuffled = self.shuffle(x)
        corrupt_x = x * (1.0 - mask) + shuffled * mask
        return corrupt_x


class LinearLayer(nn.Module):
    """
    Module to create a sequential block consisting of:

        1. Linear layer
        2. (optional) Batch normalization layer
        3. ReLu activation layer
    """

    def __init__(self, input_size: int, output_size: int, batch_norm: bool = False):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        if batch_norm:
            self.model = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
            )
        else:
            self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x: torch.tensor):
        """Run inputs through linear block."""
        return self.model(x)


class LazyMLP(nn.Module):
    def __init__(self, n_layers: int, dim_hidden: int, batch_norm: bool = False):
        super().__init__()
        self.n_layers = n_layers
        self.dim_hidden = dim_hidden
        if batch_norm:
            lazy_block = nn.Sequential(
                nn.LazyLinear(dim_hidden),
                nn.BatchNorm1d(dim_hidden),
                nn.ReLU(),
            )
        else:
            lazy_block = nn.Sequential(nn.LazyLinear(dim_hidden), nn.ReLU())

        self.model = nn.Sequential(
            lazy_block,
            *[LinearLayer(dim_hidden, dim_hidden, batch_norm) for _ in range(n_layers - 1)]
        )

    def forward(self, x: torch.tensor):
        """Run inputs through linear block."""
        return self.model(x)


class SCARFEncoder(BaseModule):
    def __init__(
        self,
        dim_hidden: int = 256,
        n_encoder_layers: int = 4,
        n_projection_layers: int = 2,
        p_mask: float = 0.6,
        loss_func: nn.Module = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=1.0)),
        optim: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__(loss_func=loss_func, optim=optim, scheduler=scheduler)

        self.save_hyperparameters()

        self.get_mask = MaskGenerator(p=p_mask)
        self.corrupt = PretextGenerator()

        self.encoder = LazyMLP(n_layers=n_encoder_layers, dim_hidden=dim_hidden)
        self.projection = LazyMLP(n_layers=n_projection_layers, dim_hidden=dim_hidden)

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def training_step(self, batch, idx):
        x = batch[0]
        mask = self.get_mask(x)
        x_corrupt = self.corrupt(x, mask)
        enc_x, enc_corrupt = self(x), self(x_corrupt)
        proj_x, proj_corrupt = self.projection(enc_x), self.projection(enc_corrupt)
        loss = self.loss_func(proj_corrupt, proj_x)

        self.log(
            "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )
        metrics = {"train-loss": loss}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, idx):
        x_corrupt, x = batch
        enc_x, enc_corrupt = self(x), self(x_corrupt)
        proj_x, proj_corrupt = self.projection(enc_x), self.projection(enc_corrupt)
        loss = self.loss_func(proj_corrupt, proj_x)

        self.log(
            "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False
        )
        metrics = {"valid-loss": loss}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss
