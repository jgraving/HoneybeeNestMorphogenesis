"""Training routine (PyTorch Lightning) for the spatial classifier.

Reproduces the paper's optimizer and schedule: AdamW (base lr 1e-3, weight
decay 1e-5, betas (0.9, 0.95)), class-embedding and positional-bias groups
trained at 10x the base lr, a 5-epoch linear warmup followed by cosine
annealing, batch size 8, 200 epochs, and a class-weighted cross-entropy loss
(weights = inverse training-set class frequencies).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from .dataset import GridDataset, compute_class_weights
from .models import SpatialClassifier


class SpatialClassifierModule(pl.LightningModule):
    """Thin Lightning wrapper around a ``SpatialClassifier``."""

    def __init__(self, model, criterion, learning_rate, weight_decay,
                 warmup_steps, cosine_steps):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

    def forward(self, grids):
        return self.model(grids)

    def training_step(self, batch, batch_idx):
        grids, _, _ = batch
        grids = grids.to(self.device, dtype=torch.long)
        logits = self(grids)
        loss = self.criterion(logits, grids)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        grids, _, _ = batch
        grids = grids.to(self.device, dtype=torch.long)
        logits = self(grids)
        loss = self.criterion(logits, grids)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.embedding_weights,
             "lr": self.learning_rate * 10},  # 10x base lr for the embedding
        ]
        if hasattr(self.model, "positional_embedding"):
            param_groups.append({"params": [self.model.positional_embedding],
                                 "lr": self.learning_rate * 10})
        param_groups.append({
            "params": [p for n, p in self.model.named_parameters()
                       if "embedding" not in n],
            "lr": self.learning_rate,
        })

        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=self.weight_decay,
                                      betas=(0.9, 0.95))
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1 / self.warmup_steps,
                        total_iters=self.warmup_steps),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.cosine_steps),
                ],
                milestones=[self.warmup_steps]),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


def train_spatial_classifier(hive_grids, seed: int = 42, **kwargs) -> SpatialClassifier:
    """Train one model over a list of per-colony (weeks, D, H, W) arrays.

    Defaults match the paper.  ``class_weights`` is computed here from the
    provided grids (inverse frequency) instead of relying on a global, as in
    the original notebook.
    """
    pl.seed_everything(seed, workers=True)

    batch_size = kwargs.get('batch_size', 8)
    embedding_dim = kwargs.get('embedding_dim', 128)
    kernel_size = kwargs.get('kernel_size', 3)
    dilation = kwargs.get('dilation', 1)
    max_epochs = kwargs.get('max_epochs', 200)
    learning_rate = kwargs.get('learning_rate', 1e-3)
    weight_decay = kwargs.get('weight_decay', 1e-5)
    warmup_epochs = kwargs.get('warmup_epochs', 5)
    device = kwargs.get('device', 'cpu')

    cosine_epochs = max_epochs - warmup_epochs

    dataset = GridDataset(hive_grids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

    # Class weighting: inverse training-set frequencies (computed, not global).
    class_weights = compute_class_weights(dataset).to(device)

    model = SpatialClassifier(
        num_classes=dataset.num_classes,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        depth=kwargs.get('depth', True),
        height=kwargs.get('height', True),
        width=kwargs.get('width', True),
        pos_embed=kwargs.get('pos_embed', False),
        context=kwargs.get('context', True),
        interaction=kwargs.get('interaction', 1),
        num_depth=kwargs.get('num_depth', dataset.shape[-3]),
        num_height=kwargs.get('num_height', dataset.shape[-2]),
        num_width=kwargs.get('num_width', dataset.shape[-1]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    total_batches = len(dataloader)
    warmup_steps = warmup_epochs * total_batches
    cosine_steps = cosine_epochs * total_batches

    spatial_module = SpatialClassifierModule(
        model=model, criterion=criterion,
        learning_rate=learning_rate, weight_decay=weight_decay,
        warmup_steps=warmup_steps, cosine_steps=cosine_steps,
    )

    if device.startswith("cuda") or (device == "cuda"):
        device_index = int(device.split(":")[1]) if ":" in device else 0
        accelerator = "gpu"
        devices = [device_index]
    elif device == "cpu":
        accelerator = "cpu"
        devices = None
    else:
        raise ValueError(f"Unsupported device string: {device}")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(spatial_module, train_dataloaders=dataloader)

    return model
