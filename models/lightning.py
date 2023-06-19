
from typing import Any, Optional

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discovqa import DisCoVQA, VQEGSuggestion

class DisCoVQALightning(LightningModule):
    def __init__(self, batch_size: int=16, lr: float=1e-3, resolution: int=224, weight_decay: float=1e-3) -> None:
        super(DisCoVQALightning, self).__init__()

        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, 3, 16, resolution, resolution)

        self.neuralnet = DisCoVQA(d_model=512, num_heads=1, sample_rate=8, dropout=0.0)
        # Video Quality Predictor
        self.vqp = VQEGSuggestion(min_s=0, max_s=5)

        # weight initialization
        self._plane_initialization(self.neuralnet.tct)

    def forward(self, x):
        x = self.neuralnet(x)
        return self.vqp(x)

    def _shared_l1_loss(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        breakpoint()
        loss = F.l1_loss(y_hat, y)
        return loss
    
    def _draw_histogram(self) -> None:
        for name, param in self.neuralnet.named_parameters():
            if param.isnan().all():
                print(f"Layer: {name} is become nan")
                continue
            if param.view(-1).size(0) < 3:
                continue
            elif 'bias' in name:
                self.logger.experiment.add_histogram(f"bias/{name}", param, global_step=self.current_epoch)
            else:
                self.logger.experiment.add_histogram(f"weight/{name}", param, global_step=self.current_epoch)

    def _plane_initialization(self, model: nn.Module) -> None:
        for param in model.parameters():
            if param.dim() < 2:
                nn.init.normal_(param)
            else:
                nn.init.xavier_normal_(param)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._shared_l1_loss(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        loss = self._shared_l1_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank == 0:
            self._draw_histogram()

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.neuralnet.parameters(), 
                                      lr=self.hparams.lr, 
                                      weight_decay=self.hparams.weight_decay)
        return optimizer