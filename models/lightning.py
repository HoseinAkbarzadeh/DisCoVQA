
from typing import Any, Optional

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F

from models.discovqa import DisCoVQA, VQEGSuggestion

class DisCoVQALightning(LightningModule):
    def __init__(self, batch_size: int=16, lr: float=1e-3, resolution: int=224, weight_decay: float=1e-3) -> None:
        super(DisCoVQALightning, self).__init__()

        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, 3, 16, 224, 224)

        self.neuralnet = DisCoVQA(d_model=512, num_heads=1, sample_rate=8, dropout=0.0)
        # Video Quality Predictor
        self.vqp = VQEGSuggestion(min_s=0, max_s=5)

    def forward(self, x):
        breakpoint()
        x = self.neuralnet(x)
        breakpoint()
        return self.vqp(x)

    def _shared_l1_loss(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(-1)
        loss = F.l1_loss(y_hat, y)
        return loss
    
    def _draw_histogram(self) -> None:
        for name, param in self.neuralnet.named_parameters():
            if 'bias' in name:
                self.logger.experiment.add_histogram(f"bias/{name}", param, global_step=self.current_epoch)
            else:
                self.logger.experiment.add_histogram(f"weight/{name}", param, global_step=self.current_epoch)
    
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