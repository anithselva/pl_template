from __future__ import annotations

import mlflow
import pytorch_lightning as pl
import tabulate
import torch
from torch import nn


class Abstract_Model(pl.LightningModule):
    def __init__(self, model, loss, metrics):
        super().__init__()

        self.model = model
        self.loss = loss
        self.metrics = metrics

    def forward(self, x):

        out = self.model(x)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x.float())
        loss = self.loss(logits, y)

        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.float())
        loss = self.loss(logits, y)

        return {'loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x.float())

        loss = self.loss(logits, y)
        prediction = torch.argmax(logits, dim=1)

        return {'loss': loss, 'preds': prediction, 'target': y}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mlflow.log_metric('train_loss', train_loss)

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mlflow.log_metric('val_loss', val_loss)

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['loss'] for x in outputs]).mean()
        preds = torch.cat([x['preds'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])
        metrics_dict = self._shared_eval_step(preds, target)

        metrics_dict['test_loss'] = test_loss.item()

        print(
            tabulate.tabulate(
                list(metrics_dict.items()), ['metric', 'value'],
            ),
        )

        mlflow.log_metrics(metrics_dict)

    def _shared_eval_step(self, preds, target):
        metrics_dict = {}

        for metrics in self.metrics:
            metrics_name = metrics.__name__
            metrics_val = metrics(preds, target)
            metrics_dict[metrics_name] = metrics_val

        return metrics_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
