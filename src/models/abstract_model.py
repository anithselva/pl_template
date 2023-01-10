from __future__ import annotations

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

        logs = {'train_loss': loss}
        self.log('loss', loss)

        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x.float())
        loss = self.loss(logits, y)
        self.log('loss', loss)

        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x.float())

        loss = self.loss(logits, y)
        prediction = torch.argmax(logits, dim=1)

        return {'test_loss': loss, 'preds': prediction, 'target': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):

        preds = torch.cat([x['preds'] for x in outputs])
        target = torch.cat([x['target'] for x in outputs])
        metrics_dict = self._shared_eval_step(preds, target)

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        metrics_dict['avg_loss'] = avg_loss
        print(
            tabulate.tabulate(
                list(metrics_dict.items()), ['metric', 'value'],
            ),
        )

        return metrics_dict

    def _shared_eval_step(self, preds, target):
        metrics_dict = {}

        for metrics in self.metrics:
            metrics_name = metrics.__name__
            metrics_val = metrics(preds, target)
            metrics_dict[metrics_name] = metrics_val

        return metrics_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return
