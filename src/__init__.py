from __future__ import annotations

from .losses import *
from .metrics import *
from .models import *
from .models.abstract_model import Abstract_Model


def create_model(model_config):

    model_kw = model_config._model.args
    model = getattr(models, model_config._model.name)(**model_kw)

    loss_kw = model_config.loss.args
    loss = getattr(losses, model_config.loss.type)(**loss_kw)

    metric = [getattr(metrics, metric) for metric in model_config.metrics]

    mc = Abstract_Model(model, loss, metric)

    return mc
