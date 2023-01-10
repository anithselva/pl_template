from __future__ import annotations

import argparse

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger

import dataset
from src import create_model
from utils.config import *


def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format',
    )
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    mlf_logger = MLFlowLogger(
        experiment_name=config.exp_name, save_dir='./experiments',
    )
    mlflow.pytorch.autolog()

    log_dir = os.path.join(
        mlf_logger.save_dir,
        mlf_logger.experiment_id, mlf_logger.run_id,
    )
    setup_logging(log_dir)

    logging.getLogger().info('Configuration Complete')
    logging.getLogger().info('Starting Pipeline')

    mc = create_model(config.model)

    train_dataset = dataset.MNIST(config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=2,
    )

    train_dataset = dataset.MNIST(config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=2,
    )

    test_dataset = dataset.MNIST(config, 'test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=3, shuffle=True, num_workers=2,
    )

    trainer = pl.Trainer(
        limit_train_batches=1, limit_val_batches=1, limit_test_batches=3,
        max_epochs=3, logger=mlf_logger,
    )
    trainer.fit(mc, train_dataloader, train_dataloader)

    trainer.test(mc, test_dataloader)

    print(' *************************************** ')


if __name__ == '__main__':
    main()
