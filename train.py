from __future__ import annotations

import argparse
from pathlib import Path

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

    save_dir = './experiments'
    mlflow.set_tracking_uri(save_dir)
    experiment = mlflow.get_experiment_by_name(config.exp_name)
    if not experiment:
        exp_id = mlflow.create_experiment(
            config.exp_name, artifact_location=Path.cwd().joinpath(save_dir).as_uri(),
        )
    else:
        exp_id = experiment.experiment_id

    mlflow.start_run(experiment_id=exp_id)
    run = mlflow.active_run()
    exp_id, run_id = run.info.experiment_id, run.info.run_uuid
    log_dir = os.path.join(save_dir, exp_id, run_id, 'artifacts')
    setup_logging(log_dir)

    logging.getLogger().info('Configuration Complete')
    logging.getLogger().info('Starting Pipeline')

    mc = create_model(config.model)

    train_dataset = dataset.MNIST(config)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2,
    )

    val_dataset = dataset.MNIST(config)
    val_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2,
    )

    test_dataset = dataset.MNIST(config, 'test')
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=True, num_workers=2,
    )

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(mc, train_dataloader, val_dataloader)
    trainer.test(mc, test_dataloader)
    mlflow.end_run()

    print(' *************************************** ')


if __name__ == '__main__':
    main()
