import torch
import argparse

import dataset
import pytorch_lightning as pl
from utils.config import *
from src import create_model

from pytorch_lightning.loggers import MLFlowLogger

import mlflow

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    mlflow.set_tracking_uri("file:./experiments")
    mlflow.set_experiment(config.exp_name)
    mlflow.pytorch.autolog()
    mlflow.start_run()

    mlflow.log_params(pl.utilities.logger._flatten_dict(config))

    mlf_logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id,
    )

    # mlf_logger = MLFlowLogger(experiment_name=config.exp_name, 
    #                           tracking_uri="file:./experiments", 
    #                           artifact_location=config.out_dir)

    mc = create_model(config.model)

    train_dataset = dataset.MNIST(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=3,
                            shuffle=True,
                            num_workers=2)

    train_dataset = dataset.MNIST(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=3,
                            shuffle=True,
                            num_workers=2)

    test_dataset = dataset.MNIST(config, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=3,
                            shuffle=True,
                            num_workers=2)

    trainer = pl.Trainer(limit_train_batches=1, 
                         limit_val_batches=1, 
                         limit_test_batches=3, 
                         max_epochs=3,
                         logger=mlf_logger
                         )
    trainer.fit(mc, train_dataloader, train_dataloader)
    
    # mlf_logger.experiment.log_artifact(
    #     run_id=mlf_logger.run_id,
    #     local_path=checkpoint_callback.best_model_path
    #     )

    
    trainer.test(mc, test_dataloader)
    
    
    print(" *************************************** ")


if __name__ == "__main__" :
    main()