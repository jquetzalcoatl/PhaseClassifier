"""
This script serves as the main entry point for training and evaluating models in the PhaseClassifier project.
It leverages Hydra for configuration management, PyTorch for deep learning, and integrates with Weights & Biases for experiment tracking.
Main Components:
----------------
- Configuration loading and management via Hydra and OmegaConf.
- Device selection logic for CPU/GPU training.
- Model instantiation and setup using custom ModelCreator.
- Data management via DataManager.
- Training and evaluation loop managed by Engine.
- Logging and experiment tracking.
Functions:
----------
- main(cfg=None): Hydra main function. Loads configuration, sets up model and engine, and starts training.
- set_device(config=None): Determines and sets the computation device (CPU/GPU) based on configuration.
- setup_model(config=None): Initializes data manager, model, engine, optimizer, and assigns them to the engine.
- run(engine, _callback=lambda _: False): Runs the training and evaluation loop for the specified number of epochs.
- callback(engine, epoch): Placeholder for a callback function to be used during training.
Usage:
------
Run this script directly to start the training process with the specified configuration.
Note:
-----
- Custom modules (DataManager, Engine, ModelCreator) are expected to be implemented in the PhaseClassifier package.
- The script assumes configuration files are located in the '../config' directory.
"""
import os

import torch
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

# PyTorch imports
from torch import device
from torch.nn import DataParallel
from torch.cuda import is_available
    
# Weights and Biases
import wandb

#self defined imports
from PhaseClassifier import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManager
from engine.engine import Engine
from model.modelCreator import ModelCreator


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg=None):
    mode = cfg.wandb.mode
    if cfg.load_state:
        pass
        # logger.info(f"Loading config from {cfg.config_path}")
        # engine = load_model_instance(cfg)
        # cfg = engine._config
    else:
        engine = setup_model(config=cfg)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
        wandb.watch(engine.model)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run(engine)

def set_device(config=None):
    if (config.device == 'gpu') and config.gpu_list:
        logger.info('Requesting GPUs. GPU list :' + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in list(config.gpu_list)]
        logger.info("Main GPU : " + devids[0])
        
        if is_available():
            print(devids[0])
            dev = device(devids[0])
            if len(devids) > 1:
                logger.info(f"Using DataParallel on {devids}")
                model = DataParallel(model, device_ids=list(config.gpu_list))
            logger.info("CUDA available")
        else:
            dev = device('cpu')
            logger.info("CUDA unavailable")
    else:
        logger.info('Requested CPU or unable to use GPU. Setting CPU as device.')
        dev = device('cpu')
    return dev


def setup_model(config=None):
    """
    Run m
    """
    dataMgr = DataManager(config)

    #create model handling object
    modelCreator = ModelCreator(config)

    #instantiate the chosen model
    #loads from file 
    model=modelCreator.init_model()
    # model.print_model_info()

    # Load the model on the GPU if applicable
    dev = set_device(config)
        
    # Send the model to the selected device
    model.to(dev)

    # For some reason, need to use postional parameter cfg instead of named parameter
    # with updated Hydra - used to work with named param but now is cfg=None 
    engine=instantiate(config.engine, config)
    #add dataMgr instance to engine namespace
    engine.data_mgr=dataMgr
    #add device instance to engine namespace
    engine.device=dev    
    #instantiate and register optimisation algorithm
    engine.optimiser = torch.optim.Adam(model.parameters(),
                                        lr=config.engine.learning_rate)
    #add the model instance to the engine namespace
    engine.model = model
    # add the modelCreator instance to engine namespace
    engine.model_creator = modelCreator
    
    return engine

def run(engine, _callback=lambda _: False):
    logger.info("Training AutoEncoder")
    for epoch in range(engine._config.epoch_start, engine._config.n_epochs):
        engine.fit_model(epoch)

        engine.eval_model(engine.data_mgr.val_loader, epoch)

        if (epoch+1) % 10 == 0:
            engine._save_model(name=str(epoch))
        
        if _callback(engine):
            break
    engine._save_model(name="Final")

def callback(engine, epoch):
    """
    Callback function to be used with the engine.
    """
    pass

def load_model_instance(cfg, adjust_epoch_start=True):
    config = OmegaConf.load(cfg.config_path)
    if adjust_epoch_start:
        # Adjust the epoch start based on the run_path
        config.epoch_start = int(config.run_path.split("_")[-1].split(".")[0]) + 1
    config.gpu_list = cfg.gpu_list
    config.load_state = cfg.load_state
    self = setup_model(config)
    self._model_creator.load_state(config.run_path, self.device)
    return self

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")