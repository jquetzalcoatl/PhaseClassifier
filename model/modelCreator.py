"""
Classes:
--------
ModelCreator:
    Interface for initializing, saving, and loading machine learning models.
    Methods
    -------
    __init__(cfg=None):
        Initializes the ModelCreator with an optional configuration.
    init_model():
        Instantiates the model specified in the configuration.
    model (property):
        Returns the initialized model. Raises an assertion error if the model is not defined.
    model (setter):
        Sets the internal model instance.
    save_state(cfg_string='test'):
        Saves the state dictionaries of all model modules and the configuration to disk.
    load_state(run_path, device):
        Loads the state dictionaries for all model modules from a checkpoint file.
"""

import os
import torch
import wandb

from omegaconf import OmegaConf

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

#import defined models
from model.transformer import Classifier
# import uuid
# from datetime import datetime

_MODEL_DICT={
    "transformer": Classifier,
}

class ModelCreator():
    def __init__(self, cfg=None):
        self._config=cfg
        self._model=None
    
    def init_model(self):
        logger.info("::Creating Model")
        self._model = _MODEL_DICT[self._config.model.model_name](cfg=self._config)
        return self._model

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self,model):
        self._model=model
    
        
    def save_state(self, cfg_string='test'):
        if wandb.run is not None and self._config.wandb.mode != "disabled" and self._config.load_state == 0:
            save_dir = wandb.run.dir
            logger.info(f"Using WandB run directory: {save_dir}")
        elif wandb.run is not None and self._config.wandb.mode != "disabled" and self._config.load_state == 1:
            save_dir = self._config.run_path.split('files')[0] + 'files/'
        else:
            # create a short unique string (timestamp + 8 hex chars)
            # unique = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            unique = "tmp"
            if hasattr(self._config, 'dir'):
                logger.info(f"Using config run directory: {self._config.dir}")
                # append unique string to the configured run directory
                save_dir = os.path.join(self._config.dir, unique)
            else:
                logger.warning("No config run directory found, using current working directory.")
                save_dir = os.path.join(os.getcwd(), unique)

        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"{cfg_string}.pth")
        logger.info(f"Saving model state to {path}")

        modules = list(self._model._modules.keys())
        state_dict = {module: getattr(self._model, module).state_dict() for module in modules}

        torch.save(state_dict, path)

        config_path = os.path.join(save_dir, f"{cfg_string}_config.yaml")
        self._config.run_path = path
        self._config.config_path = config_path
        OmegaConf.save(self._config, config_path, resolve=True)
        return config_path
        
    def load_state(self, run_path, device):
        logger.info("Loading state")
        model_loc = run_path
        
        # Open a file in read-binary mode
        with open(model_loc, 'rb') as f:
            # Interpret the file using torch.load()
            checkpoint=torch.load(f, map_location=device)

            logger.info("Loading weights from file : {0}".format(run_path))
            
            local_module_keys=list(self._model._modules.keys())
            for module in checkpoint.keys():
                if module in local_module_keys:
                    print("Loading weights for module = ", module)
                    getattr(self._model, module).load_state_dict(checkpoint[module])


if __name__=="__main__":
    logger.info("Start!")
    mm=ModelCreator()
    logger.info("End!")