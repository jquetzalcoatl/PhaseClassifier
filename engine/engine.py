"""
Attributes:
    _config (OmegaConf or None): Configuration object for the engine.
    _model (torch.nn.Module or None): The model used by the engine.
    _optimiser (torch.optim.Optimizer or None): Optimizer for training the model.
    _data_mgr (object or None): Data manager providing data loaders.
    _device (torch.device or None): Device on which computations are performed.
    best_val_loss (float): Best validation loss achieved.
Properties:
    model (torch.nn.Module): Gets or sets the model.
    optimiser (torch.optim.Optimizer): Gets or sets the optimizer.
    data_mgr (object): Gets or sets the data manager.
    device (torch.device): Gets or sets the device.
    model_creator (object): Gets or sets the model creator.
Methods:
    __init__(cfg=None, **kwargs): Initializes the Engine with optional configuration.
    _save_model(name="blank"): Saves the model state using the model creator and returns the path.
    fit_model(epoch): Trains the model for one epoch and logs batch losses.
    eval_model(data_loader, epoch): Evaluates the model on the given data loader and logs the average loss.
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

from omegaconf import OmegaConf

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

class Engine():
    def __init__(self, cfg=None, **kwargs):
        self._config = cfg
        
        self._model = None
        self._optimiser = None
        self._data_mgr = None
        self._device = None
        self.best_val_loss = float("inf")

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimiser(self):
        return self._optimiser
    
    @optimiser.setter
    def optimiser(self, optimiser):
        self._optimiser = optimiser

    @property
    def data_mgr(self):
        return self._data_mgr
    
    @data_mgr.setter   
    def data_mgr(self,data_mgr):
        assert data_mgr is not None, "Empty Data Manager"
        self._data_mgr=data_mgr
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device=device

    @property
    def model_creator(self):
        return self._model_creator
    
    @model_creator.setter
    def model_creator(self, model_creator):
        assert model_creator is not None
        self._model_creator = model_creator

    def _save_model(self, name="blank"):
        config_string = "_".join(str(i) for i in [self._config.model.model_name,f'{name}'])
        config_path = self._model_creator.save_state(config_string)
        return config_path
    
    def fit_model(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self._model.train()
        for i, (x, y, t) in enumerate(self._data_mgr.train_loader):
            self.optimiser.zero_grad()
            output = self._model(x.to(self.device))
            loss = self._model.loss(output, y.to(self.device))
            loss.backward()
            self.optimiser.step()

            if (i % log_batch_idx) == 0:
                logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                    i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                    loss.item()))
                wandb.log({"loss": loss.item()})

    def eval_model(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_eval, 1)
        self._model.eval()
        self.total_loss = 0
        with torch.no_grad():
            for i, (x, y, t) in enumerate(data_loader):
                output = self._model(x.to(self.device))
                loss = self._model.loss(output, y.to(self.device))
                self.total_loss += loss.item()

            self.total_loss /= len(data_loader)
            
            logger.info('Epoch: {} \t Batch Loss: {:.4f}'.format(epoch,
                    self.total_loss))
            wandb.log({"val_loss": self.total_loss})