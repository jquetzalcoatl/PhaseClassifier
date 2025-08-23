"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

from omegaconf import OmegaConf

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

class Engine():
    def __init__(self, cfg=None, **kwargs):
        super(Engine,self).__init__()

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
        for i, (x, y) in enumerate(self._data_mgr.train_loader):
            self.optimiser.zero_grad()
            output = self._model(x.to(self.device))
            loss = self._model.loss(output, y.to(self.device))
            loss.backward()
            self.optimiser.step()

            if (i % log_batch_idx) == 0:
                logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                    i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                    loss.item()))

    def eval_model(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_eval, 1)
        self._model.eval()
        self.total_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                output = self._model(x.to(self.device))
                loss = self._model.loss(output, y.to(self.device))
                self.total_loss += loss.item()

            self.total_loss /= len(data_loader)
            
            logger.info('Epoch: {} \t Batch Loss: {:.4f}'.format(epoch,
                    self.total_loss))