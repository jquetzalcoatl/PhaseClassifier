
"""
Module for loading and managing datasets for phase classification tasks.
Classes:
    myDataset(Dataset):
        Custom PyTorch Dataset for handling snapshots and labels.
        Args:
            dataset (tuple): Tuple containing snapshot and label tensors.
        Methods:
            __len__(): Returns the number of samples in the dataset.
            __getitem__(index): Returns the snapshot and label at the specified index as float tensors.
    DataManager:
        Handles loading, splitting, and batching of datasets.
        Args:
            cfg (object, optional): Configuration object containing dataset parameters.
        Methods:
            load_dataset(): Loads dataset from HDF5 file specified in the configuration.
            select_dataset(): Selects and loads the dataset based on configuration.
            create_dataloaders(): Splits the dataset into train/val/test sets and creates corresponding DataLoaders.
Attributes:
    logger: Logger instance for logging dataset operations.
"""
import torch, h5py, numpy as np
from torch.utils.data import DataLoader, Dataset

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

class myDataset(Dataset):
    def __init__(self, dataset):
        self.snapshot, self.label, self.temperature = dataset[0], dataset[1], dataset[2]

    def __len__(self):
        return len(self.snapshot)

    def __getitem__(self, index):
        return self.snapshot[index, :].float(), self.label[index, :], self.temperature[index, :]


class DataManager():
    def __init__(self, cfg=None):
        self._config = cfg
        self.select_dataset()          # for different datasets
        self.create_dataloaders()      # slice into train/val/test

    def load_dataset(self):
        with h5py.File(self._config.data.data_path, 'r') as file:
            self.f = {}
            logger.info("Keys: %s" % list(file.keys()))
            for key in file.keys():
                self.f[key] = torch.tensor(np.array(file[key]))

        logger.info(f'{self.f.keys()}')

    def select_dataset(self):
        logger.info(f"Loading dataset: {self._config.data.dataset_name}")
        self.load_dataset()

    def create_dataloaders(self):
        total = self.f["Snapshot"].shape[0]
        logger.info(f'Creating dataloader. Total number of samples: {total}')
        frac_train = self._config.data.frac_train_dataset
        frac_val = self._config.data.frac_val_dataset

        tr = int(np.floor(total * frac_train/2))
        va = int(np.floor(total * frac_val/2))

        tr_idx = list(range(tr)) + list(range(int(np.floor(total/2)), int(np.floor(total/2))+tr))
        va_idx = list(range(tr, tr + va)) + list(range(int(np.floor(total/2))+tr, int(np.floor(total/2))+tr + va))
        te_idx = list(range(tr + va, int(np.floor(total/2)))) + list(range(int(np.floor(total/2))+tr + va, total))

        assert list(set(range(total)) - set(tr_idx) - set(va_idx) - set(te_idx)) == []

        snapshot = self.f["Snapshot"]
        label = self.f["Label"]
        temperature = self.f["Temperature"]

        self.train_loader = DataLoader(
            myDataset((snapshot[tr_idx, :], label[tr_idx, :], temperature[tr_idx, :])),
            batch_size=self._config.data.batch_size_tr,
            shuffle=True,
            num_workers=self._config.data.num_workers
        )

        self.val_loader = DataLoader(
            myDataset((snapshot[va_idx, :], label[va_idx, :], temperature[va_idx, :])),
            batch_size=self._config.data.batch_size_val,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        self.test_loader = DataLoader(
            myDataset((snapshot[te_idx, :], label[te_idx, :], temperature[te_idx, :])),
            batch_size=self._config.data.batch_size_test,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )
        
        logger.info("{0}: {2} events, {1} batches".format(self.train_loader, len(self.train_loader), len(self.train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.test_loader, len(self.test_loader), len(self.test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.val_loader, len(self.val_loader), len(self.val_loader.dataset)))