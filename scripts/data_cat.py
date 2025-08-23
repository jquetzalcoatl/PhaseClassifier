

"""
This script processes raw HDF5 dataset files containing temperature and snapshot data for phase classification.
It reads all files from a specified directory, extracts temperature and snapshot arrays, concatenates them,
and generates labels based on a temperature threshold (2.269). The processed data is then saved into a new HDF5 file.

Functions:
    main():
        - Reads all HDF5 files from the raw dataset directory.
        - Extracts 'Temperature' and 'Snapshot' data from each file.
        - Concatenates data across all files.
        - Generates binary labels indicating phase based on temperature.
        - Saves the processed temperature, snapshot, and label tensors into a new HDF5 file.

Usage:
    Run the script directly to process and save the dataset.
"""

import h5py
import numpy as np
import torch
import os

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

def main():
    filenames = os.listdir('/raid/javier/Projects/PhaseClassifier/Dataset/raw')
    data = {}
    for file in filenames:
        hdf5_path = f'/raid/javier/Projects/PhaseClassifier/Dataset/raw/{file}'
        data[file] = {'Temperature': [], 'Snapshot': []}
        with h5py.File(hdf5_path, 'r') as f:
            logger.info(f"Processing {file}")
            for key1 in f.keys():
                for key in f[key1].keys():
                    # print(f[key1][key])
                    if key == 'Temperature':
                        tmp = f[key1][key]
                        data[file]['Temperature'].append(tmp[()])
                    if key == 'Snapshot':
                        tmp = f[key1][key]
                        data[file]['Snapshot'].extend(torch.tensor(tmp[()]).unsqueeze(0).unsqueeze(1).unsqueeze(2))  # add channel dim
        data[file]['Temperature'] = torch.tensor(data[file]['Temperature'])
        data[file]['Snapshot'] = torch.cat(data[file]['Snapshot'],dim=0)

    d = {}
    d['Temperature'] = torch.cat([data[key]['Temperature'].unsqueeze(1) for key in filenames],dim=0).to(dtype=torch.float64)
    d['Snapshot'] = torch.cat([data[key]['Snapshot'] for key in filenames],dim=0).to(dtype=torch.float64)
    d['Label'] = (d['Temperature'] > 2.269).to(dtype=torch.int64)

    logger.info("Saving processed data.")
    idx = torch.sort(d['Label'][:,0]).indices
    with h5py.File('/raid/javier/Projects/PhaseClassifier/Dataset/data.hdf5', 'w') as f:
        # Create a dataset within the file
        # 'my_dataset' is the name of the dataset within the HDF5 file
        dset = f.create_dataset('Temperature', data=d['Temperature'][idx,:])
        dset = f.create_dataset('Snapshot', data=d['Snapshot'][idx,:,:])
        dset = f.create_dataset('Label', data=d['Label'][idx,:])

if __name__ == "__main__":
    logger.info("Starting data processing.")
    main()
    logger.info("Data processing completed.")
    #To plot samples
    # plt.imshow(data['Snapshot'][0,:,:], cmap='gray')
    # torch.nonzero(data['Temperature'] < 2.2)