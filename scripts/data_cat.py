

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

# PATH_RAW = '/raid/javier/Projects/PhaseClassifier/Dataset/raw'
# PATH_PROCESSED = '/raid/javier/Projects/PhaseClassifier/Dataset/data.hdf5'
# PATH_RAW2 = '/raid/javier/Projects/PhaseClassifier/Dataset/raw2'
# PATH_PROCESSED2 = '/raid/javier/Projects/PhaseClassifier/Dataset/data_2.hdf5'

PATH_RAW = '/raid/javier/Projects/PhaseClassifier/Dataset/raw128/Ising_128_grid'
PATH_PROCESSED = '/raid/javier/Projects/PhaseClassifier/Dataset/data128.hdf5'
PATH_RAW2 = '/raid/javier/Projects/PhaseClassifier/Dataset/raw128/Ising_128_grid'
PATH_PROCESSED2 = '/raid/javier/Projects/PhaseClassifier/Dataset/data128.hdf5'

def main(key):
    if key == 'Train':
        path_read = PATH_RAW
        path_processed = PATH_PROCESSED
    else:
        path_read = PATH_RAW2
        path_processed = PATH_PROCESSED2

    filenames = os.listdir(path_read)
    data = {}
    for file in filenames:
        hdf5_path = f'{path_read}/{file}'
        data[file] = {'Temperature': [], 'Snapshot': [], 'Susceptibility': [], 'Magnetization': []}
        with h5py.File(hdf5_path, 'r') as f:
            logger.info(f"Processing {file}")
            for key1 in f.keys():
                for key in f[key1].keys():
                    # print(f[key1][key])
                    if key == 'Temperature':
                        tmp = f[key1][key]
                        data[file]['Temperature'].append(tmp[()])
                    elif key == 'Snapshot':
                        tmp = f[key1][key]
                        data[file]['Snapshot'].extend(torch.tensor(tmp[()]).unsqueeze(0).unsqueeze(1).unsqueeze(2))  # add channel dim
                    elif key == 'Susceptibility':
                        tmp = f[key1][key]
                        data[file]['Susceptibility'].extend(torch.tensor(tmp[()]).unsqueeze(0).unsqueeze(1).unsqueeze(2))  # add channel dim
                    elif key == 'Magnetization':
                        tmp = f[key1][key]
                        data[file]['Magnetization'].extend(torch.tensor(tmp[()]).unsqueeze(0).unsqueeze(1).unsqueeze(2))  # add channel dim
        for key in data[file].keys():
            if key == 'Snapshot':
                data[file][key] = torch.cat(data[file][key],dim=0)
            else:
                data[file][key] = torch.tensor(data[file][key])

    d = {}
    for feature in ['Temperature', 'Snapshot', 'Susceptibility', 'Magnetization']:
        if feature == 'Snapshot':
            d['Snapshot'] = torch.cat([data[key]['Snapshot'] for key in filenames],dim=0).to(dtype=torch.float64)
        else:
            d[feature] = torch.cat([data[key][feature].unsqueeze(1) for key in filenames],dim=0).to(dtype=torch.float64)
        logger.info(d[feature].shape)
    d['Label'] = (d['Temperature'] < 2.269).to(dtype=torch.int64)

    logger.info("Saving processed data.")
    idx = torch.sort(d['Label'][:,0]).indices
    with h5py.File(path_processed, 'w') as f:
        for feature in d.keys():
            dset = f.create_dataset(feature, data=d[feature][idx,:] if feature != 'Snapshot' else d[feature][idx,:,:])

if __name__ == "__main__":
    logger.info("Starting data processing.")
    main('test')
    logger.info("Data processing completed.")
    #To plot samples
    # plt.imshow(data['Snapshot'][0,:,:], cmap='gray')
    # torch.nonzero(data['Temperature'] < 2.2)