import os
import numpy as np
from utils import *
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from scipy.ndimage import zoom


class MRIDataset(Dataset):
    def __init__(self, root_dir, ds_name, dist_type, index_list=None, sample_length=10):
        self.root_dir = root_dir
        self.ds_name = ds_name
        self.index_list = index_list
        self.sample_length = sample_length
        self.filepaths = []
        self.labels = []

        if dist_type == 'all':
            for category_name in os.listdir(root_dir):
                category_dir = os.path.join(root_dir, category_name)
                for filename in os.listdir(category_dir):
                    filepath = os.path.join(category_dir, filename)
                    self.filepaths.append(filepath)
                    self.labels.append(all_labels.index(category_name))
        else:
            dist_labels = [dist_type + '_' + str(x) for x in range(1, 5)]
            for category_name in dist_labels:
                category_dir = os.path.join(root_dir, category_name)
                for filename in os.listdir(category_dir):
                    filepath = os.path.join(category_dir, filename)
                    self.filepaths.append(filepath)
                    self.labels.append(dist_labels.index(category_name))

            
        src_dir = f'datasets/src/{ds_name}/'
        for filename in os.listdir(src_dir):
            filepath = os.path.join(src_dir, filename)
            self.filepaths.append(filepath)
            if dist_type == 'all':
                self.labels.append(len(all_labels))
            else:
                self.labels.append(len(dist_labels))
        
        if self.index_list is not None:
            self.filepaths = [self.filepaths[idx] for idx in self.index_list]
            self.labels = [self.labels[idx] for idx in self.index_list]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        
        mri_data = np.load(filepath)

        if self.ds_name == "NFBS_T1w":
            mri_data = np.transpose(mri_data, (1, 0, 2))    # From (193, 229, 193) to (229, 193, 193)
        elif self.ds_name in ['QTAB_T2w', 'IXI_T1w', 'IXI_T2w', 'OASIS_T1w']:
            mri_data = np.transpose(mri_data, (2, 0, 1))    # From (256, 256, height) to (height, 256, 256)
        
        # Downsample to (img_shape, img_shape, height)
        zoom_factors = (1, img_shape / mri_data.shape[1], img_shape / mri_data.shape[2])
        mri_data_resampled = zoom(mri_data, zoom_factors, order=1)

        # resample the middle part of self.sample_length
        if self.sample_length < height[self.ds_name]:
            median_h = int(mri_data_resampled.shape[0] / 2 - self.sample_length / 2)
            mri_data_resampled = mri_data_resampled[median_h: median_h+self.sample_length, :, :]
        
        # perform DFT
        mri_data_fft = np.fft.fftn(mri_data_resampled, axes=(0,1,2))

        # aggregate the data
        data = (mri_data_resampled, mri_data_fft)

        return data, label


def get_data_loaders(root_dir, ds_name, dist_type, batch_size, num_workers, sample_length, k_fold_splits=5):
    dataset = MRIDataset(root_dir, ds_name, dist_type)
    kf = KFold(n_splits=k_fold_splits, shuffle=True)
    data_loaders = []

    for train_index, test_index in kf.split(dataset):
        train_dataset = MRIDataset(root_dir, ds_name, dist_type, index_list=train_index, sample_length=sample_length)
        test_dataset = MRIDataset(root_dir, ds_name, dist_type, index_list=test_index, sample_length=sample_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        data_loaders.append({'train': train_loader, 'test': test_loader})

    return data_loaders


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

