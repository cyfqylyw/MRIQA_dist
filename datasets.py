import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.ndimage import zoom


class MRIDataset(Dataset):
    def __init__(self, root_dir, ds_name, index_list=None, transform=None):
        self.root_dir = root_dir
        self.ds_name = ds_name
        self.index_list = index_list
        self.transform = transform
        self.filepaths = []
        self.labels = []

        for category_name in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, category_name)
            for filename in os.listdir(category_dir):
                filepath = os.path.join(category_dir, filename)
                self.filepaths.append(filepath)
                self.labels.append(category_name)
        
        if self.index_list is not None:
            self.filepaths = [self.filepaths[idx] for idx in self.index_list]
            self.labels = [self.labels[idx] for idx in self.index_list]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        
        mri_data = np.load(filepath)
        if self.transform:
            mri_data = self.transform(mri_data, self.ds_name)
        
        return mri_data, label


class FourierTransform(object):
    def __call__(self, mri_data, ds_name):
        # Reorder the dimensions
        if ds_name == "NFBS_T1w":
            mri_data = np.transpose(mri_data, (0, 2, 1))  # From (193, 229, 193) to (193, 193, 229)
        elif ds_name in ["QTAB_T1w", "ARC_T1w", "ARC_T2w"]:
            mri_data = np.transpose(mri_data, (1, 2, 0))  # From (176, 300, 320) to (300, 320, 176)

        # Downsample to (128, 128, height)
        zoom_factors = (128 / mri_data.shape[0], 128 / mri_data.shape[1], 1)
        mri_data_resampled = zoom(mri_data, zoom_factors, order=1)
        
        # Perform Fourier Transform on MRI data
        mri_data_fft = np.fft.fftn(mri_data_resampled, axes=(0,1,2))
        return mri_data_resampled, mri_data_fft


def get_data_loaders(root_dir, ds_name, batch_size, num_workers, transform=None, k_fold_splits=10, device=None):
    dataset = MRIDataset(root_dir, ds_name, transform=transform)
    kf = KFold(n_splits=k_fold_splits, shuffle=True)
    data_loaders = []

    for train_index, test_index in kf.split(dataset):
        train_dataset = MRIDataset(root_dir, ds_name, index_list=train_index, transform=transform)
        test_dataset = MRIDataset(root_dir, ds_name, index_list=test_index, transform=transform)

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


# Example usage:
seed_everything()
ds_name="NFBS_T1w"
root_dir = 'datasets/dist/NFBS_T1w/'
batch_size = 32
num_workers = 4
transform = FourierTransform()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_loaders = get_data_loaders(root_dir, ds_name, batch_size, num_workers, transform=transform, device=device)

# Iterate through 10 fold data loaders
for fold_idx, fold_data in enumerate(data_loaders):
    print(f"Fold {fold_idx + 1}:")
    train_loader = fold_data['train']
    test_loader = fold_data['test']
    
    # Output sample counts and class counts for train_loader
    print("Train samples:", len(train_loader.dataset))
    
    # Output sample counts and class counts for val_loader
    print("Test samples:", len(test_loader.dataset))

    for data, labels in train_loader:
        print(data[0].shape, data[1].shape, labels)
        break  # Only process the first batch for debugging purposes

    for data, labels in test_loader:
        print(data[0].shape, data[1].shape, labels)
        break  # Only process the first batch for debugging purposes

    print("\n")
    