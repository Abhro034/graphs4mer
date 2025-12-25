"""
Data module for loading FIF (FIFF) files for GraphS4mer
Compatible with MNE-Python epoch files
"""

import os
import pytorch_lightning as pl
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from typing import Optional
from tqdm import tqdm


class FIFDataset(Dataset):
    """
    PyTorch Geometric Dataset for FIF files
    """
    def __init__(
        self,
        root,
        fif_file_path,
        split='train',
        scaler=None,
        transform=None,
        pre_transform=None,
    ):
        self.root = root
        self.fif_file_path = fif_file_path
        self.split = split
        self.scaler = scaler

        # Load FIF file
        print(f"Loading FIF file from: {fif_file_path}")
        self._load_fif_data()

        # process
        super().__init__(root, transform, pre_transform)

    def _load_fif_data(self):
        """Load FIF file using MNE"""
        import mne

        # Load epochs
        epochs = mne.read_epochs(self.fif_file_path, preload=True, verbose=False)

        # Extract data: (n_epochs, n_channels, n_samples)
        self.data = epochs.get_data()
        self.n_epochs, self.n_channels, self.n_samples = self.data.shape

        # Get sampling frequency
        self.sfreq = epochs.info['sfreq']

        # Extract labels (event IDs)
        self.labels = epochs.events[:, 2]  # Event codes

        # Convert labels to binary (0 or 1) for classification
        unique_labels = np.unique(self.labels)
        if len(unique_labels) > 2:
            print(f"Warning: Found {len(unique_labels)} unique labels. Converting to binary.")
        self.labels = (self.labels == unique_labels[0]).astype(int)

        print(f"Loaded {self.n_epochs} epochs with {self.n_channels} channels and {self.n_samples} samples each")
        print(f"Sampling frequency: {self.sfreq} Hz")
        print(f"Labels: {np.unique(self.labels, return_counts=True)}")

    @property
    def raw_file_names(self):
        return [self.fif_file_path]

    def len(self):
        return self.n_epochs

    def get_labels(self):
        return torch.FloatTensor(self.labels)

    def get(self, idx):
        """
        Get a single epoch as PyG Data object

        Returns:
            Data: PyG Data object with:
                - x: node features (n_channels, n_samples, 1)
                - y: label
                - writeout_fn: identifier
        """
        # Get epoch data: (n_channels, n_samples)
        epoch_data = self.data[idx]

        # Reshape to (n_channels, n_samples, 1) for PyG
        # This matches the format expected by GraphS4mer: (num_nodes, seq_len, feature_dim)
        x = torch.FloatTensor(epoch_data).unsqueeze(-1)

        # Get label
        y = torch.LongTensor([self.labels[idx]])

        # Create identifier
        writeout_fn = f"epoch_{idx}"

        # Create PyG Data object
        data = Data(x=x.float(), y=y, writeout_fn=writeout_fn)

        return data


class FIF_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FIF files
    """
    def __init__(
        self,
        fif_file_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        train_batch_size=32,
        test_batch_size=32,
        num_workers=4,
        pin_memory=True,
        standardize=True,
        balanced_sampling=False,
    ):
        super().__init__()
        self.fif_file_path = fif_file_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.standardize = standardize
        self.balanced_sampling = balanced_sampling

        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets by splitting the FIF file
        """
        # Load full dataset
        full_dataset = FIFDataset(
            root='./data/processed_fif',
            fif_file_path=self.fif_file_path,
            split='full'
        )

        # Split dataset
        n_total = len(full_dataset)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val

        print(f"\nDataset split:")
        print(f"  Train: {n_train} epochs")
        print(f"  Val: {n_val} epochs")
        print(f"  Test: {n_test} epochs")

        # Use PyTorch's random_split
        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        # Store dataset properties for model initialization
        self.num_nodes = full_dataset.n_channels
        self.max_seq_len = full_dataset.n_samples
        self.sampling_freq = full_dataset.sfreq

        print(f"\nDataset properties:")
        print(f"  Number of channels (nodes): {self.num_nodes}")
        print(f"  Sequence length: {self.max_seq_len}")
        print(f"  Sampling frequency: {self.sampling_freq} Hz")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
