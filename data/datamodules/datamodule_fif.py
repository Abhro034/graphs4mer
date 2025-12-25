"""
Data module for loading FIF (FIFF) files for GraphS4mer
Compatible with MNE-Python epoch files
Supports: Sleep stage classification, ADHD classification
"""

import os
import pytorch_lightning as pl
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from typing import Optional
from tqdm import tqdm
from pathlib import Path


def process_subject_fif(fif_path):
    """Process a single subject from .fif file"""
    import mne

    stage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'R': 4}

    # Load data
    raw = mne.read_epochs(fif_path, preload=True, verbose=False)
    patient_eeg = raw.get_data()  # (n_epochs, n_channels, n_timepoints)

    # Get ADHD label
    if hasattr(raw, 'metadata') and raw.metadata is not None and 'ADHD' in raw.metadata.columns:
        label = raw.metadata['ADHD'].iloc[0]
    else:
        # Fallback: no ADHD label, use 0
        label = 0

    # Get sleep stages
    sleep_stages = raw.events[:, 2]
    if sleep_stages.dtype == object:
        sleep_stages = np.array([stage_map.get(s, 0) for s in sleep_stages])
    else:
        sleep_stages = np.array([stage_map.get(int(s), int(s)) for s in sleep_stages])

    # Normalize each epoch with robust handling
    normalized_eeg = []
    for epoch in patient_eeg:
        # Check for invalid values first
        if np.isnan(epoch).any() or np.isinf(epoch).any():
            print(f"WARNING: Found NaN/Inf in epoch data before normalization")
            # Replace NaN/Inf with median
            epoch = np.nan_to_num(epoch, nan=np.nanmedian(epoch), posinf=np.nanmedian(epoch), neginf=np.nanmedian(epoch))

        # Per-channel normalization with robust epsilon
        mean = epoch.mean(axis=1, keepdims=True)
        std = epoch.std(axis=1, keepdims=True)

        # Use larger epsilon and clip extreme values
        epsilon = 1e-6
        normalized_epoch = (epoch - mean) / (std + epsilon)

        # Clip to prevent extreme values
        normalized_epoch = np.clip(normalized_epoch, -10, 10)

        normalized_eeg.append(normalized_epoch)

    patient_eeg = np.array(normalized_eeg, dtype=np.float32)

    return patient_eeg, sleep_stages, label


def load_fif_data(fif_directory):
    """Load all .fif files from directory"""
    fif_dir = Path(fif_directory)
    fif_files = sorted(list(fif_dir.glob('*.fif')))

    if len(fif_files) == 0:
        raise ValueError(f"No .fif files found in {fif_directory}")

    print(f"Found {len(fif_files)} .fif files")

    fold_data = []
    sleep_labels = []
    adhd_labels = []
    subject_ids = []

    for fif_path in tqdm(fif_files, desc="Loading .fif files"):
        try:
            epoch_data, sleep_stages, adhd_label = process_subject_fif(fif_path)

            fold_data.append(epoch_data)
            sleep_labels.append(sleep_stages)
            adhd_labels.append(adhd_label)
            subject_ids.append(fif_path.stem)

        except Exception as e:
            print(f"Error processing {fif_path.name}: {e}")
            continue

    adhd_labels = np.array(adhd_labels)
    fold_len = np.array([len(s) for s in sleep_labels])

    print(f"Loaded {len(fold_data)} subjects successfully")
    print(f"Total epochs: {sum(fold_len)}")
    print(f"ADHD: {np.sum(adhd_labels)}, Control: {len(adhd_labels) - np.sum(adhd_labels)}")

    return fold_data, sleep_labels, adhd_labels, subject_ids, fold_len


class FIFDataset(Dataset):
    """
    PyTorch Geometric Dataset for FIF files with sleep stages and ADHD labels
    """
    def __init__(
        self,
        root,
        fif_directory,
        task='sleep_stage',  # 'sleep_stage' or 'adhd'
        split='train',
        scaler=None,
        transform=None,
        pre_transform=None,
    ):
        self.root = root
        self.fif_directory = fif_directory
        self.task = task
        self.split = split
        self.scaler = scaler

        # Load FIF data
        print(f"Loading FIF data from: {fif_directory}")
        self._load_fif_data()

        # process
        super().__init__(root, transform, pre_transform)

    def _load_fif_data(self):
        """Load FIF files using custom loader"""
        fold_data, sleep_labels, adhd_labels, subject_ids, fold_len = load_fif_data(self.fif_directory)

        # Store loaded data
        self.fold_data = fold_data  # List of (n_epochs, n_channels, n_samples) arrays
        self.sleep_labels = sleep_labels  # List of sleep stage arrays
        self.adhd_labels = adhd_labels  # Array of ADHD labels per subject
        self.subject_ids = subject_ids
        self.fold_len = fold_len

        # Create flat epoch index
        self.epoch_to_subject = []
        self.epoch_indices = []

        for subj_idx, n_epochs in enumerate(fold_len):
            for epoch_idx in range(n_epochs):
                self.epoch_to_subject.append(subj_idx)
                self.epoch_indices.append(epoch_idx)

        # Get dimensions from first subject
        first_data = fold_data[0]
        self.n_channels = first_data.shape[1]
        self.n_samples = first_data.shape[2]

        # Determine number of classes based on task
        if self.task == 'sleep_stage':
            self.n_classes = 5  # W, N1, N2, N3, REM
        elif self.task == 'adhd':
            self.n_classes = 2  # ADHD vs Control
        else:
            raise ValueError(f"Unknown task: {self.task}")

        print(f"\nDataset Statistics:")
        print(f"  Task: {self.task}")
        print(f"  Total subjects: {len(fold_data)}")
        print(f"  Total epochs: {len(self.epoch_to_subject)}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Samples per epoch: {self.n_samples}")
        print(f"  Classes: {self.n_classes}")

    @property
    def raw_file_names(self):
        return []

    def len(self):
        return len(self.epoch_to_subject)

    def get_labels(self):
        """Get labels based on task"""
        if self.task == 'sleep_stage':
            # Flatten all sleep labels
            all_labels = []
            for sleep_stages in self.sleep_labels:
                all_labels.extend(sleep_stages)
            return torch.LongTensor(all_labels)
        elif self.task == 'adhd':
            # Repeat ADHD label for each epoch
            all_labels = []
            for subj_idx, n_epochs in enumerate(self.fold_len):
                all_labels.extend([self.adhd_labels[subj_idx]] * n_epochs)
            return torch.LongTensor(all_labels)

    def get(self, idx):
        """
        Get a single epoch as PyG Data object

        Returns:
            Data: PyG Data object with:
                - x: node features (n_channels, n_samples, 1)
                - y: label (sleep stage or ADHD)
                - writeout_fn: identifier
        """
        # Get subject and epoch index
        subj_idx = self.epoch_to_subject[idx]
        epoch_idx = self.epoch_indices[idx]

        # Get epoch data: (n_channels, n_samples)
        epoch_data = self.fold_data[subj_idx][epoch_idx]

        # Reshape to (n_channels, n_samples, 1) for PyG
        # This matches the format expected by GraphS4mer: (num_nodes, seq_len, feature_dim)
        x = torch.FloatTensor(epoch_data).unsqueeze(-1)

        # Get label based on task
        if self.task == 'sleep_stage':
            label = self.sleep_labels[subj_idx][epoch_idx]
        elif self.task == 'adhd':
            label = self.adhd_labels[subj_idx]

        y = torch.LongTensor([label])

        # Create identifier
        subject_id = self.subject_ids[subj_idx]
        writeout_fn = f"{subject_id}_epoch_{epoch_idx}"

        # Create PyG Data object
        data = Data(x=x.float(), y=y, writeout_fn=writeout_fn)

        return data


class FIF_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FIF files
    Supports sleep stage classification and ADHD classification
    """
    def __init__(
        self,
        fif_directory,
        task='sleep_stage',  # 'sleep_stage' or 'adhd'
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        train_batch_size=32,
        test_batch_size=32,
        num_workers=4,
        pin_memory=True,
        standardize=False,
        balanced_sampling=False,
    ):
        super().__init__()
        self.fif_directory = fif_directory
        self.task = task
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
        Setup datasets by splitting the FIF data
        """
        # Load full dataset
        full_dataset = FIFDataset(
            root='./data/processed_fif',
            fif_directory=self.fif_directory,
            task=self.task,
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
        self.output_dim = full_dataset.n_classes

        print(f"\nDataset properties:")
        print(f"  Number of channels (nodes): {self.num_nodes}")
        print(f"  Sequence length: {self.max_seq_len}")
        print(f"  Task: {self.task}")
        print(f"  Number of classes: {self.output_dim}")

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
