"""
Multi-task dataset for Sleep Stage Classification and ADHD Detection
Handles both epoch-level (sleep stage) and patient-level (ADHD) labels
"""
import os
import glob
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data


class MultiTaskPSGDataset(Dataset):
    """
    Multi-task dataset for:
    1. Sleep stage classification (epoch-level, 5 classes)
    2. ADHD detection (patient-level, binary)

    Each sample contains:
    - EEG epoch data: (n_channels, n_timepoints)
    - Sleep stage label: {0: Wake, 1: N1, 2: N2, 3: N3, 4: REM}
    - ADHD label: {0: non-ADHD, 1: ADHD}
    """

    def __init__(self,
                 file_paths: List[str],
                 sampling_rate: int = 100,
                 epoch_length: int = 30,
                 n_channels: Optional[int] = None,
                 picks: Optional[List[str]] = None,
                 return_pyg: bool = True):
        """
        Args:
            file_paths: List of paths to .fif files (pre-epoched data)
            sampling_rate: Sampling rate of the data (Hz)
            epoch_length: Length of each epoch in seconds
            n_channels: Number of channels to use (if None, use all available)
            picks: List of channel names to use (if None, use all)
            return_pyg: Whether to return PyTorch Geometric Data objects
        """
        self.file_paths = file_paths
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.n_channels = n_channels
        self.picks = picks
        self.return_pyg = return_pyg

        # Load all data
        self.data = []
        self.sleep_labels = []
        self.adhd_labels = []
        self.patient_ids = []
        self.file_names = []

        print("Loading multi-task PSG data...")
        for patient_idx, fpath in enumerate(tqdm(file_paths)):
            try:
                # Load pre-epoched data
                epochs = mne.read_epochs(fpath, preload=True, verbose=False)

                # Get ADHD label from metadata
                if epochs.metadata is None or 'ADHD' not in epochs.metadata:
                    print(f"Warning: {fpath} missing ADHD metadata, skipping...")
                    continue

                adhd_label = int(epochs.metadata['ADHD'].iloc[0])
                adhd_label = 0 if adhd_label == 0 else 1  # Ensure binary

                # Get channel names and select channels
                ch_names = epochs.ch_names
                if picks is not None:
                    use_channels = [ch for ch in picks if ch in ch_names]
                else:
                    use_channels = ch_names

                if n_channels is not None:
                    use_channels = use_channels[:n_channels]

                if len(use_channels) == 0:
                    print(f"Warning: {fpath} has no matching channels, skipping...")
                    continue

                # Get data: shape (n_epochs, n_channels, n_timepoints)
                data = epochs.get_data(picks=use_channels)

                # Get sleep stage labels from events
                if hasattr(epochs, 'events') and epochs.events is not None and len(epochs.events) > 0:
                    sleep_stage_labels = epochs.events[:, 2]
                else:
                    print(f"Warning: {fpath} has no sleep stage labels, skipping...")
                    continue

                n_epochs = data.shape[0]

                # Validate label count matches epoch count
                if len(sleep_stage_labels) != n_epochs:
                    print(f"Warning: {fpath} has {n_epochs} epochs but {len(sleep_stage_labels)} labels. Using minimum.")
                    n_epochs = min(n_epochs, len(sleep_stage_labels))
                    data = data[:n_epochs]
                    sleep_stage_labels = sleep_stage_labels[:n_epochs]

                # Process each epoch
                for i in range(n_epochs):
                    epoch_data = data[i]  # Shape: (n_channels, n_timepoints)

                    # Ensure correct shape
                    if epoch_data.shape[1] != (epoch_length * sampling_rate):
                        # Resample or pad/crop to correct length
                        target_len = epoch_length * sampling_rate
                        if epoch_data.shape[1] < target_len:
                            # Pad
                            pad_width = target_len - epoch_data.shape[1]
                            epoch_data = np.pad(epoch_data, ((0, 0), (0, pad_width)), mode='constant')
                        else:
                            # Crop
                            epoch_data = epoch_data[:, :target_len]

                    # Normalize each epoch (z-score per channel)
                    mean = epoch_data.mean(axis=1, keepdims=True)
                    std = epoch_data.std(axis=1, keepdims=True) + 1e-8
                    epoch_data = (epoch_data - mean) / std

                    self.data.append(epoch_data)

                    # Handle sleep stage label conversion
                    current_label = sleep_stage_labels[i]
                    if isinstance(current_label, str):
                        # Parse string labels to integer
                        label_map = {'Wake': 0, 'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N4': 3, 'REM': 4, 'R': 4}
                        sleep_label = label_map.get(current_label, 0)
                    else:
                        # Assume it's already an integer
                        sleep_label = int(current_label)
                        # Map to standard 5-class system if needed
                        if sleep_label > 4:
                            sleep_label = sleep_label % 5

                    self.sleep_labels.append(sleep_label)
                    self.adhd_labels.append(adhd_label)
                    self.patient_ids.append(patient_idx)
                    self.file_names.append(Path(fpath).stem)

                print(f"  Patient {patient_idx + 1}: Loaded {n_epochs} epochs from {Path(fpath).name} (ADHD={adhd_label})")

            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue

        self.data = np.array(self.data, dtype=np.float32)
        self.sleep_labels = np.array(self.sleep_labels, dtype=np.int64)
        self.adhd_labels = np.array(self.adhd_labels, dtype=np.int64)
        self.patient_ids = np.array(self.patient_ids, dtype=np.int32)

        print(f"\nTotal: Loaded {len(self.data)} epochs from {len(file_paths)} patients")
        print(f"Data shape: {self.data.shape}")

        # Print label distributions
        unique_sleep, counts_sleep = np.unique(self.sleep_labels, return_counts=True)
        print("\nSleep Stage distribution:")
        label_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        for label, count in zip(unique_sleep, counts_sleep):
            if label < len(label_names):
                print(f"  {label_names[label]}: {count} epochs ({count/len(self.sleep_labels)*100:.2f}%)")

        unique_adhd, counts_adhd = np.unique(self.adhd_labels, return_counts=True)
        print("\nADHD distribution:")
        for label, count in zip(unique_adhd, counts_adhd):
            print(f"  {'Non-ADHD' if label == 0 else 'ADHD'}: {count} epochs ({count/len(self.adhd_labels)*100:.2f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            If return_pyg:
                PyTorch Geometric Data object with:
                - x: (n_channels, seq_len, 1) node features
                - y_sleep: sleep stage label
                - y_adhd: ADHD label
                - patient_id: patient identifier
            Else:
                (data, sleep_label, adhd_label)
        """
        epoch_data = self.data[idx]  # (n_channels, n_timepoints)
        sleep_label = self.sleep_labels[idx]
        adhd_label = self.adhd_labels[idx]
        patient_id = self.patient_ids[idx]

        if self.return_pyg:
            # Transpose to (seq_len, n_channels) then add feature dim -> (n_channels, seq_len, 1)
            n_channels, seq_len = epoch_data.shape

            # For PyG format: each channel is a node with time series features
            x = torch.FloatTensor(epoch_data).unsqueeze(-1)  # (n_channels, seq_len, 1)

            # Create Data object
            data = Data(
                x=x,
                y_sleep=torch.LongTensor([sleep_label]),
                y_adhd=torch.LongTensor([adhd_label]),
                patient_id=torch.LongTensor([patient_id]),
                writeout_fn=self.file_names[idx]
            )

            return data
        else:
            return (
                torch.FloatTensor(epoch_data),
                torch.LongTensor([sleep_label])[0],
                torch.LongTensor([adhd_label])[0]
            )


def load_patient_data_multitask(
    patients_dir: str,
    fs_target: int = 100,
    n_channels: int = 19,
    picks: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    return_pyg: bool = True
) -> Tuple[MultiTaskPSGDataset, MultiTaskPSGDataset, MultiTaskPSGDataset]:
    """
    Load and split patient data for multi-task learning.

    Args:
        patients_dir: Directory containing .fif files
        fs_target: Target sampling frequency
        n_channels: Number of channels to use
        picks: List of channel names to pick
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        return_pyg: Whether to return PyTorch Geometric datasets

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Get all patient files
    file_paths = sorted(glob.glob(os.path.join(patients_dir, "*_psg.fif")))

    if len(file_paths) == 0:
        raise ValueError(f"No .fif files found in {patients_dir}")

    print(f"Found {len(file_paths)} patient files")

    # Split files into train/val/test
    n_total = len(file_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Shuffle files
    np.random.seed(42)
    indices = np.random.permutation(n_total)

    train_files = [file_paths[i] for i in indices[:n_train]]
    val_files = [file_paths[i] for i in indices[n_train:n_train+n_val]]
    test_files = [file_paths[i] for i in indices[n_train+n_val:]]

    print(f"\nSplit: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Create datasets
    train_dataset = MultiTaskPSGDataset(
        train_files,
        sampling_rate=fs_target,
        n_channels=n_channels,
        picks=picks,
        return_pyg=return_pyg
    )
    val_dataset = MultiTaskPSGDataset(
        val_files,
        sampling_rate=fs_target,
        n_channels=n_channels,
        picks=picks,
        return_pyg=return_pyg
    )
    test_dataset = MultiTaskPSGDataset(
        test_files,
        sampling_rate=fs_target,
        n_channels=n_channels,
        picks=picks,
        return_pyg=return_pyg
    )

    return train_dataset, val_dataset, test_dataset


def multitask_collate_fn(batch):
    """
    Custom collate function for PyTorch Geometric Data objects in multi-task setting.

    Args:
        batch: List of Data objects

    Returns:
        Batched Data object with properly stacked labels
    """
    from torch_geometric.data import Batch

    # Extract labels before batching
    sleep_labels = torch.stack([item.y_sleep for item in batch])
    adhd_labels = torch.stack([item.y_adhd for item in batch])
    patient_ids = torch.stack([item.patient_id for item in batch])
    file_names = [item.writeout_fn for item in batch]

    # Remove labels from individual items to avoid batching issues
    for item in batch:
        del item.y_sleep
        del item.y_adhd
        del item.patient_id
        del item.writeout_fn

    # Batch the data
    batched_data = Batch.from_data_list(batch)

    # Add batched labels
    batched_data.y_sleep = sleep_labels.squeeze(-1)
    batched_data.y_adhd = adhd_labels.squeeze(-1)
    batched_data.patient_id = patient_ids.squeeze(-1)
    batched_data.writeout_fn = file_names

    return batched_data
