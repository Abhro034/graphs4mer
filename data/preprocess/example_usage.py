"""
Example usage of the FIF to HDF5 conversion pipeline.

This script demonstrates how to:
1. Convert MNE Epochs to HDF5
2. Read and verify the converted files
3. Create file markers for training

You can modify this script for your specific use case.
"""

import mne
import numpy as np
import h5py
import pandas as pd
from convert_epochs_to_hdf5 import (
    convert_epochs_to_hdf5,
    create_file_markers,
    split_file_markers
)
from verify_hdf5 import verify_single_file


# =============================================================================
# Example 1: Convert a single FIF file to HDF5
# =============================================================================

def example_single_conversion():
    """Convert a single MNE Epochs file to HDF5."""
    print("\n" + "="*70)
    print("Example 1: Converting a Single FIF File")
    print("="*70)

    # Path to your epochs file
    epochs_file = "path/to/your/epochs.fif"
    output_dir = "path/to/output/hdf5"

    # ADHD labels (optional)
    adhd_labels = {
        'sub-001': 1,  # ADHD
        'sub-002': 0,  # Control
        'sub-003': 1,  # ADHD
    }

    # Convert
    result = convert_epochs_to_hdf5(
        epochs_file=epochs_file,
        output_dir=output_dir,
        adhd_label_dict=adhd_labels,
        channel_order=None,  # Or specify: ['C3', 'C4', 'F3', 'F4', ...]
        sleep_stage_mapping=None  # Or specify custom mapping
    )

    if result:
        print(f"\n✓ Conversion successful: {result}")

        # Verify the converted file
        print("\nVerifying converted file...")
        verify_single_file(result, verbose=True)


# =============================================================================
# Example 2: Read HDF5 file and extract data
# =============================================================================

def example_read_hdf5(hdf5_file):
    """Demonstrate how to read the converted HDF5 file."""
    print("\n" + "="*70)
    print("Example 2: Reading HDF5 File")
    print("="*70)

    with h5py.File(hdf5_file, 'r') as hf:
        # Read metadata
        subject_id = hf.attrs['subject_id']
        n_epochs = hf.attrs['n_epochs']
        n_channels = hf.attrs['n_channels']
        sfreq = hf.attrs['sfreq']

        print(f"\nMetadata:")
        print(f"  Subject: {subject_id}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Channels: {n_channels}")
        print(f"  Sampling Frequency: {sfreq} Hz")

        # Read sleep stage labels
        hypnogram = hf['hypnogram'][:]
        print(f"\nSleep Stages: {hypnogram[:10]}...")  # First 10

        # Read ADHD label
        adhd_label = hf['adhd_label'][()]
        print(f"ADHD Label: {adhd_label}")

        # Read signal data for one channel
        # Signals are flattened: (n_epochs * n_times,)
        channel_name = 'C3_M2'  # Example channel
        if 'signals' in hf and 'EEG' in hf['signals']:
            if channel_name in hf['signals']['EEG']:
                signal = hf['signals']['EEG'][channel_name][:]
                print(f"\nSignal shape for {channel_name}: {signal.shape}")

                # Extract specific epoch (e.g., epoch 5)
                epoch_idx = 5
                n_times_per_epoch = hf.attrs['n_times_per_epoch']
                start_idx = epoch_idx * n_times_per_epoch
                end_idx = start_idx + n_times_per_epoch

                epoch_data = signal[start_idx:end_idx]
                print(f"Epoch {epoch_idx} data shape: {epoch_data.shape}")
                print(f"Epoch {epoch_idx} sleep stage: {hypnogram[epoch_idx]}")


# =============================================================================
# Example 3: Create custom sleep stage mapping
# =============================================================================

def example_custom_sleep_mapping():
    """Demonstrate custom sleep stage mapping."""
    print("\n" + "="*70)
    print("Example 3: Custom Sleep Stage Mapping")
    print("="*70)

    # If your MNE events use different IDs
    custom_mapping = {
        10: 0,  # Wake
        11: 1,  # N1
        12: 2,  # N2
        13: 3,  # N3
        14: 4,  # REM
        99: -1, # Artifact/Unknown
    }

    print("Custom mapping:")
    for event_id, sleep_stage in custom_mapping.items():
        print(f"  Event ID {event_id} → Sleep Stage {sleep_stage}")

    # Use in conversion
    epochs_file = "path/to/epochs.fif"
    output_dir = "path/to/output"

    result = convert_epochs_to_hdf5(
        epochs_file=epochs_file,
        output_dir=output_dir,
        adhd_label_dict=None,
        sleep_stage_mapping=custom_mapping  # Use custom mapping
    )


# =============================================================================
# Example 4: Batch conversion with progress tracking
# =============================================================================

def example_batch_conversion(input_dir, output_dir, adhd_label_csv):
    """Convert multiple FIF files with progress tracking."""
    print("\n" + "="*70)
    print("Example 4: Batch Conversion")
    print("="*70)

    from pathlib import Path
    from tqdm import tqdm

    # Load ADHD labels
    adhd_df = pd.read_csv(adhd_label_csv)
    adhd_labels = dict(zip(adhd_df['subject_id'], adhd_df['adhd_label']))

    # Find all FIF files
    fif_files = list(Path(input_dir).rglob('*.fif'))
    print(f"Found {len(fif_files)} FIF files")

    # Convert each file
    successful = []
    failed = []

    for fif_file in tqdm(fif_files, desc="Converting"):
        try:
            result = convert_epochs_to_hdf5(
                epochs_file=str(fif_file),
                output_dir=output_dir,
                adhd_label_dict=adhd_labels,
            )
            if result:
                successful.append(result)
            else:
                failed.append(fif_file)
        except Exception as e:
            print(f"Error converting {fif_file}: {e}")
            failed.append(fif_file)

    print(f"\nConversion complete:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    return successful, failed


# =============================================================================
# Example 5: Creating file markers from converted HDF5 files
# =============================================================================

def example_create_markers(hdf5_dir):
    """Create file markers from HDF5 files."""
    print("\n" + "="*70)
    print("Example 5: Creating File Markers")
    print("="*70)

    # Create file markers
    markers_file = "my_file_markers.csv"
    df = create_file_markers(
        hdf5_dir=hdf5_dir,
        output_csv=markers_file,
        dataset_name='my_sleep_study'
    )

    print(f"\nFile markers preview:")
    print(df.head(10))

    # Split into train/val/test
    markers_dir = "."
    split_file_markers(
        df=df,
        output_dir=markers_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print("\nSplit files created:")
    print("  - train_file_markers.csv")
    print("  - val_file_markers.csv")
    print("  - test_file_markers.csv")


# =============================================================================
# Example 6: Reading data for training (simulates DataLoader)
# =============================================================================

def example_read_for_training(hdf5_file, clip_index):
    """Simulate how the training DataLoader reads data."""
    print("\n" + "="*70)
    print("Example 6: Reading Data Like Training Pipeline")
    print("="*70)

    with h5py.File(hdf5_file, 'r') as hf:
        # Get parameters
        n_times = hf.attrs['n_times_per_epoch']
        sfreq = hf.attrs['sfreq']
        seq_len_sec = hf.attrs['epoch_duration_sec']

        print(f"Epoch duration: {seq_len_sec} seconds")
        print(f"Samples per epoch: {n_times}")
        print(f"Sampling frequency: {sfreq} Hz")

        # Read sleep stage label for this clip
        label = hf['hypnogram'][clip_index]
        print(f"\nClip {clip_index} sleep stage: {label}")

        # Read all channel data for this specific epoch
        start_idx = clip_index * n_times
        end_idx = start_idx + n_times

        # Collect all channels
        channels_data = []
        channel_names = []

        for modality in hf['signals'].keys():
            for ch_name in hf['signals'][modality].keys():
                # Extract this epoch's data
                epoch_data = hf['signals'][modality][ch_name][start_idx:end_idx]
                channels_data.append(epoch_data)
                channel_names.append(ch_name)

        # Stack into (n_channels, n_times, 1) format
        x = np.stack(channels_data, axis=0)  # (n_channels, n_times)
        x = np.expand_dims(x, axis=-1)  # (n_channels, n_times, 1)

        print(f"\nLoaded data shape: {x.shape}")
        print(f"Channels: {channel_names}")

        return x, label


# =============================================================================
# Example 7: Working with epochs metadata
# =============================================================================

def example_with_metadata():
    """Create epochs with metadata and convert to HDF5."""
    print("\n" + "="*70)
    print("Example 7: Epochs with Metadata")
    print("="*70)

    # This example shows how to create epochs with metadata
    # The metadata will be preserved in the HDF5 file

    # Simulated example (you would use your actual data)
    """
    # In your preprocessing:
    import mne
    import pandas as pd

    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'sleep_stage': [0, 1, 2, 2, 3, 3, 4, 0, 1, 2],
        'epoch_quality': [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        'arousal': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    })

    # Create epochs with metadata
    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=30,
        baseline=None,
        metadata=metadata
    )

    # Save
    epochs.save('epochs_with_metadata-epo.fif', overwrite=True)

    # Convert to HDF5
    # The metadata will be stored in the 'metadata' group
    """

    print("""
    When you convert epochs with metadata, the HDF5 file will contain:

    file.h5
    ├── signals/
    ├── hypnogram
    ├── metadata/              ← Stored here
    │   ├── sleep_stage
    │   ├── epoch_quality
    │   └── arousal
    └── ...

    You can access metadata when reading:

    with h5py.File('file.h5', 'r') as hf:
        epoch_quality = hf['metadata']['epoch_quality'][:]
        arousal = hf['metadata']['arousal'][:]
    """)


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FIF to HDF5 Conversion Examples")
    print("="*70)

    print("""
    This script demonstrates various ways to use the conversion pipeline.

    To run these examples, modify the file paths and uncomment the examples
    you want to test.
    """)

    # Uncomment to run examples:

    # example_single_conversion()

    # example_read_hdf5("path/to/file.h5")

    # example_custom_sleep_mapping()

    # example_batch_conversion(
    #     input_dir="path/to/fif/files",
    #     output_dir="path/to/hdf5/output",
    #     adhd_label_csv="adhd_labels.csv"
    # )

    # example_create_markers("path/to/hdf5/files")

    # example_read_for_training("path/to/file.h5", clip_index=5)

    # example_with_metadata()

    print("\n" + "="*70)
    print("See convert_epochs_to_hdf5.py for the full implementation")
    print("See verify_hdf5.py for verification utilities")
    print("See README_FIF_TO_HDF5.md for complete documentation")
    print("="*70 + "\n")
