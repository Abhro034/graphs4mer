"""
Convert MNE Epochs (.fif files) to HDF5 format for GraphS4mer training.

This script converts preprocessed sleep EEG epochs to HDF5 format compatible with
the existing GraphS4mer training pipeline, preserving:
- Sleep stage labels per epoch
- ADHD labels per subject
- Channel information
- Sampling frequency
- All MNE metadata

Usage:
    python convert_epochs_to_hdf5.py --input_dir /path/to/fif/files \
                                     --output_dir /path/to/save/hdf5 \
                                     --adhd_labels /path/to/adhd_labels.csv
"""

import os
import argparse
import numpy as np
import h5py
import mne
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json


def read_adhd_labels(adhd_label_file):
    """
    Read ADHD labels from CSV file.

    Expected CSV format:
        subject_id,adhd_label
        sub-001,1
        sub-002,0
        ...

    Args:
        adhd_label_file: Path to CSV file with subject IDs and ADHD labels

    Returns:
        dict: {subject_id: adhd_label}
    """
    if adhd_label_file is None or not os.path.exists(adhd_label_file):
        print("Warning: ADHD label file not found. Setting all subjects to -1 (unknown).")
        return {}

    df = pd.read_csv(adhd_label_file)
    adhd_dict = dict(zip(df['subject_id'], df['adhd_label']))
    print(f"Loaded ADHD labels for {len(adhd_dict)} subjects")
    return adhd_dict


def extract_subject_id_from_filename(filename):
    """
    Extract subject ID from filename.

    Modify this function based on your naming convention.
    Examples:
        'sub-001_epochs.fif' -> 'sub-001'
        'patient_ABC_sleep_epo.fif' -> 'patient_ABC'

    Args:
        filename: Filename string

    Returns:
        str: Subject ID
    """
    # Example: Remove common suffixes
    subject_id = filename.replace('_epochs.fif', '') \
                        .replace('_epo.fif', '') \
                        .replace('-epo.fif', '') \
                        .replace('.fif', '')
    return subject_id


def convert_epochs_to_hdf5(
    epochs_file,
    output_dir,
    adhd_label_dict=None,
    channel_order=None,
    sleep_stage_mapping=None
):
    """
    Convert a single MNE Epochs .fif file to HDF5 format.

    Args:
        epochs_file: Path to .fif file containing MNE Epochs
        output_dir: Directory to save HDF5 file
        adhd_label_dict: Dictionary mapping subject IDs to ADHD labels
        channel_order: List of channel names in desired order (optional)
        sleep_stage_mapping: Dict to map event IDs to sleep stage labels

    Returns:
        str: Path to saved HDF5 file, or None if failed
    """
    try:
        # Read epochs
        print(f"Reading: {epochs_file}")
        epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)

        # Extract subject ID from filename
        filename = os.path.basename(epochs_file)
        subject_id = extract_subject_id_from_filename(filename)

        # Get ADHD label for this subject
        adhd_label = adhd_label_dict.get(subject_id, -1) if adhd_label_dict else -1

        # Get data: shape (n_epochs, n_channels, n_times)
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape

        # Get channel names
        ch_names = epochs.ch_names

        # Reorder channels if specified
        if channel_order is not None:
            try:
                # Find indices for reordering
                channel_indices = [ch_names.index(ch) for ch in channel_order]
                data = data[:, channel_indices, :]
                ch_names = channel_order
                print(f"  Reordered channels to: {ch_names}")
            except ValueError as e:
                print(f"  Warning: Could not reorder channels. {e}")

        # Get sleep stage labels from events
        # MNE Epochs stores events as: (sample_idx, 0, event_id)
        events = epochs.events
        event_ids = events[:, 2]  # Extract event IDs (3rd column)

        # Map event IDs to sleep stage labels
        if sleep_stage_mapping is None:
            # Default mapping (adjust based on your event coding)
            sleep_stage_mapping = {
                0: 0,  # Wake
                1: 1,  # N1
                2: 2,  # N2
                3: 3,  # N3
                4: 4,  # REM
                5: 0,  # Movement (treat as wake)
                6: -1, # Unknown/Artifact
            }

        # Convert event IDs to sleep stage labels
        sleep_stage_labels = np.array([
            sleep_stage_mapping.get(event_id, -1)
            for event_id in event_ids
        ])

        # Alternative: If sleep stages stored in epochs.metadata
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            if 'sleep_stage' in epochs.metadata.columns:
                sleep_stage_labels = epochs.metadata['sleep_stage'].values
                print(f"  Using sleep stages from metadata")

        # Get sampling frequency
        sfreq = epochs.info['sfreq']

        # Prepare output filename
        output_filename = filename.replace('.fif', '.h5')
        output_path = os.path.join(output_dir, output_filename)

        # Save to HDF5
        with h5py.File(output_path, 'w') as hf:

            # ===== Main Data Groups =====

            # Create signals group (organized by modality like original code)
            signals_group = hf.create_group('signals')

            # Identify channel types
            eeg_channels = []
            ecg_channels = []
            eog_channels = []
            emg_channels = []
            other_channels = []

            for idx, ch_name in enumerate(ch_names):
                ch_type = mne.io.pick.channel_type(epochs.info, idx)
                if ch_type == 'eeg':
                    eeg_channels.append((idx, ch_name))
                elif ch_type == 'ecg':
                    ecg_channels.append((idx, ch_name))
                elif ch_type == 'eog':
                    eog_channels.append((idx, ch_name))
                elif ch_type in ['emg', 'misc']:
                    emg_channels.append((idx, ch_name))
                else:
                    other_channels.append((idx, ch_name))

            # Store channels by type
            if eeg_channels:
                eeg_group = signals_group.create_group('EEG')
                for idx, ch_name in eeg_channels:
                    # Flatten all epochs for this channel: (n_epochs * n_times,)
                    channel_data = data[:, idx, :].flatten()
                    eeg_group.create_dataset(ch_name, data=channel_data, compression='gzip')

            if ecg_channels:
                ecg_group = signals_group.create_group('ECG')
                for idx, ch_name in ecg_channels:
                    channel_data = data[:, idx, :].flatten()
                    ecg_group.create_dataset(ch_name, data=channel_data, compression='gzip')

            if eog_channels:
                eog_group = signals_group.create_group('EOG')
                for idx, ch_name in eog_channels:
                    channel_data = data[:, idx, :].flatten()
                    eog_group.create_dataset(ch_name, data=channel_data, compression='gzip')

            if emg_channels:
                emg_group = signals_group.create_group('EMG')
                for idx, ch_name in emg_channels:
                    channel_data = data[:, idx, :].flatten()
                    emg_group.create_dataset(ch_name, data=channel_data, compression='gzip')

            if other_channels:
                other_group = signals_group.create_group('OTHER')
                for idx, ch_name in other_channels:
                    channel_data = data[:, idx, :].flatten()
                    other_group.create_dataset(ch_name, data=channel_data, compression='gzip')

            # ===== Sleep Stage Labels (Hypnogram) =====
            hf.create_dataset('hypnogram', data=sleep_stage_labels, compression='gzip')

            # ===== ADHD Label (Subject-level) =====
            hf.create_dataset('adhd_label', data=adhd_label)

            # ===== Metadata as Attributes =====
            hf.attrs['subject_id'] = subject_id
            hf.attrs['n_epochs'] = n_epochs
            hf.attrs['n_channels'] = n_channels
            hf.attrs['n_times_per_epoch'] = n_times
            hf.attrs['sfreq'] = sfreq
            hf.attrs['ch_names'] = json.dumps(ch_names)  # Store as JSON string

            # Store epoch duration
            epoch_duration = n_times / sfreq
            hf.attrs['epoch_duration_sec'] = epoch_duration

            # Store MNE info as JSON (important metadata)
            mne_info = {
                'ch_names': ch_names,
                'ch_types': [mne.io.pick.channel_type(epochs.info, i) for i in range(n_channels)],
                'sfreq': sfreq,
                'highpass': epochs.info['highpass'],
                'lowpass': epochs.info['lowpass'],
                'description': epochs.info.get('description', ''),
            }
            hf.attrs['mne_info'] = json.dumps(mne_info)

            # Store channel positions if available
            if epochs.info['dig'] is not None:
                try:
                    montage = epochs.get_montage()
                    if montage is not None:
                        positions = montage.get_positions()
                        ch_pos = positions['ch_pos']
                        # Convert to arrays for storage
                        ch_pos_array = np.array([
                            ch_pos.get(ch, [np.nan, np.nan, np.nan])
                            for ch in ch_names
                        ])
                        hf.create_dataset('channel_positions', data=ch_pos_array)
                except:
                    print("  Warning: Could not save channel positions")

            # Store metadata if available
            if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                metadata_group = hf.create_group('metadata')
                for col in epochs.metadata.columns:
                    try:
                        metadata_group.create_dataset(
                            col,
                            data=epochs.metadata[col].values,
                            compression='gzip'
                        )
                    except:
                        # For non-numeric columns, store as strings
                        metadata_group.create_dataset(
                            col,
                            data=epochs.metadata[col].astype(str).values.astype('S'),
                            compression='gzip'
                        )

            # Store event information
            hf.create_dataset('events', data=events, compression='gzip')
            hf.attrs['event_id'] = json.dumps(epochs.event_id)

        print(f"  ✓ Saved: {output_path}")
        print(f"    - {n_epochs} epochs × {n_channels} channels × {n_times} samples")
        print(f"    - Sleep stages: {np.unique(sleep_stage_labels, return_counts=True)}")
        print(f"    - ADHD label: {adhd_label}")

        return output_path

    except Exception as e:
        print(f"  ✗ Failed to convert {epochs_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_file_markers(hdf5_dir, output_csv, dataset_name='custom'):
    """
    Create file markers CSV compatible with GraphS4mer training pipeline.

    This creates a CSV index mapping each epoch to its file and clip index,
    similar to the DOD-H file markers used in the codebase.

    Args:
        hdf5_dir: Directory containing HDF5 files
        output_csv: Path to save file markers CSV
        dataset_name: Name of the dataset (for identification)
    """
    records = []

    hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.h5')])

    print(f"\nCreating file markers from {len(hdf5_files)} HDF5 files...")

    for h5_file in tqdm(hdf5_files):
        h5_path = os.path.join(hdf5_dir, h5_file)

        try:
            with h5py.File(h5_path, 'r') as hf:
                n_epochs = hf.attrs['n_epochs']
                sleep_stages = hf['hypnogram'][:]
                adhd_label = hf['adhd_label'][()]
                subject_id = hf.attrs['subject_id']

                # Create one row per epoch
                for clip_idx in range(n_epochs):
                    records.append({
                        'record_id': h5_file,
                        'subject_id': subject_id,
                        'clip_index': clip_idx,
                        'label': sleep_stages[clip_idx],  # Sleep stage label
                        'adhd_label': adhd_label,  # ADHD label (subject-level)
                        'dataset': dataset_name
                    })
        except Exception as e:
            print(f"Warning: Could not read {h5_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Filter out invalid sleep stages if desired
    # df = df[df['label'] >= 0]  # Remove unknown/artifact epochs

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"✓ File markers saved to: {output_csv}")
    print(f"  Total epochs: {len(df)}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")
    print(f"\nSleep stage distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"\nADHD label distribution:")
    print(df['adhd_label'].value_counts().sort_index())

    return df


def split_file_markers(df, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split file markers into train/val/test sets by subject.

    Args:
        df: DataFrame with file markers
        output_dir: Directory to save split CSVs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)

    # Get unique subjects
    subjects = df['subject_id'].unique()
    n_subjects = len(subjects)

    # Shuffle subjects
    np.random.shuffle(subjects)

    # Split subjects
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]

    # Create splits
    train_df = df[df['subject_id'].isin(train_subjects)]
    val_df = df[df['subject_id'].isin(val_subjects)]
    test_df = df[df['subject_id'].isin(test_subjects)]

    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_file_markers.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_file_markers.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_file_markers.csv'), index=False)

    print(f"\n✓ Split file markers saved to: {output_dir}")
    print(f"  Train: {len(train_df)} epochs from {len(train_subjects)} subjects")
    print(f"  Val:   {len(val_df)} epochs from {len(val_subjects)} subjects")
    print(f"  Test:  {len(test_df)} epochs from {len(test_subjects)} subjects")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MNE Epochs (.fif) to HDF5 format for GraphS4mer training'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing .fif epoch files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save HDF5 files'
    )
    parser.add_argument(
        '--adhd_labels',
        type=str,
        default=None,
        help='CSV file with subject_id and adhd_label columns'
    )
    parser.add_argument(
        '--channel_order',
        type=str,
        nargs='+',
        default=None,
        help='Desired channel order (space-separated channel names)'
    )
    parser.add_argument(
        '--create_markers',
        action='store_true',
        help='Create file markers CSV after conversion'
    )
    parser.add_argument(
        '--markers_output',
        type=str,
        default='file_markers.csv',
        help='Output path for file markers CSV'
    )
    parser.add_argument(
        '--split_data',
        action='store_true',
        help='Split file markers into train/val/test'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='custom',
        help='Name of the dataset'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ADHD labels
    adhd_label_dict = read_adhd_labels(args.adhd_labels)

    # Find all .fif files
    fif_files = sorted(Path(args.input_dir).rglob('*.fif'))
    print(f"\nFound {len(fif_files)} .fif files")

    # Convert each file
    successful = 0
    failed = 0

    for fif_file in tqdm(fif_files, desc="Converting files"):
        result = convert_epochs_to_hdf5(
            epochs_file=str(fif_file),
            output_dir=args.output_dir,
            adhd_label_dict=adhd_label_dict,
            channel_order=args.channel_order
        )

        if result is not None:
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")

    # Create file markers if requested
    if args.create_markers:
        df = create_file_markers(
            hdf5_dir=args.output_dir,
            output_csv=args.markers_output,
            dataset_name=args.dataset_name
        )

        # Split if requested
        if args.split_data:
            markers_dir = os.path.dirname(args.markers_output) or '.'
            split_file_markers(df, markers_dir)


if __name__ == '__main__':
    main()
