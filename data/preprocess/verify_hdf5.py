"""
Utility script to verify and inspect HDF5 files created from MNE Epochs.

This script helps you:
1. Verify the HDF5 file structure
2. Check metadata and labels
3. Ensure compatibility with GraphS4mer training pipeline

Usage:
    python verify_hdf5.py --file /path/to/file.h5
    python verify_hdf5.py --dir /path/to/hdf5/directory --summary
"""

import argparse
import h5py
import numpy as np
import json
from pathlib import Path


def print_hdf5_structure(hf, indent=0):
    """Recursively print HDF5 file structure."""
    for key in hf.keys():
        item = hf[key]
        prefix = "  " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}📁 Group: {key}/")
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}📄 Dataset: {key}")
            print(f"{prefix}   Shape: {item.shape}, Dtype: {item.dtype}")


def verify_single_file(hdf5_file, verbose=True):
    """
    Verify a single HDF5 file.

    Args:
        hdf5_file: Path to HDF5 file
        verbose: Print detailed information

    Returns:
        dict: Summary information
    """
    try:
        with h5py.File(hdf5_file, 'r') as hf:
            if verbose:
                print(f"\n{'='*70}")
                print(f"File: {hdf5_file}")
                print(f"{'='*70}")

                # Print attributes
                print("\n📋 Attributes:")
                for key, value in hf.attrs.items():
                    if key == 'mne_info' or key == 'ch_names' or key == 'event_id':
                        # Parse JSON
                        try:
                            parsed = json.loads(value)
                            print(f"  {key}: {parsed}")
                        except:
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")

                # Print structure
                print("\n🗂️  File Structure:")
                print_hdf5_structure(hf)

                # Check critical datasets
                print("\n✓ Critical Datasets Check:")

                # Hypnogram (sleep stages)
                if 'hypnogram' in hf:
                    hypnogram = hf['hypnogram'][:]
                    print(f"  ✓ Hypnogram found: {len(hypnogram)} epochs")
                    unique, counts = np.unique(hypnogram, return_counts=True)
                    print(f"    Sleep stage distribution:")
                    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', -1: 'Unknown'}
                    for stage, count in zip(unique, counts):
                        stage_name = stage_names.get(int(stage), f'Stage_{stage}')
                        print(f"      {stage_name} ({stage}): {count} epochs ({count/len(hypnogram)*100:.1f}%)")
                else:
                    print("  ✗ Hypnogram NOT found")

                # ADHD label
                if 'adhd_label' in hf:
                    adhd_label = hf['adhd_label'][()]
                    print(f"  ✓ ADHD label found: {adhd_label}")
                else:
                    print("  ✗ ADHD label NOT found")

                # Signals
                if 'signals' in hf:
                    print(f"  ✓ Signals group found")
                    signals = hf['signals']
                    for modality in signals.keys():
                        print(f"    {modality}:")
                        for ch_name in signals[modality].keys():
                            ch_data = signals[modality][ch_name]
                            print(f"      {ch_name}: {ch_data.shape}")
                else:
                    print("  ✗ Signals group NOT found")

                # Check data integrity
                print("\n🔍 Data Integrity Check:")
                n_epochs = hf.attrs.get('n_epochs', -1)
                n_times = hf.attrs.get('n_times_per_epoch', -1)
                sfreq = hf.attrs.get('sfreq', -1)

                if n_epochs > 0 and n_times > 0:
                    expected_length = n_epochs * n_times
                    print(f"  Expected signal length: {expected_length} samples")
                    print(f"    ({n_epochs} epochs × {n_times} samples/epoch)")

                    # Check one channel
                    if 'signals' in hf:
                        for modality in hf['signals'].keys():
                            for ch_name in hf['signals'][modality].keys():
                                actual_length = hf['signals'][modality][ch_name].shape[0]
                                if actual_length == expected_length:
                                    print(f"  ✓ Channel {ch_name}: {actual_length} samples (CORRECT)")
                                else:
                                    print(f"  ✗ Channel {ch_name}: {actual_length} samples (MISMATCH!)")
                                break
                            break

                # Check hypnogram length matches n_epochs
                if 'hypnogram' in hf:
                    hypno_length = len(hf['hypnogram'][:])
                    if hypno_length == n_epochs:
                        print(f"  ✓ Hypnogram length matches n_epochs: {hypno_length}")
                    else:
                        print(f"  ✗ Hypnogram length ({hypno_length}) != n_epochs ({n_epochs})")

            # Collect summary
            summary = {
                'file': str(hdf5_file),
                'valid': True,
                'n_epochs': hf.attrs.get('n_epochs', -1),
                'n_channels': hf.attrs.get('n_channels', -1),
                'sfreq': hf.attrs.get('sfreq', -1),
                'subject_id': hf.attrs.get('subject_id', 'unknown'),
                'has_hypnogram': 'hypnogram' in hf,
                'has_adhd_label': 'adhd_label' in hf,
                'has_signals': 'signals' in hf,
            }

            if 'adhd_label' in hf:
                summary['adhd_label'] = int(hf['adhd_label'][()])

            if 'hypnogram' in hf:
                hypnogram = hf['hypnogram'][:]
                unique, counts = np.unique(hypnogram, return_counts=True)
                summary['sleep_stages'] = {int(s): int(c) for s, c in zip(unique, counts)}

            return summary

    except Exception as e:
        print(f"✗ Error reading {hdf5_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'file': str(hdf5_file),
            'valid': False,
            'error': str(e)
        }


def verify_directory(hdf5_dir, summary_only=False):
    """
    Verify all HDF5 files in a directory.

    Args:
        hdf5_dir: Directory containing HDF5 files
        summary_only: Only print summary statistics
    """
    hdf5_files = sorted(Path(hdf5_dir).glob('*.h5'))

    if not hdf5_files:
        print(f"No .h5 files found in {hdf5_dir}")
        return

    print(f"\nFound {len(hdf5_files)} HDF5 files")

    summaries = []
    for hdf5_file in hdf5_files:
        summary = verify_single_file(hdf5_file, verbose=not summary_only)
        summaries.append(summary)

    # Print overall summary
    print(f"\n{'='*70}")
    print("📊 OVERALL SUMMARY")
    print(f"{'='*70}")

    valid_files = [s for s in summaries if s.get('valid', False)]
    invalid_files = [s for s in summaries if not s.get('valid', False)]

    print(f"\nTotal files: {len(summaries)}")
    print(f"  Valid: {len(valid_files)}")
    print(f"  Invalid: {len(invalid_files)}")

    if valid_files:
        total_epochs = sum(s.get('n_epochs', 0) for s in valid_files)
        total_subjects = len(set(s.get('subject_id', '') for s in valid_files))

        print(f"\nTotal epochs: {total_epochs}")
        print(f"Unique subjects: {total_subjects}")

        # Aggregate sleep stage counts
        all_sleep_stages = {}
        for s in valid_files:
            if 'sleep_stages' in s:
                for stage, count in s['sleep_stages'].items():
                    all_sleep_stages[stage] = all_sleep_stages.get(stage, 0) + count

        if all_sleep_stages:
            print("\nAggregated sleep stage distribution:")
            stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', -1: 'Unknown'}
            for stage in sorted(all_sleep_stages.keys()):
                count = all_sleep_stages[stage]
                stage_name = stage_names.get(stage, f'Stage_{stage}')
                print(f"  {stage_name} ({stage}): {count} epochs ({count/total_epochs*100:.1f}%)")

        # ADHD label distribution
        adhd_labels = [s.get('adhd_label', -1) for s in valid_files]
        unique_adhd, counts_adhd = np.unique(adhd_labels, return_counts=True)
        print("\nADHD label distribution:")
        for label, count in zip(unique_adhd, counts_adhd):
            print(f"  {label}: {count} subjects ({count/len(valid_files)*100:.1f}%)")

        # Check consistency
        print("\n🔍 Consistency Check:")
        sfreqs = [s.get('sfreq', -1) for s in valid_files]
        unique_sfreqs = set(sfreqs)
        if len(unique_sfreqs) == 1:
            print(f"  ✓ All files have same sampling frequency: {list(unique_sfreqs)[0]} Hz")
        else:
            print(f"  ⚠ Multiple sampling frequencies found: {unique_sfreqs}")

        n_channels_list = [s.get('n_channels', -1) for s in valid_files]
        unique_n_channels = set(n_channels_list)
        if len(unique_n_channels) == 1:
            print(f"  ✓ All files have same number of channels: {list(unique_n_channels)[0]}")
        else:
            print(f"  ⚠ Multiple channel counts found: {unique_n_channels}")

    if invalid_files:
        print("\n❌ Invalid files:")
        for s in invalid_files:
            print(f"  {s['file']}: {s.get('error', 'Unknown error')}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify HDF5 files created from MNE Epochs'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to a single HDF5 file to verify'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing HDF5 files to verify'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show only summary (no detailed per-file output)'
    )

    args = parser.parse_args()

    if args.file:
        verify_single_file(args.file, verbose=True)
    elif args.dir:
        verify_directory(args.dir, summary_only=args.summary)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
