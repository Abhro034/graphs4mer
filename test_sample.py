#!/usr/bin/env python3
"""
Test script to run GraphS4mer on synthetic sample data.
This script creates minimal synthetic data to test if the model runs without errors.
"""

import numpy as np
import os
import h5py
import pandas as pd
import torch
import pickle
from pathlib import Path

def create_sample_data():
    """Create synthetic sample data for testing"""

    # Create directories
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)

    raw_data_dir = sample_dir / "raw"
    raw_data_dir.mkdir(exist_ok=True)

    preproc_dir = sample_dir / "processed"
    preproc_dir.mkdir(exist_ok=True)

    save_dir = sample_dir / "results"
    save_dir.mkdir(exist_ok=True)

    # Parameters for TUH-like data
    num_nodes = 19  # EEG channels
    freq = 200  # Hz
    seq_len = 60  # seconds
    num_samples_train = 10
    num_samples_val = 3
    num_samples_test = 3

    # Create synthetic h5 files
    def create_h5_files(num_files, prefix):
        file_list = []
        for i in range(num_files):
            filename = f"{prefix}_{i:03d}.h5"
            filepath = raw_data_dir / filename

            # Create synthetic EEG signal (num_nodes, time_points)
            # Multiple 60s clips per file
            num_clips = 2
            total_time_points = freq * seq_len * num_clips
            signal = np.random.randn(num_nodes, total_time_points).astype(np.float32)

            with h5py.File(filepath, 'w') as hf:
                hf.create_dataset('resampled_signal', data=signal)

            # Create entries for each clip
            for clip_idx in range(num_clips):
                is_seizure = np.random.randint(0, 2)  # Random binary label
                file_list.append({
                    'file_name': filename,
                    'clip_index': clip_idx,
                    'is_seizure': is_seizure
                })

        return file_list

    # Create train, val, test files
    print("Creating synthetic training data...")
    train_files = create_h5_files(num_samples_train, 'train')

    print("Creating synthetic validation data...")
    val_files = create_h5_files(num_samples_val, 'val')

    print("Creating synthetic test data...")
    test_files = create_h5_files(num_samples_test, 'test')

    # Create file markers CSV - use the expected directory name for TUH
    file_markers_dir = Path("data/file_markers_sample")
    file_markers_dir.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(train_files).to_csv(file_markers_dir / 'train_file_markers_60s.csv', index=False)
    pd.DataFrame(val_files).to_csv(file_markers_dir / 'val_file_markers_60s.csv', index=False)
    pd.DataFrame(test_files).to_csv(file_markers_dir / 'test_file_markers_60s.csv', index=False)

    print(f"Created {len(train_files)} training samples")
    print(f"Created {len(val_files)} validation samples")
    print(f"Created {len(test_files)} test samples")

    return str(raw_data_dir.absolute()), str(preproc_dir.absolute()), str(save_dir.absolute())


def run_test():
    """Run the model on sample data"""
    import sys
    from args import get_args
    from train import main

    # Create sample data
    raw_data_dir, preproc_dir, save_dir = create_sample_data()

    # Set up arguments for training
    sys.argv = [
        'test_sample.py',
        '--dataset', 'tuh',
        '--raw_data_dir', raw_data_dir,
        '--preproc_dir', preproc_dir,
        '--max_seq_len', '60',
        '--num_nodes', '19',
        '--input_dim', '1',
        '--output_dim', '1',
        '--train_batch_size', '2',
        '--test_batch_size', '2',
        '--num_workers', '0',
        '--adj_mat_dir', 'data/eeg_electrode_graph/adj_mx_3d.pkl',
        '--model_name', 'graphs4mer',
        '--graph_learn_metric', 'self_attention',
        '--dropout', '0.1',
        '--g_conv', 'gine',
        '--num_gcn_layers', '1',
        '--hidden_dim', '32',  # Reduced for faster testing
        '--num_temporal_layers', '2',  # Reduced for faster testing
        '--state_dim', '16',  # Reduced for faster testing
        '--bidirectional', 'False',
        '--temporal_model', 's4',
        '--temporal_pool', 'mean',
        '--resolution', '2000',
        '--graph_pool', 'max',
        '--activation_fn', 'leaky_relu',
        '--prune_method', 'thresh_abs',
        '--thresh', '0.1',
        '--use_prior', 'False',
        '--knn', '2',
        '--residual_weight', '0.6',
        '--regularizations', 'feature_smoothing', 'degree', 'sparse',
        '--feature_smoothing_weight', '0.05',
        '--degree_weight', '0.05',
        '--sparse_weight', '0.05',
        '--save_dir', save_dir,
        '--metric_name', 'auroc',
        '--eval_metrics', 'auroc', 'F1', 'precision', 'recall',
        '--metric_avg', 'binary',
        '--lr_init', '8e-4',
        '--l2_wd', '5e-3',
        '--num_epochs', '2',  # Just 2 epochs for testing
        '--scheduler', 'timm_cosine',
        '--t_initial', '2',
        '--warmup_t', '1',
        '--optimizer', 'adamw',
        '--do_train', 'True',
        '--balanced_sampling', 'False',  # Disable for faster testing
        '--accumulate_grad_batches', '1',
        '--gpus', '0',  # CPU only for testing
        '--rand_seed', '42',
        '--file_marker_dir', 'data/file_markers_sample',
    ]

    print("\n" + "="*80)
    print("Running GraphS4mer on synthetic sample data...")
    print("="*80 + "\n")

    # Run main training
    args = get_args()
    main(args)

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_test()
