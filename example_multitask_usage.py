"""
Example usage script for multi-task Sleep Stage + ADHD detection

This script demonstrates how to:
1. Prepare data
2. Initialize the model
3. Train the model
4. Make predictions
5. Evaluate results
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mne

# Import our custom modules
from model.multitask_graphs4mer import MultiTaskGraphS4mer
from data.multitask_dataset import MultiTaskPSGDataset, multitask_collate_fn


def create_sample_data(save_dir='./sample_data', n_patients=5, n_epochs_per_patient=100):
    """
    Create sample .fif files for demonstration purposes.
    In practice, you would load real PSG data instead.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    sampling_rate = 100  # Hz
    epoch_length = 30    # seconds
    n_channels = 19
    n_times = sampling_rate * epoch_length  # 3000 timepoints

    # Channel names (standard 10-20 system)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)

    print(f"Creating {n_patients} sample patient files...")

    for patient_id in range(n_patients):
        # Generate random EEG-like data
        data = np.random.randn(n_epochs_per_patient, n_channels, n_times).astype(np.float32)

        # Add some structure to simulate real EEG
        for epoch_idx in range(n_epochs_per_patient):
            for ch in range(n_channels):
                # Add some low-frequency components
                t = np.linspace(0, epoch_length, n_times)
                data[epoch_idx, ch] += 10 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz
                data[epoch_idx, ch] += 5 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz (alpha)

        # Generate sleep stage labels (random distribution)
        sleep_stages = np.random.choice([0, 1, 2, 3, 4], size=n_epochs_per_patient,
                                       p=[0.2, 0.1, 0.3, 0.2, 0.2])  # Wake, N1, N2, N3, REM

        # Generate ADHD label (binary, same for all epochs of a patient)
        adhd_label = np.random.choice([0, 1])  # 0: Non-ADHD, 1: ADHD

        # Create events array for sleep stages
        events = np.zeros((n_epochs_per_patient, 3), dtype=int)
        events[:, 0] = np.arange(n_epochs_per_patient) * n_times  # Sample indices
        events[:, 2] = sleep_stages  # Event IDs (sleep stages)

        # Create MNE Epochs object
        epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)

        # Add metadata with ADHD label
        import pandas as pd
        metadata = pd.DataFrame({'ADHD': [adhd_label] * n_epochs_per_patient})
        epochs.metadata = metadata

        # Save to file
        filename = os.path.join(save_dir, f'patient_{patient_id:03d}_psg.fif')
        epochs.save(filename, overwrite=True, verbose=False)

        print(f"  Created {filename} - ADHD: {adhd_label}, {n_epochs_per_patient} epochs")

    print(f"\nSample data created in {save_dir}")
    return save_dir


def example_training(data_dir, save_dir='./example_experiment'):
    """
    Example of how to train the multi-task model.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Training Multi-Task Model")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    sampling_rate = 100
    epoch_length = 30
    n_channels = 19
    max_seq_len = sampling_rate * epoch_length  # 3000

    # Calculate resolution (must divide max_seq_len evenly)
    resolution = max_seq_len // 10  # Use 10 dynamic graphs
    while max_seq_len % resolution != 0:
        resolution -= 1

    print(f"\nModel configuration:")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Resolution: {resolution}")
    print(f"  Num dynamic graphs: {max_seq_len // resolution}")

    # Load data
    from data.multitask_dataset import load_patient_data_multitask

    train_dataset, val_dataset, test_dataset = load_patient_data_multitask(
        patients_dir=data_dir,
        fs_target=sampling_rate,
        n_channels=n_channels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        return_pyg=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=multitask_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=multitask_collate_fn
    )

    # Create model
    print("\nInitializing model...")
    model = MultiTaskGraphS4mer(
        input_dim=1,
        num_nodes=n_channels,
        dropout=0.3,
        num_temporal_layers=4,
        g_conv='gine',
        num_gnn_layers=2,
        hidden_dim=128,
        max_seq_len=max_seq_len,
        resolution=resolution,
        num_sleep_classes=5,
        num_adhd_classes=2,
        state_dim=64,
        temporal_model='s4',
        temporal_pool='mean',
        graph_pool='mean',
        edge_top_perc=0.2,
        K=3,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-3)

    # Training loop (simplified - just 3 epochs for demo)
    print("\nTraining (simplified demo - 3 epochs)...")
    num_epochs = 3

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            sleep_logits, adhd_logits, reg_losses = model(batch, task='both')

            # Compute losses
            sleep_loss = F.cross_entropy(sleep_logits, batch.y_sleep.long())
            adhd_loss = F.cross_entropy(adhd_logits, batch.y_adhd.long())
            loss = sleep_loss + adhd_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        sleep_correct = 0
        adhd_correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                sleep_logits, adhd_logits, _ = model(batch, task='both')

                sleep_loss = F.cross_entropy(sleep_logits, batch.y_sleep.long())
                adhd_loss = F.cross_entropy(adhd_logits, batch.y_adhd.long())
                loss = sleep_loss + adhd_loss

                val_loss += loss.item()

                # Calculate accuracy
                sleep_preds = torch.argmax(sleep_logits, dim=1)
                adhd_preds = torch.argmax(adhd_logits, dim=1)

                sleep_correct += (sleep_preds == batch.y_sleep).sum().item()
                adhd_correct += (adhd_preds == batch.y_adhd).sum().item()
                total += batch.y_sleep.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Sleep Acc: {sleep_correct/total:.4f}")
        print(f"  ADHD Acc: {adhd_correct/total:.4f}")

    # Save model
    checkpoint_path = os.path.join(save_dir, 'example_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': {
            'sampling_rate': sampling_rate,
            'epoch_length': epoch_length,
            'num_nodes': n_channels,
            'input_dim': 1,
            'hidden_dim': 128,
            'num_temporal_layers': 4,
            'num_gnn_layers': 2,
            'state_dim': 64,
            'temporal_model': 's4',
            'g_conv': 'gine',
            'temporal_pool': 'mean',
            'graph_pool': 'mean',
            'edge_top_perc': 0.2,
            'knn': 3,
            'dropout': 0.3,
        }
    }, checkpoint_path)

    print(f"\nModel saved to {checkpoint_path}")
    return checkpoint_path


def example_inference(checkpoint_path, data_dir):
    """
    Example of how to make predictions with the trained model.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Making Predictions")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"\nLoading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    # Rebuild model
    max_seq_len = args['sampling_rate'] * args['epoch_length']
    resolution = max_seq_len // 10
    while max_seq_len % resolution != 0:
        resolution -= 1

    model = MultiTaskGraphS4mer(
        input_dim=args['input_dim'],
        num_nodes=args['num_nodes'],
        dropout=args['dropout'],
        num_temporal_layers=args['num_temporal_layers'],
        g_conv=args['g_conv'],
        num_gnn_layers=args['num_gnn_layers'],
        hidden_dim=args['hidden_dim'],
        max_seq_len=max_seq_len,
        resolution=resolution,
        num_sleep_classes=5,
        num_adhd_classes=2,
        state_dim=args['state_dim'],
        temporal_model=args['temporal_model'],
        temporal_pool=args['temporal_pool'],
        graph_pool=args['graph_pool'],
        edge_top_perc=args['edge_top_perc'],
        K=args['knn'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully!")

    # Load test data
    import glob
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*_psg.fif")))[:2]  # Just 2 files for demo

    dataset = MultiTaskPSGDataset(
        file_paths=file_paths,
        sampling_rate=args['sampling_rate'],
        n_channels=args['num_nodes'],
        return_pyg=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=multitask_collate_fn
    )

    # Make predictions
    print("\nMaking predictions...")
    sleep_label_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    adhd_label_names = ['Non-ADHD', 'ADHD']

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            sleep_logits, adhd_logits, _ = model(batch, task='both')

            sleep_probs = F.softmax(sleep_logits, dim=1)
            adhd_probs = F.softmax(adhd_logits, dim=1)

            sleep_preds = torch.argmax(sleep_logits, dim=1)
            adhd_preds = torch.argmax(adhd_logits, dim=1)

            print(f"\nBatch {i+1}:")
            print(f"  Sample predictions:")
            for j in range(min(3, sleep_preds.size(0))):  # Show first 3 samples
                print(f"    Epoch {j+1}:")
                print(f"      Sleep Stage: {sleep_label_names[sleep_preds[j]]} "
                      f"(prob: {sleep_probs[j, sleep_preds[j]]:.3f})")
                print(f"      ADHD: {adhd_label_names[adhd_preds[j]]} "
                      f"(prob: {adhd_probs[j, adhd_preds[j]]:.3f})")

            break  # Just show one batch for demo

    print("\nInference complete!")


def main():
    """
    Run the complete example pipeline.
    """
    print("="*60)
    print("Multi-Task Sleep Stage + ADHD Detection Example")
    print("="*60)

    # Step 1: Create sample data
    print("\nStep 1: Creating sample data...")
    data_dir = create_sample_data(save_dir='./sample_data', n_patients=5, n_epochs_per_patient=50)

    # Step 2: Train model
    print("\nStep 2: Training model...")
    checkpoint_path = example_training(data_dir, save_dir='./example_experiment')

    # Step 3: Make predictions
    print("\nStep 3: Making predictions...")
    example_inference(checkpoint_path, data_dir)

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Replace sample data with your real PSG data")
    print("2. Adjust hyperparameters in train_multitask.py")
    print("3. Train for more epochs (100+)")
    print("4. Evaluate on your test set")
    print("5. Use inference_multitask.py for production predictions")


if __name__ == "__main__":
    main()
