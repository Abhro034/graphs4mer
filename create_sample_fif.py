"""
Create a sample FIF file for testing the GraphS4mer model
Shape: (n_epochs=variable, n_channels=10, epoch_duration=30s, n_samples=3001)
"""

import numpy as np
import os

def create_sample_fif_file(
    n_epochs=20,
    n_channels=10,
    epoch_duration=30,  # seconds
    n_samples_per_epoch=3001,
    output_dir="./data/sample_fif",
    filename="sample_data.fif"
):
    """
    Create a sample FIF file using MNE-Python

    Args:
        n_epochs: Number of epochs (variable)
        n_channels: Number of channels (10)
        epoch_duration: Duration of each epoch in seconds (30s)
        n_samples_per_epoch: Number of samples per epoch (3001)
        output_dir: Directory to save the file
        filename: Name of the output file
    """

    # Import MNE
    try:
        import mne
        print("MNE-Python is available")
    except ImportError:
        print("MNE-Python not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "mne"])
        import mne
        print("MNE-Python installed successfully")

    # Calculate sampling frequency
    # n_samples_per_epoch = sfreq * epoch_duration
    # 3001 = sfreq * 30
    sfreq = n_samples_per_epoch / epoch_duration
    print(f"Calculated sampling frequency: {sfreq} Hz")

    # Create channel names
    ch_names = [f'CH{i+1}' for i in range(n_channels)]

    # Create channel types (all EEG)
    ch_types = ['eeg'] * n_channels

    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Generate synthetic data with realistic EEG-like signals
    # Shape: (n_channels, n_samples_total)
    n_samples_total = n_epochs * n_samples_per_epoch

    # Generate data with some patterns
    data = np.zeros((n_channels, n_samples_total))

    for ch in range(n_channels):
        # Add multiple frequency components to make it more realistic
        time = np.arange(n_samples_total) / sfreq

        # Add different frequency bands
        # Delta (0.5-4 Hz)
        data[ch, :] += 5 * np.sin(2 * np.pi * 2 * time)
        # Theta (4-8 Hz)
        data[ch, :] += 3 * np.sin(2 * np.pi * 6 * time + ch * 0.5)
        # Alpha (8-13 Hz)
        data[ch, :] += 4 * np.sin(2 * np.pi * 10 * time + ch * 0.3)
        # Beta (13-30 Hz)
        data[ch, :] += 2 * np.sin(2 * np.pi * 20 * time + ch * 0.7)

        # Add some noise
        data[ch, :] += np.random.randn(n_samples_total) * 0.5

    # Convert to microvolts (typical EEG scale)
    data = data * 1e-6

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Create events for epoching
    # Create events at regular intervals (start of each epoch)
    events = np.zeros((n_epochs, 3), dtype=int)
    for i in range(n_epochs):
        events[i, 0] = i * n_samples_per_epoch  # Sample number
        events[i, 1] = 0  # Previous event (not used)
        events[i, 2] = 1 if i % 2 == 0 else 2  # Event ID (alternating between 1 and 2)

    # Create epochs
    event_id = {'condition_1': 1, 'condition_2': 2}
    tmin = 0  # Start of epoch
    tmax = epoch_duration - 1/sfreq  # End of epoch (almost 30s)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=None, preload=True, verbose=False)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save as FIF file
    output_path = os.path.join(output_dir, filename)
    epochs.save(output_path, overwrite=True)

    print(f"\nSample FIF file created successfully!")
    print(f"Location: {output_path}")
    print(f"\nFile specifications:")
    print(f"  - Number of epochs: {len(epochs)}")
    print(f"  - Number of channels: {len(epochs.ch_names)}")
    print(f"  - Epoch duration: {epoch_duration} seconds")
    print(f"  - Samples per epoch: {epochs.get_data().shape[-1]}")
    print(f"  - Sampling frequency: {epochs.info['sfreq']} Hz")
    print(f"  - Data shape: {epochs.get_data().shape} (epochs, channels, samples)")

    return output_path, epochs


if __name__ == "__main__":
    # Create sample file with specified parameters
    output_path, epochs = create_sample_fif_file(
        n_epochs=20,
        n_channels=10,
        epoch_duration=30,
        n_samples_per_epoch=3001,
        output_dir="./data/sample_fif",
        filename="sample_epochs.fif"
    )

    print("\nData preview:")
    print(epochs)
