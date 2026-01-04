# Converting MNE Epochs (.fif) to HDF5 for GraphS4mer Training

This guide explains how to convert your preprocessed MNE Epochs files to HDF5 format for training with GraphS4mer.

## Overview

The conversion pipeline:
```
.fif Epochs → HDF5 Files → File Markers CSV → Train/Val/Test Splits
```

## Prerequisites

```bash
pip install mne h5py pandas numpy tqdm
```

## Step 1: Prepare ADHD Labels CSV

Create a CSV file with subject-level ADHD labels:

```csv
subject_id,adhd_label
sub-001,1
sub-002,0
sub-003,1
```

**Format:**
- `subject_id`: Identifier matching your .fif filenames
- `adhd_label`: 0 (control) or 1 (ADHD), or other encoding

**Example:** `example_adhd_labels.csv`

## Step 2: Convert FIF to HDF5

### Basic Usage

```bash
python convert_epochs_to_hdf5.py \
    --input_dir /path/to/fif/files \
    --output_dir /path/to/save/hdf5 \
    --adhd_labels adhd_labels.csv
```

### With File Markers Creation

```bash
python convert_epochs_to_hdf5.py \
    --input_dir /path/to/fif/files \
    --output_dir /path/to/save/hdf5 \
    --adhd_labels adhd_labels.csv \
    --create_markers \
    --markers_output file_markers.csv \
    --dataset_name my_sleep_dataset
```

### With Train/Val/Test Split

```bash
python convert_epochs_to_hdf5.py \
    --input_dir /path/to/fif/files \
    --output_dir /path/to/save/hdf5 \
    --adhd_labels adhd_labels.csv \
    --create_markers \
    --markers_output file_markers.csv \
    --split_data \
    --dataset_name my_sleep_dataset
```

### Specify Channel Order

If you need specific channel ordering (to match DODH format):

```bash
python convert_epochs_to_hdf5.py \
    --input_dir /path/to/fif/files \
    --output_dir /path/to/save/hdf5 \
    --adhd_labels adhd_labels.csv \
    --channel_order C3_M2 F3_F4 F3_M2 F3_O1 F4_M1 F4_O2 FP1_F3 FP1_M2 FP1_O1 FP2_F4 FP2_M1 FP2_O2 ECG EMG EOG1 EOG2
```

## Step 3: Verify Converted Files

### Verify Single File

```bash
python verify_hdf5.py --file /path/to/file.h5
```

**Output:**
```
======================================================================
File: /path/to/file.h5
======================================================================

📋 Attributes:
  subject_id: sub-001
  n_epochs: 800
  n_channels: 16
  n_times_per_epoch: 7500
  sfreq: 250.0
  ...

🗂️  File Structure:
📁 Group: signals/
  📁 Group: EEG/
    📄 Dataset: C3_M2
       Shape: (6000000,), Dtype: float64
    ...

✓ Critical Datasets Check:
  ✓ Hypnogram found: 800 epochs
    Sleep stage distribution:
      Wake (0): 120 epochs (15.0%)
      N1 (1): 80 epochs (10.0%)
      N2 (2): 320 epochs (40.0%)
      N3 (3): 160 epochs (20.0%)
      REM (4): 120 epochs (15.0%)
  ✓ ADHD label found: 1
  ✓ Signals group found
```

### Verify All Files in Directory

```bash
# Detailed output for each file
python verify_hdf5.py --dir /path/to/hdf5/directory

# Summary only
python verify_hdf5.py --dir /path/to/hdf5/directory --summary
```

## HDF5 File Structure

Each converted HDF5 file contains:

```
file.h5
├── signals/                          # Signal data organized by modality
│   ├── EEG/
│   │   ├── C3_M2                    # (n_epochs * n_times,) flattened array
│   │   ├── F3_F4
│   │   └── ...
│   ├── ECG/
│   │   └── ECG
│   ├── EOG/
│   │   ├── EOG1
│   │   └── EOG2
│   └── EMG/
│       └── EMG
├── hypnogram                         # (n_epochs,) sleep stage labels
├── adhd_label                        # Scalar ADHD label
├── events                            # (n_epochs, 3) MNE events array
├── channel_positions                 # (n_channels, 3) electrode positions
└── metadata/                         # Optional epoch-level metadata
    ├── column1
    └── column2

Attributes:
  - subject_id: Subject identifier
  - n_epochs: Number of epochs
  - n_channels: Number of channels
  - n_times_per_epoch: Samples per epoch
  - sfreq: Sampling frequency (Hz)
  - epoch_duration_sec: Epoch duration in seconds
  - ch_names: Channel names (JSON)
  - mne_info: MNE info dict (JSON)
  - event_id: Event ID mapping (JSON)
```

## File Markers CSV Format

The generated file markers CSV looks like:

```csv
record_id,subject_id,clip_index,label,adhd_label,dataset
sub-001_epochs.h5,sub-001,0,2,1,my_sleep_dataset
sub-001_epochs.h5,sub-001,1,2,1,my_sleep_dataset
sub-001_epochs.h5,sub-001,2,3,1,my_sleep_dataset
...
```

**Columns:**
- `record_id`: HDF5 filename
- `subject_id`: Subject identifier
- `clip_index`: Epoch index within file (0-based)
- `label`: Sleep stage label for this epoch (0-4)
- `adhd_label`: ADHD label for this subject
- `dataset`: Dataset name

## Sleep Stage Label Mapping

Default mapping (configurable in script):

```python
sleep_stage_mapping = {
    0: 0,  # Wake
    1: 1,  # N1
    2: 2,  # N2
    3: 3,  # N3
    4: 4,  # REM
    5: 0,  # Movement (treat as wake)
    6: -1, # Unknown/Artifact
}
```

## Customizing the Conversion

### Extracting Subject ID

The script uses this function to extract subject IDs from filenames:

```python
def extract_subject_id_from_filename(filename):
    # Modify based on your naming convention
    subject_id = filename.replace('_epochs.fif', '') \
                        .replace('_epo.fif', '') \
                        .replace('.fif', '')
    return subject_id
```

**Edit this function** in `convert_epochs_to_hdf5.py` to match your filename format.

### Sleep Stage Encoding

If your sleep stages are stored differently:

**Option 1: Stored in epochs.events (default)**
```python
# Events array contains event IDs
events = epochs.events
event_ids = events[:, 2]
# Mapped using sleep_stage_mapping dict
```

**Option 2: Stored in epochs.metadata**
```python
# If you have epochs.metadata['sleep_stage']
if 'sleep_stage' in epochs.metadata.columns:
    sleep_stage_labels = epochs.metadata['sleep_stage'].values
```

**Edit the conversion script** if your encoding differs.

### Channel Type Detection

Channels are automatically grouped by type:
- `eeg` → signals/EEG/
- `ecg` → signals/ECG/
- `eog` → signals/EOG/
- `emg` → signals/EMG/
- `misc` → signals/EMG/

MNE detects types from channel names. Ensure your channels are named correctly:
```python
# In your preprocessing:
raw.set_channel_types({'ECG': 'ecg', 'EOG1': 'eog', 'EOG2': 'eog'})
```

## Integration with GraphS4mer Training

After conversion, use the HDF5 files with the existing training pipeline:

### 1. Update constants.py

```python
# Add your channel configuration
MY_DATASET_CHANNELS = [
    "C3_M2", "F3_F4", "F3_M2", "F3_O1",
    "F4_M1", "F4_O2", "FP1_F3", "FP1_M2",
    "FP1_O1", "FP2_F4", "FP2_M1", "FP2_O2",
    "ECG", "EMG", "EOG1", "EOG2"
]
```

### 2. Modify datamodule_dreem.py

Update the file marker directory:

```python
MY_DATASET_FILEMARKER_DIR = "data/file_markers_my_dataset"
```

### 3. Train the Model

```bash
python train.py \
    --dataset dodh \
    --raw_data_dir /path/to/hdf5/files \
    --model_name graphs4mer \
    --train_batch_size 50 \
    --balanced_sampling \
    --num_epochs 100
```

## Troubleshooting

### Issue: "Channel positions not saved"

**Solution:** Set montage before saving epochs:
```python
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)
```

### Issue: "Sleep stages all -1 (unknown)"

**Solution:** Check your event encoding:
```python
# Print event IDs in your epochs
print(epochs.event_id)
# Update sleep_stage_mapping in convert_epochs_to_hdf5.py
```

### Issue: "Subject ID extraction incorrect"

**Solution:** Modify `extract_subject_id_from_filename()`:
```python
# Example for pattern: "patient_001_night1_epochs.fif"
def extract_subject_id_from_filename(filename):
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}"  # Returns "patient_001"
```

### Issue: "Different epoch durations"

**Solution:** Ensure all epochs have same duration:
```python
# In preprocessing
epochs = mne.Epochs(raw, events, tmin=0, tmax=30, baseline=None)
```

## Example Workflow

```bash
# 1. Convert FIF to HDF5 with all options
python convert_epochs_to_hdf5.py \
    --input_dir ./raw_fif_files \
    --output_dir ./hdf5_converted \
    --adhd_labels adhd_labels.csv \
    --create_markers \
    --markers_output ./file_markers_my_dataset/file_markers.csv \
    --split_data \
    --dataset_name my_adhd_sleep

# 2. Verify conversion
python verify_hdf5.py --dir ./hdf5_converted --summary

# 3. Check file markers
head ./file_markers_my_dataset/train_file_markers.csv
head ./file_markers_my_dataset/val_file_markers.csv
head ./file_markers_my_dataset/test_file_markers.csv

# 4. Train model
python train.py \
    --dataset dodh \
    --raw_data_dir ./hdf5_converted \
    --train_batch_size 50 \
    --test_batch_size 128 \
    --balanced_sampling \
    --num_epochs 100 \
    --model_name graphs4mer
```

## Notes

- **Data Compression**: HDF5 files use gzip compression to save disk space
- **Memory Efficiency**: Only the required 30-second clip is loaded during training
- **Parallel Loading**: Multi-worker DataLoader can read different files simultaneously
- **Batch Speed**: HDF5 format provides 5-10× faster batch loading vs. direct FIF reading

## Contact

For issues or questions:
- Check the verification output: `python verify_hdf5.py --file your_file.h5`
- Review the conversion logs for error messages
- Ensure MNE epochs are properly formatted with sleep stage annotations
