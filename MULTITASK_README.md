# Multi-Task Learning: Sleep Stage Classification + ADHD Detection

This directory contains code for multi-task learning with GraphS4mer, designed to simultaneously perform:
1. **Sleep Stage Classification** (epoch-level, 5 classes: Wake, N1, N2, N3, REM)
2. **ADHD Detection** (patient-level, binary: Non-ADHD vs ADHD)

## Overview

The multi-task model shares a common GraphS4mer backbone for feature extraction and uses two separate classification heads for the respective tasks. This approach allows the model to learn shared representations that benefit both tasks.

### Key Features
- **No size mismatch issues**: Carefully designed data loading and model architecture
- **Flexible architecture**: Supports both S4 and GRU temporal models
- **Graph learning**: Automatically learns brain connectivity patterns
- **Multi-task optimization**: Joint training with configurable loss weights
- **Comprehensive evaluation**: Detailed metrics for both tasks

## File Structure

```
graphs4mer/
├── model/
│   └── multitask_graphs4mer.py      # Multi-task model architecture
├── data/
│   └── multitask_dataset.py         # Data loading utilities
├── train_multitask.py               # Training script
├── inference_multitask.py           # Inference script
└── MULTITASK_README.md              # This file
```

## Data Format

### Input Data Requirements

The code expects pre-epoched PSG data in MNE-Python `.fif` format with the following structure:

1. **File naming**: `*_psg.fif` (e.g., `patient001_psg.fif`)
2. **Data structure**: MNE Epochs object containing:
   - EEG signals: shape `(n_epochs, n_channels, n_timepoints)`
   - Sleep stage labels: stored in `epochs.events[:, 2]`
   - ADHD labels: stored in `epochs.metadata['ADHD']`

### Label Encoding

**Sleep Stages:**
- 0: Wake
- 1: N1 (Stage 1)
- 2: N2 (Stage 2)
- 3: N3 (Stage 3/Deep sleep)
- 4: REM

**ADHD:**
- 0: Non-ADHD
- 1: ADHD

## Installation

### Prerequisites

```bash
pip install torch torch-geometric mne numpy pandas scikit-learn tqdm
```

### Additional Requirements

If you don't have the S4 model dependencies:
```bash
pip install einops opt-einsum
```

## Usage

### 1. Training

Basic training command:

```bash
python train_multitask.py \
    --patients_dir /path/to/patient/data \
    --save_dir ./experiments/multitask_exp1 \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

Advanced training with custom parameters:

```bash
python train_multitask.py \
    --patients_dir /path/to/patient/data \
    --save_dir ./experiments/multitask_exp2 \
    --num_epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_dim 256 \
    --num_temporal_layers 4 \
    --num_gnn_layers 2 \
    --dropout 0.3 \
    --sleep_loss_weight 1.0 \
    --adhd_loss_weight 1.0 \
    --temporal_model s4 \
    --graph_pool mean \
    --temporal_pool mean \
    --patience 20
```

### 2. Inference

Run inference on new data:

```bash
python inference_multitask.py \
    --checkpoint ./experiments/multitask_exp1/best_model.pt \
    --data_dir /path/to/test/data \
    --output_dir ./predictions \
    --save_probabilities
```

### 3. Evaluation Only

Evaluate a trained model without training:

```bash
python train_multitask.py \
    --patients_dir /path/to/patient/data \
    --checkpoint ./experiments/multitask_exp1/best_model.pt \
    --eval_only
```

## Key Parameters

### Data Parameters
- `--patients_dir`: Directory containing `.fif` files
- `--sampling_rate`: Sampling rate in Hz (default: 100)
- `--epoch_length`: Epoch length in seconds (default: 30)
- `--num_nodes`: Number of EEG channels (default: 19)

### Model Parameters
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--num_temporal_layers`: Number of temporal layers (default: 4)
- `--num_gnn_layers`: Number of GNN layers (default: 2)
- `--temporal_model`: Temporal model type - `s4` or `gru` (default: s4)
- `--g_conv`: GNN layer type - `gine` or `graphsage` (default: gine)
- `--dropout`: Dropout rate (default: 0.3)

### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 5e-3)
- `--sleep_loss_weight`: Weight for sleep stage loss (default: 1.0)
- `--adhd_loss_weight`: Weight for ADHD loss (default: 1.0)
- `--patience`: Early stopping patience (default: 20)

### Regularization Parameters
- `--feature_smoothing_weight`: Feature smoothing regularization (default: 0.0)
- `--degree_weight`: Degree regularization (default: 0.0)
- `--sparse_weight`: Sparsity regularization (default: 0.0)

## Output Files

### Training Outputs

After training, the following files are saved in `save_dir`:

1. **`best_model.pt`**: Best model checkpoint based on validation loss
2. **`latest_model.pt`**: Most recent model checkpoint
3. **`args.json`**: Training arguments and hyperparameters
4. **`history.json`**: Training and validation metrics history
5. **`test_results.json`**: Final test set results

### Inference Outputs

Inference produces the following files in `output_dir`:

1. **`predictions.csv`**: Detailed predictions for each epoch
   - Columns: file_name, patient_id, sleep_prediction, sleep_label, adhd_prediction, adhd_label, correctness flags
   - Optional: probability distributions if `--save_probabilities` is used

2. **`patient_level_adhd.csv`**: Patient-level ADHD predictions (aggregated by majority vote)
   - Columns: patient_id, adhd_prediction, adhd_label, correct, prediction_rate

3. **`summary.json`**: Summary statistics
   - Overall accuracies
   - Class distributions
   - Total number of predictions

## Example Workflow

### Complete Pipeline

```bash
# 1. Prepare your data
# Ensure .fif files are in the correct format with both sleep stage and ADHD labels

# 2. Train the model
python train_multitask.py \
    --patients_dir ./data/patients \
    --save_dir ./experiments/exp1 \
    --num_epochs 100 \
    --batch_size 32

# 3. Monitor training
# Check ./experiments/exp1/history.json for training curves

# 4. Evaluate on test set
# The script automatically evaluates on the test set after training

# 5. Run inference on new data
python inference_multitask.py \
    --checkpoint ./experiments/exp1/best_model.pt \
    --data_dir ./data/new_patients \
    --output_dir ./predictions \
    --save_probabilities

# 6. Analyze results
# Check predictions.csv and patient_level_adhd.csv
```

## Model Architecture Details

### Shared Backbone
The model uses a GraphS4mer backbone that consists of:
1. **Temporal Layer**: S4 or GRU for processing time series
2. **Graph Learning**: Learns brain connectivity patterns
3. **GNN Layers**: Process graph-structured data
4. **Pooling**: Temporal and graph pooling for feature aggregation

### Task-Specific Heads

**Sleep Stage Head:**
```
Input (hidden_dim) → Dropout → Linear(hidden_dim/2) → ReLU → Dropout → Linear(5 classes)
```

**ADHD Head:**
```
Input (hidden_dim) → Dropout → Linear(hidden_dim/2) → ReLU → Dropout → Linear(2 classes)
```

### Loss Function

The total loss is a weighted combination:
```
Total Loss = λ_sleep × CrossEntropy(sleep) + λ_adhd × CrossEntropy(adhd) + λ_reg × Regularization
```

Where:
- `λ_sleep`: Sleep stage loss weight (default: 1.0)
- `λ_adhd`: ADHD loss weight (default: 1.0)
- `λ_reg`: Regularization loss weight (default: 0.001)

## Avoiding Size Mismatches

The code includes several safeguards to prevent size mismatches:

1. **Automatic Resolution Calculation**: The temporal resolution is automatically calculated to evenly divide the sequence length
2. **Data Validation**: Input data is validated and padded/cropped to the correct length
3. **Flexible Batch Collation**: Custom collate function handles variable-length sequences
4. **Channel Selection**: Automatically handles different numbers of channels across files

## Performance Tips

### Memory Optimization
- Reduce `batch_size` if running out of GPU memory
- Use smaller `hidden_dim` (e.g., 64 or 128)
- Reduce `num_temporal_layers` or `num_gnn_layers`

### Training Speed
- Use `--num_workers 0` if data loading is slow
- Reduce `num_gnn_layers` for faster training
- Use `temporal_model gru` instead of `s4` for faster iterations

### Accuracy Improvement
- Increase `hidden_dim` to 256 or 512
- Add more `num_temporal_layers` (4-6)
- Tune loss weights (`--sleep_loss_weight`, `--adhd_loss_weight`)
- Enable regularization (`--feature_smoothing_weight 0.01`, `--sparse_weight 0.01`)
- Increase training time with more `num_epochs`

## Troubleshooting

### Common Issues

**1. "No .fif files found"**
- Ensure files end with `_psg.fif`
- Check the `--patients_dir` path is correct

**2. "Missing ADHD metadata"**
- Ensure your `.fif` files have `epochs.metadata['ADHD']` field
- Check the data loading section in the code

**3. "CUDA out of memory"**
- Reduce `--batch_size`
- Reduce `--hidden_dim`
- Use `--device cpu` for CPU-only training

**4. "Size mismatch in forward pass"**
- This should not happen with the current implementation
- If it does, check that `max_seq_len` is divisible by `resolution`
- The code automatically handles this, but you can manually set resolution if needed

**5. "Poor performance"**
- Try different learning rates (`--lr 0.0001` to `0.01`)
- Adjust loss weights to balance tasks
- Increase model capacity (`--hidden_dim 256`)
- Check data quality and label distributions

## Citation

If you use this code, please cite the original GraphS4mer paper and acknowledge the multi-task extension.

## License

This code follows the same license as the parent GraphS4mer repository.
