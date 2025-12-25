"""
Inference script for Multi-task Sleep Stage Classification and ADHD Detection
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from model.multitask_graphs4mer import MultiTaskGraphS4mer
from data.multitask_dataset import MultiTaskPSGDataset, multitask_collate_fn


def get_args():
    parser = argparse.ArgumentParser(description="Multi-task inference for Sleep Stage and ADHD Detection")

    # Required args
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with .fif files to predict")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory for predictions")

    # Optional args
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--save_probabilities", action="store_true", help="Save prediction probabilities")

    return parser.parse_args()


def predict(model, dataloader, device, save_probabilities=False):
    """
    Run inference on the data.

    Returns:
        results: Dictionary with predictions and labels
    """
    model.eval()

    all_sleep_preds = []
    all_sleep_probs = []
    all_sleep_labels = []

    all_adhd_preds = []
    all_adhd_probs = []
    all_adhd_labels = []

    all_patient_ids = []
    all_file_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = batch.to(device)

            # Forward pass
            sleep_logits, adhd_logits, _ = model(
                batch,
                task="both",
                return_attention=False
            )

            # Get probabilities
            sleep_probs = F.softmax(sleep_logits, dim=1).cpu().numpy()
            adhd_probs = F.softmax(adhd_logits, dim=1).cpu().numpy()

            # Get predictions
            sleep_preds = np.argmax(sleep_probs, axis=1)
            adhd_preds = np.argmax(adhd_probs, axis=1)

            # Collect results
            all_sleep_preds.extend(sleep_preds)
            all_sleep_probs.extend(sleep_probs)
            all_sleep_labels.extend(batch.y_sleep.cpu().numpy())

            all_adhd_preds.extend(adhd_preds)
            all_adhd_probs.extend(adhd_probs)
            all_adhd_labels.extend(batch.y_adhd.cpu().numpy())

            all_patient_ids.extend(batch.patient_id.cpu().numpy())
            all_file_names.extend(batch.writeout_fn)

    results = {
        'sleep_predictions': np.array(all_sleep_preds),
        'sleep_labels': np.array(all_sleep_labels),
        'adhd_predictions': np.array(all_adhd_preds),
        'adhd_labels': np.array(all_adhd_labels),
        'patient_ids': np.array(all_patient_ids),
        'file_names': all_file_names,
    }

    if save_probabilities:
        results['sleep_probabilities'] = np.array(all_sleep_probs)
        results['adhd_probabilities'] = np.array(all_adhd_probs)

    return results


def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving predictions to: {args.output_dir}")

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']

    # Calculate max_seq_len
    sampling_rate = model_args.get('sampling_rate', 100)
    epoch_length = model_args.get('epoch_length', 30)
    max_seq_len = epoch_length * sampling_rate

    # Calculate resolution
    resolution = max_seq_len // 10
    while max_seq_len % resolution != 0:
        resolution -= 1

    # Build model
    print("\nBuilding model...")
    model = MultiTaskGraphS4mer(
        input_dim=model_args['input_dim'],
        num_nodes=model_args['num_nodes'],
        dropout=model_args['dropout'],
        num_temporal_layers=model_args['num_temporal_layers'],
        g_conv=model_args['g_conv'],
        num_gnn_layers=model_args['num_gnn_layers'],
        hidden_dim=model_args['hidden_dim'],
        max_seq_len=max_seq_len,
        resolution=resolution,
        num_sleep_classes=5,
        num_adhd_classes=2,
        state_dim=model_args['state_dim'],
        temporal_model=model_args['temporal_model'],
        temporal_pool=model_args['temporal_pool'],
        graph_pool=model_args['graph_pool'],
        edge_top_perc=model_args['edge_top_perc'],
        K=model_args['knn'],
        regularizations=["feature_smoothing", "degree", "sparse"],
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load data
    print("\nLoading data...")
    import glob
    file_paths = sorted(glob.glob(os.path.join(args.data_dir, "*_psg.fif")))

    if len(file_paths) == 0:
        raise ValueError(f"No .fif files found in {args.data_dir}")

    print(f"Found {len(file_paths)} files")

    dataset = MultiTaskPSGDataset(
        file_paths=file_paths,
        sampling_rate=sampling_rate,
        n_channels=model_args['num_nodes'],
        picks=model_args.get('picks', None),
        return_pyg=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)} epochs")

    # Run inference
    print("\n" + "="*50)
    print("Running inference...")
    print("="*50)

    results = predict(model, dataloader, device, save_probabilities=args.save_probabilities)

    # Calculate accuracy
    sleep_acc = np.mean(results['sleep_predictions'] == results['sleep_labels'])
    adhd_acc = np.mean(results['adhd_predictions'] == results['adhd_labels'])

    print(f"\nResults:")
    print(f"  Sleep Stage Accuracy: {sleep_acc:.4f}")
    print(f"  ADHD Accuracy: {adhd_acc:.4f}")

    # Create detailed predictions DataFrame
    sleep_label_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    adhd_label_names = ['Non-ADHD', 'ADHD']

    predictions_df = pd.DataFrame({
        'file_name': results['file_names'],
        'patient_id': results['patient_ids'],
        'sleep_prediction': [sleep_label_names[p] for p in results['sleep_predictions']],
        'sleep_label': [sleep_label_names[l] for l in results['sleep_labels']],
        'sleep_correct': results['sleep_predictions'] == results['sleep_labels'],
        'adhd_prediction': [adhd_label_names[p] for p in results['adhd_predictions']],
        'adhd_label': [adhd_label_names[l] for l in results['adhd_labels']],
        'adhd_correct': results['adhd_predictions'] == results['adhd_labels'],
    })

    # Add probabilities if requested
    if args.save_probabilities:
        for i, name in enumerate(sleep_label_names):
            predictions_df[f'sleep_prob_{name}'] = results['sleep_probabilities'][:, i]
        for i, name in enumerate(adhd_label_names):
            predictions_df[f'adhd_prob_{name}'] = results['adhd_probabilities'][:, i]

    # Save predictions
    predictions_csv = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"\nPredictions saved to: {predictions_csv}")

    # Save summary statistics
    summary = {
        'total_epochs': len(results['sleep_predictions']),
        'sleep_stage_accuracy': float(sleep_acc),
        'adhd_accuracy': float(adhd_acc),
        'sleep_stage_distribution': {
            sleep_label_names[i]: int(np.sum(results['sleep_predictions'] == i))
            for i in range(5)
        },
        'adhd_distribution': {
            adhd_label_names[i]: int(np.sum(results['adhd_predictions'] == i))
            for i in range(2)
        }
    }

    summary_json = os.path.join(args.output_dir, 'summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_json}")

    # Patient-level ADHD predictions (aggregate by patient)
    patient_adhd_preds = {}
    patient_adhd_labels = {}

    for i, patient_id in enumerate(results['patient_ids']):
        if patient_id not in patient_adhd_preds:
            patient_adhd_preds[patient_id] = []
            patient_adhd_labels[patient_id] = results['adhd_labels'][i]
        patient_adhd_preds[patient_id].append(results['adhd_predictions'][i])

    # Aggregate by majority vote
    patient_level_results = []
    for patient_id in patient_adhd_preds:
        preds = patient_adhd_preds[patient_id]
        label = patient_adhd_labels[patient_id]
        # Majority vote
        pred = 1 if np.mean(preds) >= 0.5 else 0
        patient_level_results.append({
            'patient_id': patient_id,
            'adhd_prediction': adhd_label_names[pred],
            'adhd_label': adhd_label_names[label],
            'correct': pred == label,
            'prediction_rate': np.mean(preds)
        })

    patient_df = pd.DataFrame(patient_level_results)
    patient_csv = os.path.join(args.output_dir, 'patient_level_adhd.csv')
    patient_df.to_csv(patient_csv, index=False)
    print(f"Patient-level ADHD predictions saved to: {patient_csv}")

    patient_acc = np.mean(patient_df['correct'])
    print(f"Patient-level ADHD Accuracy: {patient_acc:.4f}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
