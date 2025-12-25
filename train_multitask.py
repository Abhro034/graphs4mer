"""
Training script for Multi-task Sleep Stage Classification and ADHD Detection
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from model.multitask_graphs4mer import MultiTaskGraphS4mer
from data.multitask_dataset import load_patient_data_multitask, multitask_collate_fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score


def get_args():
    parser = argparse.ArgumentParser(description="Multi-task training for Sleep Stage and ADHD Detection")

    # Data args
    parser.add_argument("--patients_dir", type=str, required=True, help="Directory with patient .fif files")
    parser.add_argument("--save_dir", type=str, default="./experiments/multitask", help="Save directory")
    parser.add_argument("--sampling_rate", type=int, default=100, help="Sampling rate (Hz)")
    parser.add_argument("--epoch_length", type=int, default=30, help="Epoch length in seconds")
    parser.add_argument("--num_nodes", type=int, default=19, help="Number of EEG channels")
    parser.add_argument("--picks", type=str, nargs="+", default=None, help="Channel names to use")

    # Model args
    parser.add_argument("--input_dim", type=int, default=1, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_temporal_layers", type=int, default=4, help="Number of temporal layers")
    parser.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--state_dim", type=int, default=64, help="State dimension for S4")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--temporal_model", type=str, default="s4", choices=["s4", "gru"], help="Temporal model")
    parser.add_argument("--g_conv", type=str, default="gine", choices=["gine", "graphsage"], help="GNN layer type")
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "sum", "max"], help="Graph pooling")
    parser.add_argument("--temporal_pool", type=str, default="mean", choices=["mean", "last"], help="Temporal pooling")
    parser.add_argument("--edge_top_perc", type=float, default=0.2, help="Top percentage of edges to keep")
    parser.add_argument("--knn", type=int, default=3, help="K for KNN graph")

    # Training args
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--sleep_loss_weight", type=float, default=1.0, help="Weight for sleep stage loss")
    parser.add_argument("--adhd_loss_weight", type=float, default=1.0, help="Weight for ADHD loss")
    parser.add_argument("--reg_loss_weight", type=float, default=0.001, help="Weight for regularization losses")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--gradient_clip", type=float, default=5.0, help="Gradient clipping value")

    # Regularization weights
    parser.add_argument("--feature_smoothing_weight", type=float, default=0.0, help="Feature smoothing weight")
    parser.add_argument("--degree_weight", type=float, default=0.0, help="Degree regularization weight")
    parser.add_argument("--sparse_weight", type=float, default=0.0, help="Sparsity regularization weight")

    # Other args
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_only", action="store_true", help="Evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation")

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels, task="sleep"):
    """Compute evaluation metrics"""
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels, preds)

    metrics = {
        f"{task}_accuracy": acc,
        f"{task}_f1": f1,
        f"{task}_precision": precision,
        f"{task}_recall": recall,
        f"{task}_kappa": kappa
    }
    return metrics


def train_epoch(model, dataloader, optimizer, args, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_sleep_loss = 0
    total_adhd_loss = 0
    total_reg_loss = 0

    all_sleep_preds = []
    all_sleep_labels = []
    all_adhd_preds = []
    all_adhd_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for batch in pbar:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        sleep_logits, adhd_logits, reg_losses = model(
            batch,
            task="both",
            return_attention=False,
            epoch=epoch,
            epoch_total=total_epochs
        )

        # Compute losses
        sleep_loss = F.cross_entropy(sleep_logits, batch.y_sleep.long())
        adhd_loss = F.cross_entropy(adhd_logits, batch.y_adhd.long())

        # Aggregate regularization losses
        reg_loss = 0.0
        if reg_losses:
            for k, v in reg_losses.items():
                if k == "feature_smoothing":
                    reg_loss += args.feature_smoothing_weight * v
                elif k == "degree":
                    reg_loss += args.degree_weight * v
                elif k == "sparse":
                    reg_loss += args.sparse_weight * v

        # Total loss
        loss = (args.sleep_loss_weight * sleep_loss +
                args.adhd_loss_weight * adhd_loss +
                args.reg_loss_weight * reg_loss)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()

        # Collect predictions
        sleep_preds = torch.argmax(sleep_logits, dim=1).cpu().numpy()
        adhd_preds = torch.argmax(adhd_logits, dim=1).cpu().numpy()

        all_sleep_preds.extend(sleep_preds)
        all_sleep_labels.extend(batch.y_sleep.cpu().numpy())
        all_adhd_preds.extend(adhd_preds)
        all_adhd_labels.extend(batch.y_adhd.cpu().numpy())

        # Update metrics
        total_loss += loss.item()
        total_sleep_loss += sleep_loss.item()
        total_adhd_loss += adhd_loss.item()
        if isinstance(reg_loss, torch.Tensor):
            total_reg_loss += reg_loss.item()

        pbar.set_postfix({
            'loss': loss.item(),
            'sleep_loss': sleep_loss.item(),
            'adhd_loss': adhd_loss.item()
        })

    # Compute metrics
    sleep_metrics = compute_metrics(all_sleep_preds, all_sleep_labels, task="sleep")
    adhd_metrics = compute_metrics(all_adhd_preds, all_adhd_labels, task="adhd")

    results = {
        'loss': total_loss / len(dataloader),
        'sleep_loss': total_sleep_loss / len(dataloader),
        'adhd_loss': total_adhd_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader) if total_reg_loss > 0 else 0,
        **sleep_metrics,
        **adhd_metrics
    }

    return results


def evaluate(model, dataloader, args, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_sleep_loss = 0
    total_adhd_loss = 0

    all_sleep_preds = []
    all_sleep_labels = []
    all_adhd_preds = []
    all_adhd_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass
            sleep_logits, adhd_logits, reg_losses = model(
                batch,
                task="both",
                return_attention=False
            )

            # Compute losses
            sleep_loss = F.cross_entropy(sleep_logits, batch.y_sleep.long())
            adhd_loss = F.cross_entropy(adhd_logits, batch.y_adhd.long())

            loss = args.sleep_loss_weight * sleep_loss + args.adhd_loss_weight * adhd_loss

            # Collect predictions
            sleep_preds = torch.argmax(sleep_logits, dim=1).cpu().numpy()
            adhd_preds = torch.argmax(adhd_logits, dim=1).cpu().numpy()

            all_sleep_preds.extend(sleep_preds)
            all_sleep_labels.extend(batch.y_sleep.cpu().numpy())
            all_adhd_preds.extend(adhd_preds)
            all_adhd_labels.extend(batch.y_adhd.cpu().numpy())

            total_loss += loss.item()
            total_sleep_loss += sleep_loss.item()
            total_adhd_loss += adhd_loss.item()

    # Compute metrics
    sleep_metrics = compute_metrics(all_sleep_preds, all_sleep_labels, task="sleep")
    adhd_metrics = compute_metrics(all_adhd_preds, all_adhd_labels, task="adhd")

    results = {
        'loss': total_loss / len(dataloader),
        'sleep_loss': total_sleep_loss / len(dataloader),
        'adhd_loss': total_adhd_loss / len(dataloader),
        **sleep_metrics,
        **adhd_metrics
    }

    return results


def main():
    args = get_args()
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving to: {args.save_dir}")

    # Save args
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)

    max_seq_len = args.epoch_length * args.sampling_rate

    train_dataset, val_dataset, test_dataset = load_patient_data_multitask(
        patients_dir=args.patients_dir,
        fs_target=args.sampling_rate,
        n_channels=args.num_nodes,
        picks=args.picks,
        return_pyg=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multitask_collate_fn,
        pin_memory=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} epochs")
    print(f"  Val: {len(val_dataset)} epochs")
    print(f"  Test: {len(test_dataset)} epochs")

    # Build model
    print("\n" + "="*50)
    print("Building model...")
    print("="*50)

    # Calculate resolution (must divide max_seq_len evenly)
    resolution = max_seq_len // 10  # Use 10 dynamic graphs
    while max_seq_len % resolution != 0:
        resolution -= 1

    model = MultiTaskGraphS4mer(
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        num_temporal_layers=args.num_temporal_layers,
        g_conv=args.g_conv,
        num_gnn_layers=args.num_gnn_layers,
        hidden_dim=args.hidden_dim,
        max_seq_len=max_seq_len,
        resolution=resolution,
        num_sleep_classes=5,
        num_adhd_classes=2,
        state_dim=args.state_dim,
        temporal_model=args.temporal_model,
        temporal_pool=args.temporal_pool,
        graph_pool=args.graph_pool,
        edge_top_perc=args.edge_top_perc,
        K=args.knn,
        regularizations=["feature_smoothing", "degree", "sparse"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')

    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluation only
    if args.eval_only:
        print("\n" + "="*50)
        print("Evaluating on test set...")
        print("="*50)
        test_results = evaluate(model, test_loader, args, device)
        print("\nTest Results:")
        for k, v in test_results.items():
            print(f"  {k}: {v:.4f}")
        return

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 50)

        # Train
        train_results = train_epoch(model, train_loader, optimizer, args, device, epoch, args.num_epochs)
        train_history.append(train_results)

        # Validate
        val_results = evaluate(model, val_loader, args, device)
        val_history.append(val_results)

        # Print results
        print(f"\nTrain - Loss: {train_results['loss']:.4f} | "
              f"Sleep Acc: {train_results['sleep_accuracy']:.4f} | "
              f"ADHD Acc: {train_results['adhd_accuracy']:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f} | "
              f"Sleep Acc: {val_results['sleep_accuracy']:.4f} | "
              f"ADHD Acc: {val_results['adhd_accuracy']:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_results': val_results,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Save latest model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'val_results': val_results,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pt'))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save training history
    history = {
        'train': train_history,
        'val': val_history
    }
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Evaluating best model on test set...")
    print("="*50)

    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate(model, test_loader, args, device)

    print("\nTest Results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")

    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    print(f"\nTraining complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
