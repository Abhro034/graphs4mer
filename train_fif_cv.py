"""
5-Fold Stratified Cross-Validation Training for Sleep Stage Classification
with comprehensive metrics and journal-standard plots
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd
import sys

# Force reload data module
modules_to_reload = [
    'data.datamodules.datamodule_fif',
    'data.datamodules',
    'data',
]
for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]

from data.datamodules.datamodule_fif import process_subject_fif, load_fif_data
from torch_geometric.data import Data, Dataset
from model.graphs4mer import GraphS4mer


class FIFDatasetCV(Dataset):
    """Dataset for K-fold cross-validation"""

    def __init__(self, fold_data, sleep_labels, indices, task='sleep_stage'):
        super().__init__()
        self.fold_data = fold_data
        self.sleep_labels = sleep_labels
        self.indices = indices
        self.task = task

        # Build epoch-level mapping
        self.epoch_to_subject = []
        self.epoch_indices = []

        for idx in indices:
            n_epochs = len(sleep_labels[idx])
            self.epoch_to_subject.extend([idx] * n_epochs)
            self.epoch_indices.extend(list(range(n_epochs)))

        # Get dimensions
        first_data = fold_data[indices[0]]
        self.n_channels = first_data.shape[1]
        self.n_samples = first_data.shape[2]
        self.n_classes = 5  # Sleep stages

    def len(self):
        return len(self.epoch_to_subject)

    def get(self, idx):
        subj_idx = self.epoch_to_subject[idx]
        epoch_idx = self.epoch_indices[idx]

        epoch_data = self.fold_data[subj_idx][epoch_idx]
        label = self.sleep_labels[subj_idx][epoch_idx]

        x = torch.FloatTensor(epoch_data).unsqueeze(-1)
        y = torch.LongTensor([label])

        return Data(x=x.float(), y=y)


class CrossValidationTrainer:
    """K-Fold Cross-Validation Trainer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Results storage
        self.fold_results = []
        self.all_histories = {
            'train_loss': [],
            'train_acc': [],
            'train_kappa': [],
            'val_loss': [],
            'val_acc': [],
            'val_kappa': []
        }

        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def create_model(self, num_nodes, max_seq_len):
        """Create model instance"""
        model = GraphS4mer(
            input_dim=1,
            num_nodes=num_nodes,
            dropout=self.config['dropout'],
            g_conv='gine',
            num_gnn_layers=self.config['num_gnn_layers'],
            hidden_dim=self.config['hidden_dim'],
            max_seq_len=max_seq_len,
            resolution=self.config['resolution'],
            num_temporal_layers=self.config['num_temporal_layers'],
            state_dim=self.config['state_dim'],
            channels=1,
            temporal_model='s4',
            bidirectional=False,
            temporal_pool='last',
            prenorm=False,
            postact=None,
            metric='self_attention',
            adj_embed_dim=16,
            gin_mlp=True,
            train_eps=True,
            prune_method='thresh',
            edge_top_perc=0.2,
            thresh=None,
            graph_pool='mean',
            activation_fn='gelu',
            num_classes=5,
            undirected_graph=True,
            use_prior=False,
            K=2,
            regularizations=['feature_smoothing', 'degree', 'sparse'],
            residual_weight=0.0,
            decay_residual_weight=False,
        )
        return model.to(self.device)

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train one epoch"""
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward
            logits, reg_loss_dict = model(batch, epoch=epoch, epoch_total=self.config['num_epochs'])

            y = batch.y.long().view(-1)
            cls_loss = criterion(logits, y)

            # Regularization
            reg_loss = sum(self.config['reg_weights'].get(k, 0.01) * v for k, v in reg_loss_dict.items())
            loss = cls_loss + reg_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])

            optimizer.step()

            # Track
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        return avg_loss, acc, kappa

    @torch.no_grad()
    def validate(self, model, val_loader, criterion, epoch):
        """Validate"""
        model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = batch.to(self.device)

            logits, _ = model(batch, epoch=epoch, epoch_total=self.config['num_epochs'])

            y = batch.y.long().view(-1)
            loss = criterion(logits, y)

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = epoch_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'kappa': kappa,
            'f1_macro': f1,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels
        }

    def train_fold(self, fold, train_dataset, val_dataset):
        """Train one fold"""
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{self.config['n_folds']}")
        print(f"{'='*80}")

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )

        # Model
        model = self.create_model(train_dataset.n_channels, train_dataset.n_samples)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'])
        criterion = nn.CrossEntropyLoss()

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_kappa': [],
            'val_loss': [],
            'val_acc': [],
            'val_kappa': []
        }

        best_val_kappa = 0.0
        best_epoch = 0

        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc, train_kappa = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )

            # Validate
            val_metrics = self.validate(model, val_loader, criterion, epoch)

            # Update scheduler
            scheduler.step()

            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_kappa'].append(train_kappa)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_kappa'].append(val_metrics['kappa'])

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"Kappa: {val_metrics['kappa']:.4f}, F1: {val_metrics['f1_macro']:.4f}")

            # Save best model
            if val_metrics['kappa'] > best_val_kappa:
                best_val_kappa = val_metrics['kappa']
                best_epoch = epoch
                torch.save(model.state_dict(), self.save_dir / f'fold_{fold+1}_best.ckpt')

        # Load best model and get final validation metrics
        model.load_state_dict(torch.load(self.save_dir / f'fold_{fold+1}_best.ckpt'))
        final_metrics = self.validate(model, val_loader, criterion, best_epoch)

        print(f"\nFold {fold+1} Best Results (Epoch {best_epoch+1}):")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Cohen's Kappa: {final_metrics['kappa']:.4f}")
        print(f"  Macro F1: {final_metrics['f1_macro']:.4f}")

        # Store results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': best_epoch + 1,
            'history': history,
            'final_metrics': final_metrics
        }

        self.fold_results.append(fold_result)

        return fold_result

    def run_cross_validation(self, fold_data, sleep_labels):
        """Run k-fold cross-validation"""

        # Get all labels for stratification (use first epoch of each subject)
        subject_labels = [labels[0] for labels in sleep_labels]
        subject_indices = np.arange(len(fold_data))

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(subject_indices, subject_labels)):
            # Create datasets
            train_dataset = FIFDatasetCV(fold_data, sleep_labels, train_idx.tolist())
            val_dataset = FIFDatasetCV(fold_data, sleep_labels, val_idx.tolist())

            print(f"\nFold {fold+1}: Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")
            print(f"           Train epochs: {len(train_dataset)}, Val epochs: {len(val_dataset)}")

            # Train fold
            self.train_fold(fold, train_dataset, val_dataset)

        # Aggregate results
        self.aggregate_results()

        # Create plots
        self.create_plots()

        # Save results
        self.save_results()

    def aggregate_results(self):
        """Aggregate results across folds"""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")

        # Overall metrics
        accuracies = [r['final_metrics']['accuracy'] for r in self.fold_results]
        kappas = [r['final_metrics']['kappa'] for r in self.fold_results]
        f1_macros = [r['final_metrics']['f1_macro'] for r in self.fold_results]

        print(f"\nOverall Metrics (Mean ± Std across {self.config['n_folds']} folds):")
        print(f"  Accuracy:      {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Cohen's Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        print(f"  Macro F1:      {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")

        # Per-stage metrics
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

        print(f"\nPer-Stage Metrics (Mean ± Std):")
        print(f"{'Stage':<8} {'Precision':<18} {'Recall':<18} {'F1-Score':<18} {'Support':<10}")
        print("-" * 80)

        for i, stage in enumerate(stage_names):
            precisions = [r['final_metrics']['precision_per_class'][i] for r in self.fold_results]
            recalls = [r['final_metrics']['recall_per_class'][i] for r in self.fold_results]
            f1s = [r['final_metrics']['f1_per_class'][i] for r in self.fold_results]
            supports = [r['final_metrics']['support'][i] for r in self.fold_results]

            print(f"{stage:<8} "
                  f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}   "
                  f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}   "
                  f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}   "
                  f"{int(np.mean(supports)):>6}")

        # Aggregate confusion matrix
        avg_cm = np.mean([r['final_metrics']['confusion_matrix'] for r in self.fold_results], axis=0)
        print(f"\nAverage Confusion Matrix:")
        print("Predicted ->")
        print(f"{'':>10} " + " ".join(f"{s:>8}" for s in stage_names))
        for i, stage in enumerate(stage_names):
            print(f"Actual {stage:<4} " + " ".join(f"{avg_cm[i][j]:>8.1f}" for j in range(5)))

    def create_plots(self):
        """Create journal-standard plots"""
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

        # Set journal style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # 1. Loss curves for all folds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training and Validation Loss Curves (5-Fold CV)', fontsize=14, fontweight='bold')

        for fold_idx, result in enumerate(self.fold_results):
            ax = axes[fold_idx // 3, fold_idx % 3]
            history = result['history']
            epochs = range(1, len(history['train_loss']) + 1)

            ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'loss_curves.pdf', bbox_inches='tight')
        plt.close()

        # 2. Accuracy curves for all folds
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training and Validation Accuracy Curves (5-Fold CV)', fontsize=14, fontweight='bold')

        for fold_idx, result in enumerate(self.fold_results):
            ax = axes[fold_idx // 3, fold_idx % 3]
            history = result['history']
            epochs = range(1, len(history['train_acc']) + 1)

            ax.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
            ax.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'accuracy_curves.pdf', bbox_inches='tight')
        plt.close()

        # 3. Cohen's Kappa curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Validation Cohen's Kappa Curves (5-Fold CV)", fontsize=14, fontweight='bold')

        for fold_idx, result in enumerate(self.fold_results):
            ax = axes[fold_idx // 3, fold_idx % 3]
            history = result['history']
            epochs = range(1, len(history['val_kappa']) + 1)

            ax.plot(epochs, history['val_kappa'], linewidth=2, color='#2E86AB')
            ax.axhline(y=result['final_metrics']['kappa'], color='r', linestyle='--',
                      label=f"Best: {result['final_metrics']['kappa']:.3f}", linewidth=1.5)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel("Cohen's Kappa", fontsize=10)
            ax.set_title(f'Fold {fold_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'kappa_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'kappa_curves.pdf', bbox_inches='tight')
        plt.close()

        # 4. Average confusion matrix
        avg_cm = np.mean([r['final_metrics']['confusion_matrix'] for r in self.fold_results], axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=stage_names, yticklabels=stage_names,
                   cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 11})
        ax.set_xlabel('Predicted Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Stage', fontsize=12, fontweight='bold')
        ax.set_title('Average Confusion Matrix (5-Fold CV)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()

        # 5. Per-stage performance comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics_data = {
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }

        for i in range(5):
            metrics_data['Precision'].append([
                r['final_metrics']['precision_per_class'][i] for r in self.fold_results
            ])
            metrics_data['Recall'].append([
                r['final_metrics']['recall_per_class'][i] for r in self.fold_results
            ])
            metrics_data['F1-Score'].append([
                r['final_metrics']['f1_per_class'][i] for r in self.fold_results
            ])

        for idx, (metric_name, metric_data) in enumerate(metrics_data.items()):
            ax = axes[idx]

            means = [np.mean(stage_scores) for stage_scores in metric_data]
            stds = [np.std(stage_scores) for stage_scores in metric_data]

            x = np.arange(len(stage_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                         color=sns.color_palette("husl", 5))

            ax.set_xlabel('Sleep Stage', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} by Sleep Stage', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(stage_names)
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'per_stage_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'per_stage_metrics.pdf', bbox_inches='tight')
        plt.close()

        # 6. Overall metrics box plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        overall_metrics = {
            'Accuracy': [r['final_metrics']['accuracy'] for r in self.fold_results],
            "Cohen's Kappa": [r['final_metrics']['kappa'] for r in self.fold_results],
            'Macro F1': [r['final_metrics']['f1_macro'] for r in self.fold_results]
        }

        for idx, (metric_name, values) in enumerate(overall_metrics.items()):
            ax = axes[idx]
            bp = ax.boxplot([values], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=8))

            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} Distribution\n(Mean: {np.mean(values):.4f} ± {np.std(values):.4f})',
                        fontsize=12, fontweight='bold')
            ax.set_xticklabels(['5-Fold CV'])
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.1])

            # Add individual points
            x = np.random.normal(1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.6, s=100, color='darkblue')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'overall_metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.save_dir / 'overall_metrics_boxplot.pdf', bbox_inches='tight')
        plt.close()

        print(f"\nPlots saved to {self.save_dir}/")

    def save_results(self):
        """Save detailed results"""

        # Summary statistics
        summary = {
            'overall_metrics': {
                'accuracy': {
                    'mean': float(np.mean([r['final_metrics']['accuracy'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['accuracy'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['accuracy']) for r in self.fold_results]
                },
                'kappa': {
                    'mean': float(np.mean([r['final_metrics']['kappa'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['kappa'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['kappa']) for r in self.fold_results]
                },
                'f1_macro': {
                    'mean': float(np.mean([r['final_metrics']['f1_macro'] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['f1_macro'] for r in self.fold_results])),
                    'folds': [float(r['final_metrics']['f1_macro']) for r in self.fold_results]
                }
            },
            'per_stage_metrics': {}
        }

        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        for i, stage in enumerate(stage_names):
            summary['per_stage_metrics'][stage] = {
                'precision': {
                    'mean': float(np.mean([r['final_metrics']['precision_per_class'][i] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['precision_per_class'][i] for r in self.fold_results]))
                },
                'recall': {
                    'mean': float(np.mean([r['final_metrics']['recall_per_class'][i] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['recall_per_class'][i] for r in self.fold_results]))
                },
                'f1': {
                    'mean': float(np.mean([r['final_metrics']['f1_per_class'][i] for r in self.fold_results])),
                    'std': float(np.std([r['final_metrics']['f1_per_class'][i] for r in self.fold_results]))
                }
            }

        # Save summary
        with open(self.save_dir / 'cv_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed fold results
        with open(self.save_dir / 'fold_results.json', 'w') as f:
            json.dump(self.fold_results, f, indent=2)

        # Create results table
        results_df = pd.DataFrame({
            'Fold': [r['fold'] for r in self.fold_results],
            'Best Epoch': [r['best_epoch'] for r in self.fold_results],
            'Accuracy': [r['final_metrics']['accuracy'] for r in self.fold_results],
            "Cohen's Kappa": [r['final_metrics']['kappa'] for r in self.fold_results],
            'Macro F1': [r['final_metrics']['f1_macro'] for r in self.fold_results]
        })

        results_df.to_csv(self.save_dir / 'fold_results.csv', index=False)

        print(f"\nResults saved to {self.save_dir}/")


def main():
    """Main cross-validation script"""

    config = {
        # Data
        'fif_directory': r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining',
        'task': 'sleep_stage',
        'batch_size': 32,
        'num_workers': 4,

        # Cross-validation
        'n_folds': 5,

        # Model
        'hidden_dim': 128,
        'num_gnn_layers': 1,
        'num_temporal_layers': 2,
        'state_dim': 32,
        'dropout': 0.1,
        'resolution': 300,

        # Training
        'num_epochs': 50,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,

        # Regularization
        'reg_weights': {
            'feature_smoothing': 0.001,
            'degree': 0.001,
            'sparse': 0.001
        },

        # Save
        'save_dir': './results/sleep_stage_cv'
    }

    print("=" * 80)
    print("GraphS4mer 5-Fold Stratified Cross-Validation")
    print("Sleep Stage Classification")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        if key != 'reg_weights':
            print(f"  {key}: {value}")
    print()

    # Load all data
    print("Loading data...")
    fold_data, sleep_labels, adhd_labels, subject_ids, fold_len = load_fif_data(config['fif_directory'])

    print(f"Loaded {len(fold_data)} subjects")
    print(f"Total epochs: {sum(len(labels) for labels in sleep_labels)}")

    # Run cross-validation
    trainer = CrossValidationTrainer(config)
    trainer.run_cross_validation(fold_data, sleep_labels)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
