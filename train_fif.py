"""
Complete training script for custom FIF dataset with sleep stage classification
Reports training metrics, validation performance, and saves best model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import json
from pathlib import Path

from data.datamodules.datamodule_fif import FIF_DataModule
from model.graphs4mer import GraphS4mer


class Trainer:
    def __init__(self, model, datamodule, config):
        self.model = model
        self.datamodule = datamodule
        self.config = config

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"\nUsing device: {self.device}")

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        # Loss function (for multi-class classification)
        self.criterion = nn.CrossEntropyLoss()

        # Tracking
        self.best_val_acc = 0.0
        self.best_val_kappa = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_kappas = []

        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loader = self.datamodule.train_dataloader()

        epoch_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass
            logits, reg_loss_dict = self.model(
                batch,
                epoch=epoch,
                epoch_total=self.config['num_epochs']
            )

            # Compute classification loss
            y = batch.y.long().view(-1)
            cls_loss = self.criterion(logits, y)

            # Add regularization losses
            reg_loss = 0
            for key, loss_val in reg_loss_dict.items():
                reg_loss += self.config['reg_weights'].get(key, 0.01) * loss_val

            # Total loss
            loss = cls_loss + reg_loss

            # Check for NaN before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf detected at batch {batch_idx}!")
                print(f"   cls_loss: {cls_loss.item()}")
                print(f"   reg_loss: {reg_loss.item()}")
                print(f"   Logits - min: {logits.min().item()}, max: {logits.max().item()}")
                print(f"   Input - min: {batch.x.min().item()}, max: {batch.x.max().item()}")
                raise ValueError("NaN/Inf loss detected - stopping training")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Check gradients for NaN
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"\n❌ NaN/Inf gradient in {name}")
                        raise ValueError(f"NaN/Inf gradient in {name}")

            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })

        # Compute epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_kappa = cohen_kappa_score(all_labels, all_preds)

        self.train_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'accuracy': train_acc,
            'kappa': train_kappa
        }

    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set"""
        self.model.eval()
        val_loader = self.datamodule.val_dataloader()

        val_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]")
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            logits, _ = self.model(
                batch,
                epoch=epoch,
                epoch_total=self.config['num_epochs']
            )

            # Compute loss
            y = batch.y.long().view(-1)
            loss = self.criterion(logits, y)

            val_loss += loss.item()

            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute metrics
        avg_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_kappa = cohen_kappa_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        self.val_losses.append(avg_loss)
        self.val_accs.append(val_acc)
        self.val_kappas.append(val_kappa)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'accuracy': val_acc,
            'kappa': val_kappa,
            'f1': val_f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }

    @torch.no_grad()
    def test(self):
        """Test on test set"""
        self.model.eval()
        test_loader = self.datamodule.test_dataloader()

        all_preds = []
        all_labels = []

        print("\nTesting on test set...")
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(self.device)

            # Forward pass
            logits, _ = self.model(batch, epoch=0, epoch_total=1)

            # Predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.long().view(-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        # Compute metrics
        test_acc = accuracy_score(all_labels, all_preds)
        test_kappa = cohen_kappa_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro')

        # Classification report
        print("\n" + "="*80)
        print("TEST SET RESULTS")
        print("="*80)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Cohen's Kappa: {test_kappa:.4f}")
        print(f"Macro F1-Score: {test_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=['Wake', 'N1', 'N2', 'N3', 'REM']
        ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print("Predicted ->")
        print("          Wake   N1    N2    N3   REM")
        for i, row in enumerate(cm):
            print(f"Actual {['Wake', 'N1  ', 'N2  ', 'N3  ', 'REM '][i]}: {row}")

        return {
            'accuracy': test_acc,
            'kappa': test_kappa,
            'f1': test_f1,
            'confusion_matrix': cm
        }

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_kappa': self.best_val_kappa,
            'config': self.config
        }

        # Save last checkpoint
        torch.save(checkpoint, self.save_dir / 'last.ckpt')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.ckpt')
            print(f"  ✓ Saved best model (Acc: {self.best_val_acc:.4f}, Kappa: {self.best_val_kappa:.4f})")

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")

        for epoch in range(self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} Summary")
            print(f"{'='*80}")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"Kappa: {train_metrics['kappa']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Kappa: {val_metrics['kappa']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Check if best model
            is_best = val_metrics['kappa'] > self.best_val_kappa
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_kappa = val_metrics['kappa']

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            print(f"{'='*80}\n")

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Val Kappa: {self.best_val_kappa:.4f}")
        print("="*80 + "\n")

        # Load best model and test
        print("Loading best model for testing...")
        checkpoint = torch.load(self.save_dir / 'best.ckpt')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.test()

        # Save final results
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'val_kappas': self.val_kappas,
            'best_val_acc': self.best_val_acc,
            'best_val_kappa': self.best_val_kappa,
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'kappa': test_metrics['kappa'],
                'f1': test_metrics['f1']
            }
        }

        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {self.save_dir}")


def main():
    """Main training script"""

    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    config = {
        # Data
        'fif_directory': r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining',
        'task': 'sleep_stage',  # 'sleep_stage' or 'adhd'
        'batch_size': 32,
        'num_workers': 4,

        # Model
        'hidden_dim': 128,
        'num_gnn_layers': 1,
        'num_temporal_layers': 4,
        'state_dim': 64,
        'dropout': 0.1,

        # Training
        'num_epochs': 50,
        'lr': 1e-4,  # Reduced from 1e-3 to prevent NaN
        'weight_decay': 1e-4,  # Reduced from 1e-3
        'grad_clip': 1.0,  # Reduced from 5.0 for more aggressive clipping

        # Regularization weights
        'reg_weights': {
            'feature_smoothing': 0.01,
            'degree': 0.01,
            'sparse': 0.01
        },

        # Save
        'save_dir': './results/sleep_stage_training'
    }

    print("="*80)
    print("GraphS4mer Sleep Stage Classification Training")
    print("="*80)
    print("\nConfiguration:")
    for key, value in config.items():
        if key != 'reg_weights':
            print(f"  {key}: {value}")
    print()

    # ========================================================================
    # DATA
    # ========================================================================
    print("Loading data...")
    datamodule = FIF_DataModule(
        fif_directory=config['fif_directory'],
        task=config['task'],
        train_batch_size=config['batch_size'],
        test_batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=False,  # Set to True if using GPU
    )

    datamodule.setup()

    # ========================================================================
    # MODEL
    # ========================================================================
    print("\nCreating model...")
    model = GraphS4mer(
        input_dim=1,
        num_nodes=datamodule.num_nodes,
        dropout=config['dropout'],
        g_conv='gine',
        num_gnn_layers=config['num_gnn_layers'],
        hidden_dim=config['hidden_dim'],
        max_seq_len=datamodule.max_seq_len,
        resolution=datamodule.max_seq_len,  # Use full sequence as one graph
        num_temporal_layers=config['num_temporal_layers'],
        state_dim=config['state_dim'],
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
        num_classes=datamodule.output_dim,
        undirected_graph=True,
        use_prior=False,
        K=2,
        regularizations=['feature_smoothing', 'degree', 'sparse'],
        residual_weight=0.0,
        decay_residual_weight=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ========================================================================
    # TRAINING
    # ========================================================================
    trainer = Trainer(model, datamodule, config)
    trainer.train()


if __name__ == "__main__":
    main()
