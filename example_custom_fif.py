"""
Example: Using custom FIF dataset with sleep stages and ADHD labels
"""

from data.datamodules.datamodule_fif import FIF_DataModule
from model.graphs4mer import GraphS4mer
import torch
import torch.nn.functional as F


def example_sleep_stage_classification():
    """Example: Sleep stage classification (5 classes)"""
    print("=" * 80)
    print("Example 1: Sleep Stage Classification")
    print("=" * 80)

    # Create datamodule for sleep stage classification
    datamodule = FIF_DataModule(
        fif_directory="./path/to/your/fif/files",  # Directory with .fif files
        task='sleep_stage',  # Task: sleep stage classification
        train_batch_size=32,
        test_batch_size=32,
        num_workers=4,
    )

    # Setup datasets
    datamodule.setup()

    # Get dataset properties
    print(f"\nDataset properties:")
    print(f"  Channels: {datamodule.num_nodes}")
    print(f"  Sequence length: {datamodule.max_seq_len}")
    print(f"  Output classes: {datamodule.output_dim}")

    # Create model
    model = GraphS4mer(
        input_dim=1,
        num_nodes=datamodule.num_nodes,
        dropout=0.1,
        g_conv='gine',
        num_gnn_layers=1,
        hidden_dim=128,
        max_seq_len=datamodule.max_seq_len,
        resolution=datamodule.max_seq_len,  # Use full sequence as one graph
        num_temporal_layers=4,
        state_dim=64,
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
        num_classes=datamodule.output_dim,  # 5 classes for sleep stages
        undirected_graph=True,
        use_prior=False,
        K=2,
        regularizations=['feature_smoothing', 'degree', 'sparse'],
        residual_weight=0.0,
        decay_residual_weight=False,
    )

    # Test forward pass
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  x shape: {batch.x.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  Labels: {batch.y.flatten().tolist()}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, reg_loss_dict = model(batch, epoch=0, epoch_total=100)

    print(f"\nModel output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {torch.argmax(logits, dim=1).tolist()}")

    return model, datamodule


def example_adhd_classification():
    """Example: ADHD classification (binary)"""
    print("\n" + "=" * 80)
    print("Example 2: ADHD Classification")
    print("=" * 80)

    # Create datamodule for ADHD classification
    datamodule = FIF_DataModule(
        fif_directory="./path/to/your/fif/files",  # Directory with .fif files
        task='adhd',  # Task: ADHD classification
        train_batch_size=32,
        test_batch_size=32,
        num_workers=4,
    )

    # Setup datasets
    datamodule.setup()

    # Get dataset properties
    print(f"\nDataset properties:")
    print(f"  Channels: {datamodule.num_nodes}")
    print(f"  Sequence length: {datamodule.max_seq_len}")
    print(f"  Output classes: {datamodule.output_dim}")

    # Create model with binary classification
    model = GraphS4mer(
        input_dim=1,
        num_nodes=datamodule.num_nodes,
        dropout=0.1,
        g_conv='gine',
        num_gnn_layers=1,
        hidden_dim=128,
        max_seq_len=datamodule.max_seq_len,
        resolution=datamodule.max_seq_len,
        num_temporal_layers=4,
        state_dim=64,
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
        num_classes=datamodule.output_dim,  # 2 classes for ADHD
        undirected_graph=True,
        use_prior=False,
        K=2,
        regularizations=['feature_smoothing', 'degree', 'sparse'],
        residual_weight=0.0,
        decay_residual_weight=False,
    )

    # Test forward pass
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print(f"\nBatch info:")
    print(f"  x shape: {batch.x.shape}")
    print(f"  y shape: {batch.y.shape}")
    print(f"  Labels: {batch.y.flatten().tolist()}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, reg_loss_dict = model(batch, epoch=0, epoch_total=100)

    print(f"\nModel output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {torch.argmax(logits, dim=1).tolist()}")

    return model, datamodule


def example_training():
    """Example: Simple training loop"""
    print("\n" + "=" * 80)
    print("Example 3: Training Loop")
    print("=" * 80)

    # Create datamodule
    datamodule = FIF_DataModule(
        fif_directory="./path/to/your/fif/files",
        task='sleep_stage',
        train_batch_size=16,
        test_batch_size=32,
        num_workers=0,  # Use 0 for debugging
    )
    datamodule.setup()

    # Create model
    model = GraphS4mer(
        input_dim=1,
        num_nodes=datamodule.num_nodes,
        dropout=0.1,
        g_conv='gine',
        num_gnn_layers=1,
        hidden_dim=64,
        max_seq_len=datamodule.max_seq_len,
        resolution=datamodule.max_seq_len,
        num_temporal_layers=2,
        state_dim=32,
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training loop
    model.train()
    train_loader = datamodule.train_dataloader()

    print("\nStarting training...")
    for epoch in range(2):  # Just 2 epochs for demo
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Just 3 batches for demo
                break

            # Forward pass
            logits, reg_loss_dict = model(batch, epoch=epoch, epoch_total=2)

            # Compute loss
            y = batch.y.long().view(-1)
            cls_loss = F.cross_entropy(logits, y)

            # Add regularization
            reg_loss = sum(reg_loss_dict.values()) * 0.01

            # Total loss
            loss = cls_loss + reg_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")

        print(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss / 3:.4f}\n")

    print("Training complete!")


if __name__ == "__main__":
    # NOTE: Update fif_directory to your actual data path before running!

    print("GraphS4mer Custom FIF Dataset Examples\n")
    print("Update 'fif_directory' paths in this script before running.\n")

    # Uncomment the example you want to run:

    # Example 1: Sleep stage classification
    # model, datamodule = example_sleep_stage_classification()

    # Example 2: ADHD classification
    # model, datamodule = example_adhd_classification()

    # Example 3: Simple training loop
    # example_training()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
