"""
Performance profiling script to identify training bottlenecks
"""

import torch
import time
from data.datamodules.datamodule_fif import FIF_DataModule
from model.graphs4mer import GraphS4mer

def profile_training_performance():
    """Profile each component to find bottlenecks"""

    print("=" * 80)
    print("TRAINING PERFORMANCE PROFILING")
    print("=" * 80)

    # Config
    fif_directory = r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining'

    # Load data
    print("\n1. DATA LOADING PROFILING")
    print("-" * 80)

    start = time.time()
    datamodule = FIF_DataModule(
        fif_directory=fif_directory,
        task='sleep_stage',
        train_batch_size=32,
        test_batch_size=32,
        num_workers=0  # Check if this is the issue
    )
    datamodule.setup()
    data_load_time = time.time() - start
    print(f"✓ Data loading time: {data_load_time:.2f}s")

    train_loader = datamodule.train_dataloader()

    # Time data iteration
    print("\n2. DATA ITERATOR PROFILING")
    print("-" * 80)
    times = []
    for i, batch in enumerate(train_loader):
        start = time.time()
        # Just load the batch
        _ = batch.x.shape
        times.append(time.time() - start)
        if i >= 10:  # First 10 batches
            break

    avg_data_time = sum(times) / len(times)
    print(f"✓ Average batch loading time: {avg_data_time*1000:.2f}ms")
    print(f"✓ Estimated data loading for 359 batches: {avg_data_time * 359:.2f}s ({avg_data_time * 359 / 60:.2f} min)")

    # Model setup
    print("\n3. MODEL INITIALIZATION")
    print("-" * 80)

    batch = next(iter(train_loader))
    num_nodes = batch.x.size(1)
    max_seq_len = batch.x.size(2)

    print(f"   Data shape: {batch.x.shape}")
    print(f"   Num nodes: {num_nodes}")
    print(f"   Max seq len: {max_seq_len}")
    print(f"   Batch size: {batch.x.size(0) // num_nodes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # Create model with current config
    start = time.time()
    model = GraphS4mer(
        input_dim=1,
        num_nodes=num_nodes,
        dropout=0.1,
        g_conv='gine',
        num_gnn_layers=1,
        hidden_dim=128,
        max_seq_len=max_seq_len,
        resolution=max_seq_len,  # <-- THIS MIGHT BE THE ISSUE!
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
        edge_top_perc=0.5,
        activation_fn='relu',
        num_classes=5,
        undirected_graph=True,
        use_prior=False,
        regularizations=['feature_smoothing', 'degree', 'sparse']
    )
    model = model.to(device)
    model_init_time = time.time() - start

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialization: {model_init_time:.2f}s")
    print(f"✓ Total parameters: {num_params:,}")

    # Profile forward pass
    print("\n4. FORWARD PASS PROFILING")
    print("-" * 80)

    batch = batch.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model(batch, epoch=0, epoch_total=1)

    # Time multiple forward passes
    forward_times = []
    with torch.no_grad():
        for i in range(5):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            logits, reg_loss = model(batch, epoch=0, epoch_total=1)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_times.append(time.time() - start)

    avg_forward = sum(forward_times) / len(forward_times)
    print(f"✓ Average forward pass: {avg_forward*1000:.2f}ms")
    print(f"✓ Estimated forward for 359 batches: {avg_forward * 359:.2f}s ({avg_forward * 359 / 60:.2f} min)")

    # Profile backward pass
    print("\n5. BACKWARD PASS PROFILING")
    print("-" * 80)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    backward_times = []
    for i in range(5):
        optimizer.zero_grad()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()

        logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)
        y = batch.y.long().view(-1)
        cls_loss = torch.nn.functional.cross_entropy(logits, y)

        # Add regularization
        reg_loss = 0
        for key, val in reg_loss_dict.items():
            reg_loss += 0.001 * val

        total_loss = cls_loss + reg_loss
        total_loss.backward()

        torch.cuda.synchronize() if device.type == 'cuda' else None
        backward_times.append(time.time() - start)

    avg_backward = sum(backward_times) / len(backward_times)
    print(f"✓ Average forward+backward: {avg_backward*1000:.2f}ms")
    print(f"✓ Estimated backward for 359 batches: {avg_backward * 359:.2f}s ({avg_backward * 359 / 60:.2f} min)")

    # Total time estimate
    print("\n6. TOTAL TIME ESTIMATE")
    print("-" * 80)

    total_per_batch = avg_data_time + avg_backward
    total_per_epoch = total_per_batch * 359

    print(f"✓ Time per batch: {total_per_batch*1000:.2f}ms")
    print(f"   - Data loading: {avg_data_time*1000:.2f}ms ({avg_data_time/total_per_batch*100:.1f}%)")
    print(f"   - Forward+Backward: {avg_backward*1000:.2f}ms ({avg_backward/total_per_batch*100:.1f}%)")
    print(f"\n✓ Estimated time per epoch: {total_per_epoch:.2f}s = {total_per_epoch/60:.2f} minutes = {total_per_epoch/3600:.2f} hours")

    # Bottleneck analysis
    print("\n7. BOTTLENECK ANALYSIS")
    print("-" * 80)

    bottlenecks = []

    if avg_data_time > 0.1:
        bottlenecks.append(f"❌ Data loading is slow ({avg_data_time*1000:.0f}ms per batch)")
        bottlenecks.append(f"   → Increase num_workers (try 4-8)")

    if avg_forward > 2.0:
        bottlenecks.append(f"❌ Forward pass is very slow ({avg_forward:.2f}s per batch)")
        bottlenecks.append(f"   → Reduce resolution from {max_seq_len} to {max_seq_len // 10} or {max_seq_len // 20}")
        bottlenecks.append(f"   → Reduce num_temporal_layers from 4 to 2")
        bottlenecks.append(f"   → Reduce state_dim from 64 to 32")

    if max_seq_len > 1000:
        bottlenecks.append(f"❌ Sequence length is very long ({max_seq_len})")
        bottlenecks.append(f"   → Use resolution={max_seq_len // 10} to split into 10 graphs")
        bottlenecks.append(f"   → Or use resolution={max_seq_len // 20} to split into 20 graphs")

    if device.type == 'cpu':
        bottlenecks.append(f"❌ Using CPU instead of GPU")
        bottlenecks.append(f"   → Enable CUDA if available")

    if len(bottlenecks) == 0:
        print("✅ No obvious bottlenecks found")
    else:
        for b in bottlenecks:
            print(b)

    # Optimization suggestions
    print("\n8. OPTIMIZATION SUGGESTIONS")
    print("-" * 80)

    print("Quick wins (try these first):")
    print(f"  1. Set num_workers=4 in datamodule")
    print(f"  2. Set resolution={max_seq_len // 10} (from {max_seq_len})")
    print(f"  3. Reduce num_temporal_layers=2 (from 4)")
    print(f"  4. Disable anomaly detection after debugging")
    print(f"\nMedium optimizations:")
    print(f"  5. Reduce state_dim=32 (from 64)")
    print(f"  6. Reduce hidden_dim=64 (from 128)")
    print(f"  7. Reduce batch_size to 16 if GPU memory allows larger batches")
    print(f"\nAdvanced optimizations:")
    print(f"  8. Use mixed precision training (torch.cuda.amp)")
    print(f"  9. Use gradient checkpointing")
    print(f"  10. Profile with torch.profiler for detailed analysis")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    profile_training_performance()
