"""
Debug NaN losses in training
"""

import torch
import numpy as np
from data.datamodules.datamodule_fif import FIF_DataModule
from model.graphs4mer import GraphS4mer
import mne

def check_data_for_nan(datamodule):
    """Check if input data contains NaN or Inf"""
    print("=" * 80)
    print("CHECKING DATA FOR NaN/Inf")
    print("=" * 80)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()

    nan_count = 0
    inf_count = 0
    total_batches = 0

    for i, batch in enumerate(train_loader):
        total_batches += 1

        # Check features
        has_nan = torch.isnan(batch.x).any().item()
        has_inf = torch.isinf(batch.x).any().item()

        if has_nan:
            nan_count += 1
            print(f"Batch {i}: FOUND NaN in features!")
            print(f"  Shape: {batch.x.shape}")
            print(f"  NaN locations: {torch.isnan(batch.x).sum().item()} values")

        if has_inf:
            inf_count += 1
            print(f"Batch {i}: FOUND Inf in features!")
            print(f"  Shape: {batch.x.shape}")
            print(f"  Inf locations: {torch.isinf(batch.x).sum().item()} values")

        # Check labels
        if torch.isnan(batch.y).any() or torch.isinf(batch.y).any():
            print(f"Batch {i}: FOUND NaN/Inf in labels!")
            print(f"  Labels: {batch.y}")

        # Print stats for first batch
        if i == 0:
            print(f"\nFirst batch statistics:")
            print(f"  Shape: {batch.x.shape}")
            print(f"  Min: {batch.x.min().item():.4f}")
            print(f"  Max: {batch.x.max().item():.4f}")
            print(f"  Mean: {batch.x.mean().item():.4f}")
            print(f"  Std: {batch.x.std().item():.4f}")
            print(f"  Labels: {batch.y.unique()}")

        if i >= 10:  # Check first 10 batches
            break

    print(f"\nSummary (first 10 batches):")
    print(f"  Batches with NaN: {nan_count}")
    print(f"  Batches with Inf: {inf_count}")
    print()

def check_model_forward(model, batch, device):
    """Check model forward pass for NaN"""
    print("=" * 80)
    print("CHECKING MODEL FORWARD PASS")
    print("=" * 80)

    model.eval()
    batch = batch.to(device)

    # Register hooks to catch NaN in intermediate layers
    nan_detected = {'flag': False, 'layer': None}

    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                nan_detected['flag'] = True
                nan_detected['layer'] = module.__class__.__name__
                print(f"  ❌ NaN detected in {module.__class__.__name__}")
                print(f"     Output shape: {output.shape}")
                print(f"     NaN count: {torch.isnan(output).sum().item()}")

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(forward_hook))

    print("\nRunning forward pass with hooks...")
    try:
        with torch.no_grad():
            logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)

        print(f"\nForward pass completed!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits - Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}, Mean: {logits.mean().item():.4f}")
        print(f"  Has NaN: {torch.isnan(logits).any().item()}")
        print(f"  Has Inf: {torch.isinf(logits).any().item()}")

        print(f"\nRegularization losses:")
        for key, val in reg_loss_dict.items():
            print(f"  {key}: {val.item():.6f} (NaN: {torch.isnan(val).any().item()})")

        if nan_detected['flag']:
            print(f"\n❌ NaN first appeared in: {nan_detected['layer']}")
        else:
            print(f"\n✅ No NaN detected in forward pass!")

    except Exception as e:
        print(f"\n❌ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Remove hooks
    for hook in hooks:
        hook.remove()

def check_model_initialization(model):
    """Check if model weights are initialized properly"""
    print("=" * 80)
    print("CHECKING MODEL INITIALIZATION")
    print("=" * 80)

    for name, param in model.named_parameters():
        if param.requires_grad:
            has_nan = torch.isnan(param).any().item()
            has_inf = torch.isinf(param).any().item()

            if has_nan or has_inf:
                print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
            else:
                # Just show first few parameters
                if 'weight' in name:
                    print(f"✅ {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

def check_graph_learning_stability(model, batch, device):
    """Check graph learning operations for numerical issues"""
    print("=" * 80)
    print("CHECKING GRAPH LEARNING STABILITY")
    print("=" * 80)

    model.eval()
    batch = batch.to(device)

    # Get node features
    x = batch.x.float()
    batch_size = batch.batch.max().item() + 1
    num_nodes = x.size(0) // batch_size

    print(f"Input shape: {x.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Num nodes: {num_nodes}")
    print(f"Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")

    # Check if node features have extreme values
    if x.abs().max() > 1e6:
        print(f"⚠️  WARNING: Input features have very large values (max={x.abs().max().item():.2e})")

    if x.abs().min() < 1e-10 and x.abs().min() > 0:
        print(f"⚠️  WARNING: Input features have very small values (min={x.abs().min().item():.2e})")

def test_simple_forward():
    """Test with synthetic data"""
    print("=" * 80)
    print("TESTING WITH SYNTHETIC DATA")
    print("=" * 80)

    from torch_geometric.data import Data, Batch

    # Create simple synthetic data
    num_nodes = 10
    seq_len = 3001
    batch_size = 2

    # Create batched data
    data_list = []
    for i in range(batch_size):
        x = torch.randn(num_nodes, seq_len, 1)  # Normal distribution
        y = torch.LongTensor([i % 5])  # Labels 0-4
        data_list.append(Data(x=x, y=y))

    batch = Batch.from_data_list(data_list)

    print(f"Synthetic data created:")
    print(f"  Shape: {batch.x.shape}")
    print(f"  Stats: min={batch.x.min().item():.4f}, max={batch.x.max().item():.4f}, mean={batch.x.mean().item():.4f}")

    # Create model
    model = GraphS4mer(
        num_nodes=num_nodes,
        num_classes=5,
        max_seq_len=seq_len,
        resolution=seq_len,
        hidden_dim=64,  # Smaller for testing
        num_gnn_layers=1,
        num_temporal_layers=2,
        state_dim=32,
        dropout=0.1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch = batch.to(device)

    print(f"\nTesting forward pass on synthetic data...")
    try:
        with torch.no_grad():
            logits, reg_loss = model(batch, epoch=0, epoch_total=1)

        print(f"✅ Forward pass successful!")
        print(f"  Logits: {logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        print(f"  Has NaN: {torch.isnan(logits).any().item()}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n" + "=" * 80)
    print("NaN DEBUGGING SCRIPT")
    print("=" * 80 + "\n")

    # Configuration (update with your path)
    fif_directory = r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining'

    # Test 1: Synthetic data first
    print("\n📍 TEST 1: Synthetic Data")
    test_simple_forward()

    # Test 2: Check real data
    print("\n📍 TEST 2: Real Data Inspection")
    try:
        datamodule = FIF_DataModule(
            fif_directory=fif_directory,
            task='sleep_stage',
            train_batch_size=32,
            test_batch_size=32,
            num_workers=0  # Disable multiprocessing for debugging
        )

        check_data_for_nan(datamodule)

        # Get a sample batch
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))

        # Get dimensions from data
        num_nodes = sample_batch.x.size(1)  # Number of channels
        max_seq_len = sample_batch.x.size(2)  # Samples per epoch

        print(f"\nDetected dimensions:")
        print(f"  Num nodes (channels): {num_nodes}")
        print(f"  Max seq len (samples): {max_seq_len}")

        # Test 3: Model initialization
        print("\n📍 TEST 3: Model Initialization")
        model = GraphS4mer(
            num_nodes=num_nodes,
            num_classes=5,
            max_seq_len=max_seq_len,
            resolution=max_seq_len,
            hidden_dim=128,
            num_gnn_layers=1,
            num_temporal_layers=4,
            state_dim=64,
            dropout=0.1
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        check_model_initialization(model)

        # Test 4: Forward pass
        print("\n📍 TEST 4: Forward Pass on Real Data")
        check_model_forward(model, sample_batch, device)

        # Test 5: Graph learning
        print("\n📍 TEST 5: Graph Learning Stability")
        check_graph_learning_stability(model, sample_batch, device)

    except Exception as e:
        print(f"\n❌ Error during real data testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
