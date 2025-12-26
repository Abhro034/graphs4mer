"""
Test to isolate gradient NaN issue
"""

import torch
from data.datamodules.datamodule_fif import FIF_DataModule
from model.graphs4mer import GraphS4mer

def test_forward_backward():
    """Test forward and backward pass with detailed diagnostics"""

    print("=" * 80)
    print("GRADIENT NaN DIAGNOSTIC TEST")
    print("=" * 80)

    # Config
    fif_directory = r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining'

    # Load data
    print("\n1. Loading data...")
    datamodule = FIF_DataModule(
        fif_directory=fif_directory,
        task='sleep_stage',
        train_batch_size=2,  # Small batch for testing
        test_batch_size=2,
        num_workers=0
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    # Get dimensions
    num_nodes = batch.x.size(1)
    max_seq_len = batch.x.size(2)

    print(f"   Data shape: {batch.x.shape}")
    print(f"   Num nodes: {num_nodes}, Max seq len: {max_seq_len}")

    # Create model
    print("\n2. Creating model...")

    # Use a resolution that divides max_seq_len
    # If max_seq_len is 3000, use 300 (gives 10 graphs)
    # If max_seq_len is 3001, it will fail - need truncation
    if max_seq_len == 3001:
        print("   ⚠️  WARNING: max_seq_len=3001 (prime number)")
        print("   ⚠️  This will fail! Need to restart kernel to reload truncated data.")
        resolution = 3001  # Use full seq to avoid error, but will be slow
    else:
        resolution = 300  # Use efficient resolution

    print(f"   Using resolution={resolution} for max_seq_len={max_seq_len}")

    model = GraphS4mer(
        input_dim=1,
        num_nodes=num_nodes,
        dropout=0.1,
        g_conv='gine',
        num_gnn_layers=1,
        hidden_dim=64,  # Smaller for testing
        max_seq_len=max_seq_len,
        resolution=resolution,  # Use calculated resolution
        num_temporal_layers=2,  # Fewer layers
        state_dim=32,  # Smaller state
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch = batch.to(device)

    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 1: Forward pass only
    print("\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)

    print(f"   ✅ Forward pass successful")
    print(f"   Logits: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
    print(f"   Regularization losses:")
    for key, val in reg_loss_dict.items():
        print(f"      {key}: {val.item():.6f}")

    # Test 2: Forward + backward WITHOUT regularization
    print("\n4. Testing backward WITHOUT regularization...")
    model.train()

    logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)
    y = batch.y.long().view(-1)
    cls_loss = torch.nn.functional.cross_entropy(logits, y)

    print(f"   Classification loss: {cls_loss.item():.4f}")

    cls_loss.backward()

    # Check gradients
    nan_count = 0
    max_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"   ❌ NaN/Inf in {name}")
                nan_count += 1
            else:
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)

    if nan_count == 0:
        print(f"   ✅ No NaN gradients (max grad norm: {max_grad:.4f})")
    else:
        print(f"   ❌ Found {nan_count} parameters with NaN gradients")

    # Test 3: Forward + backward WITH regularization
    print("\n5. Testing backward WITH regularization...")
    model.zero_grad()

    logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)
    y = batch.y.long().view(-1)
    cls_loss = torch.nn.functional.cross_entropy(logits, y)

    # Add regularization
    reg_loss = 0
    for key, val in reg_loss_dict.items():
        reg_loss += 0.001 * val  # Small weight

    total_loss = cls_loss + reg_loss

    print(f"   Classification loss: {cls_loss.item():.4f}")
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    print(f"   Total loss: {total_loss.item():.4f}")

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"   ❌ Loss is NaN/Inf!")
        return

    # Backward with anomaly detection
    print(f"   Running backward pass with anomaly detection...")
    with torch.autograd.set_detect_anomaly(True):
        total_loss.backward()

    # Check gradients
    nan_count = 0
    inf_count = 0
    max_grad = 0
    problematic_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan or has_inf:
                problematic_params.append(name)
                if has_nan:
                    nan_count += 1
                if has_inf:
                    inf_count += 1
            else:
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)

    if nan_count == 0 and inf_count == 0:
        print(f"   ✅ No NaN/Inf gradients (max grad norm: {max_grad:.4f})")
    else:
        print(f"   ❌ Found {nan_count} NaN and {inf_count} Inf gradients")
        print(f"   Problematic parameters:")
        for name in problematic_params[:10]:  # Show first 10
            print(f"      - {name}")

    # Test 4: Try with even smaller regularization
    if nan_count > 0 or inf_count > 0:
        print("\n6. Testing with NO regularization...")
        model.zero_grad()

        logits, reg_loss_dict = model(batch, epoch=0, epoch_total=1)
        y = batch.y.long().view(-1)
        cls_loss = torch.nn.functional.cross_entropy(logits, y)

        cls_loss.backward()

        nan_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_count += 1

        if nan_count == 0:
            print(f"   ✅ No NaN without regularization - regularization is the problem!")
        else:
            print(f"   ❌ Still have NaN without regularization - model issue!")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_forward_backward()
