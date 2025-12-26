"""
Quick check to verify data truncation is working
"""

from data.datamodules.datamodule_fif import FIF_DataModule

fif_directory = r'C:\Users\ap2898\OneDrive - Mississippi State University\Lab Projects\AI in Sleep\Epoch Data\thirty_sec_epochs_UMMC_data\Pretraining'

print("Checking if truncation to 3000 samples is working...")
print("=" * 80)

datamodule = FIF_DataModule(
    fif_directory=fif_directory,
    task='sleep_stage',
    train_batch_size=32,
    test_batch_size=32,
    num_workers=0
)

datamodule.setup()
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print(f"\nBatch shape: {batch.x.shape}")
print(f"  - Batch size: {batch.x.size(0) // batch.x.size(1)}")
print(f"  - Num nodes (channels): {batch.x.size(1)}")
print(f"  - Sequence length: {batch.x.size(2)}")

print(f"\nDataModule properties:")
print(f"  - num_nodes: {datamodule.num_nodes}")
print(f"  - max_seq_len: {datamodule.max_seq_len}")

print("\n" + "=" * 80)
if datamodule.max_seq_len == 3000:
    print("✅ SUCCESS: Data is truncated to 3000 samples")
    print(f"✅ Resolution 300 will work: 3000 % 300 = {3000 % 300}")
elif datamodule.max_seq_len == 3001:
    print("❌ PROBLEM: Data still has 3001 samples")
    print("   The data loader module needs to be reloaded!")
    print("\n   Solution:")
    print("   1. Restart your Python kernel/runtime")
    print("   2. Or run: import importlib; importlib.reload(datamodule)")
else:
    print(f"⚠️  UNEXPECTED: max_seq_len = {datamodule.max_seq_len}")

print("=" * 80)
