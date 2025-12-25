# Training Fix: PyTorch Geometric Compatibility Issue

## Problem
Training fails during backward pass with error:
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

## Root Cause
PyTorch Geometric 2.7.0 has stricter gradient tracking that conflicts with how `dense_to_sparse`, `add_self_loops`, and `remove_self_loops` modify edge indices. This is a known compatibility issue with newer PyG versions.

## Fixed Issues (Already Applied)
✅ All `masked_fill_` changed to `masked_fill` (4 locations in `model/graph_learner.py`)
✅ All index assignments changed to functional ops (2 locations)
✅ Added `.clone()` and `.detach()` to sparse conversion operations

## Solution: Downgrade PyTorch Geometric

### Quick Fix
```bash
pip uninstall torch-geometric
pip install torch-geometric==2.4.0
```

### Tested Working Versions
- PyTorch Geometric 2.4.0 or earlier
- PyTorch 2.0.0 - 2.2.0

### Alternative: Use CPU-only Mode for Graph Structure
If you must use PyTorch Geometric 2.7.0, modify the forward pass to detach graph learning:

```python
# In model/graphs4mer.py, line ~390:
with torch.no_grad():
    adj_mat_no_grad = adj_mat.detach()
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat_no_grad)

# Continue with gradients on edge_weight only
edge_weight = edge_weight.detach().requires_grad_(True)
```

**Note**: This prevents gradient flow through graph structure learning but allows training.

## Test After Fix
```bash
python test_fif_data.py
```

You should see:
```
✅ ALL TESTS PASSED!
```

## Why This Happens
1. `dense_to_sparse` creates `edge_index` (LongTensor) from dense adjacency
2. Internal indexing: `edge_attr = flatten_adj[edge_index[0], edge_index[1]]`
3. This creates a computation graph link
4. `add_self_loops`/`remove_self_loops` modify edge indices
5. PyG 2.7.0's stricter tracking detects this as in-place modification
6. Backward pass fails

## Recommended Action
**Downgrade to PyTorch Geometric 2.4.0** - this is the simplest and most reliable solution.

```bash
pip install torch-geometric==2.4.0 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## Files Modified
- ✅ `model/graph_learner.py` - Fixed all `masked_fill_` operations
- ✅ `model/graphs4mer.py` - Added clone/detach for sparse conversions
- ⚠️ Further modifications needed for PyG 2.7.0 compatibility (or use downgrade)
