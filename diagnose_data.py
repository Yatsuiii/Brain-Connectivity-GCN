#!/usr/bin/env python3
"""Diagnostic script to understand data issues."""

import argparse
import numpy as np
import torch
from pathlib import Path
from collections import Counter

def diagnose():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    processed_dir = Path(args.data_dir) / "processed"
    
    if not processed_dir.exists():
        print(f"❌ NO PREPROCESSED DATA FOUND at {processed_dir}")
        print("   Run: python -m brain_gcn.main --prepare_data --max_epochs 1")
        return
    
    npz_files = sorted(processed_dir.glob("*.npz"))
    print(f"\n📊 FOUND {len(npz_files)} SUBJECT FILES")
    
    if len(npz_files) == 0:
        print("❌ No .npz files found. Preprocessing failed.")
        return
    
    # Collect statistics
    labels = []
    sites = []
    window_counts = []
    num_nodes_list = []
    
    for f in npz_files[:100]:  # Sample first 100
        data = np.load(f, allow_pickle=True)
        labels.append(int(data["label"]))
        sites.append(str(data.get("site", "unknown")))
        
        if "bold_windows" in data:
            w, n = data["bold_windows"].shape
            window_counts.append(w)
        elif "window_bold" in data:
            w, n = data["window_bold"].shape
            window_counts.append(w)
        else:
            print(f"❌ File {f.name} has no bold_windows or window_bold")
            continue
            
        if "mean_fc" in data:
            num_nodes_list.append(data["mean_fc"].shape[0])
    
    # Summary statistics
    label_dist = Counter(labels)
    print(f"\n✓ Class distribution: TD={label_dist[0]}, ASD={label_dist[1]}")
    print(f"  Ratio: {label_dist[1] / (label_dist[0] + label_dist[1]):.2%} ASD")
    
    if label_dist[0] < 10 or label_dist[1] < 10:
        print("⚠️  TOO FEW SAMPLES PER CLASS - model may guess randomly")
    
    print(f"✓ Sites: {Counter(sites)}")
    print(f"✓ Windows per subject: min={min(window_counts)}, max={max(window_counts)}, mean={np.mean(window_counts):.1f}")
    print(f"✓ ROIs (nodes): {num_nodes_list[0] if num_nodes_list else 'unknown'}")
    
    # Check a sample
    print(f"\n📝 Sample file: {npz_files[0].name}")
    sample = np.load(npz_files[0], allow_pickle=True)
    print(f"  Keys: {list(sample.keys())}")
    
    if "bold_windows" in sample:
        print(f"  bold_windows shape: {sample['bold_windows'].shape}")
    if "mean_fc" in sample:
        print(f"  mean_fc shape: {sample['mean_fc'].shape}")
        fc_zero = (sample['mean_fc'] == 0).sum() / sample['mean_fc'].size
        print(f"  mean_fc sparsity: {fc_zero:.1%} zeros")
    if "label" in sample:
        print(f"  label: {sample['label']}")
    
    # Test data loading
    print(f"\n🔄 Testing data loading...")
    from brain_gcn.utils.data.dataset import ABIDEDataset
    from brain_gcn.utils.data.functional_connectivity import compute_population_adj
    
    dataset = ABIDEDataset(npz_files[:20])
    
    # Compute pop adj
    pop_adj = compute_population_adj(npz_files[:20], fc_threshold=0.2)
    print(f"✓ Population adj shape: {pop_adj.shape}, sparsity: {(pop_adj == 0).sum() / pop_adj.size:.1%}")
    
    # Load a sample
    bold, adj, label = dataset[0]
    print(f"✓ Loaded batch 0:")
    print(f"  bold: shape={bold.shape}, dtype={bold.dtype}, range=[{bold.min():.3f}, {bold.max():.3f}]")
    print(f"  adj:  shape={adj.shape}, dtype={adj.dtype}, range=[{adj.min():.3f}, {adj.max():.3f}]")
    print(f"  label: {label.item()}")
    
    # Check if there's signal
    bold_std = bold.std()
    print(f"\n✓ Signal stats:")
    print(f"  BOLD std: {bold_std:.6f}")
    if bold_std < 0.001:
        print("  ⚠️  BOLD signal very small - may not train well")
    
    print("\n✅ Data looks good for training!")

if __name__ == "__main__":
    diagnose()
