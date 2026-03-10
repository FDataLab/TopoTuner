#!/usr/bin/env python3
"""
Count the number of numpy weight files saved in each experiment.
This verifies that frozen layers are not saved (fewer files = more frozen layers).
"""

import os
from pathlib import Path

def count_files_in_experiment(base_dir, exp_name):
    """Count numpy weight files in epoch 0 for an experiment."""
    exp_dir = Path(base_dir) / exp_name
    epoch0_dir = exp_dir / "epoch_weights" / "checkpoint-epoch-0" / "numpy_weights"
    
    if not epoch0_dir.exists():
        return None, None
    
    # Count all .npy files
    npy_files = list(epoch0_dir.glob("*.npy"))
    total_files = len(npy_files)
    
    # Count by projection type
    q_count = len([f for f in npy_files if "_q.npy" in f.name])
    k_count = len([f for f in npy_files if "_k.npy" in f.name])
    v_count = len([f for f in npy_files if "_v.npy" in f.name])
    
    return total_files, {'q': q_count, 'k': k_count, 'v': v_count}

def main():
    base_dir = "/home/kadir/topo/numpy_weights/imdb/llama32_3b"
    
    experiments = [
        "full",
        "lowest_3",
        "lowest_6",
        "lowest_9",
        "lowest_12",
        "lowest_15",
        "highest_3",
        "highest_6",
        "highest_9",
        "highest_12",
        "highest_15",
    ]
    
    print("=" * 100)
    print(f"{'Experiment':<20} {'Total Files':<15} {'Q Files':<12} {'K Files':<12} {'V Files':<12} {'Expected'}")
    print("=" * 100)
    
    # Expected: 28 layers * 3 projections = 84 files (if nothing frozen)
    for exp in experiments:
        total, counts = count_files_in_experiment(base_dir, exp)
        
        if total is None:
            print(f"{exp:<20} {'NOT FOUND':<15}")
            continue
        
        # Calculate expected based on freezing
        if exp == 'full':
            expected = 84
        elif 'lowest_' in exp:
            n = int(exp.split('_')[1])
            expected = 84 - (n * 3)  # n layers frozen for each Q, K, V
        elif 'highest_' in exp:
            n = int(exp.split('_')[1])
            expected = 84 - (n * 3)  # n layers frozen for each Q, K, V
        else:
            expected = 84
        
        status = "✅" if total == expected else "❌"
        
        print(f"{exp:<20} {total:<15} {counts['q']:<12} {counts['k']:<12} {counts['v']:<12} {expected} {status}")
    
    print("=" * 100)
    print("\nNote: Expected = 84 - (frozen_layers * 3)")
    print("      Each frozen layer removes Q, K, and V files (3 files per layer)")

if __name__ == "__main__":
    main()
