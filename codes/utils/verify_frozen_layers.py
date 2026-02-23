#!/usr/bin/env python3
"""
Verify which layers are actually saved vs which should be frozen.
This helps confirm that frozen layers are truly not saved as files.
"""

import os
from pathlib import Path
import re

def get_frozen_layers_from_experiment_name(exp_name: str):
    """Get which layers should be frozen based on experiment name."""
    Q_ORDERED = [0, 16, 23, 18, 20, 17, 5, 10, 24, 12, 15, 8, 13, 19, 3, 6, 22, 2, 9, 21, 25, 11, 4, 14, 1, 7, 26, 27]
    K_ORDERED = [15, 8, 0, 12, 16, 18, 14, 17, 21, 9, 13, 10, 19, 11, 20, 24, 4, 6, 23, 7, 5, 25, 26, 22, 3, 27, 1, 2]
    V_ORDERED = [25, 23, 27, 24, 26, 20, 22, 21, 18, 15, 19, 3, 14, 13, 16, 1, 17, 11, 9, 6, 4, 8, 12, 7, 5, 10, 2, 0]
    
    frozen = {'q': [], 'k': [], 'v': []}
    
    if exp_name == 'full':
        return frozen
    
    if 'lowest_' in exp_name:
        n = int(exp_name.split('_')[1])
        frozen['q'] = Q_ORDERED[:n]
        frozen['k'] = K_ORDERED[:n]
        frozen['v'] = V_ORDERED[:n]
    elif 'highest_' in exp_name:
        n = int(exp_name.split('_')[1])
        frozen['q'] = Q_ORDERED[-n:]
        frozen['k'] = K_ORDERED[-n:]
        frozen['v'] = V_ORDERED[-n:]
    
    return frozen

def parse_filename(filename):
    """Parse layer0_q.npy -> (0, 'q')"""
    match = re.match(r"layer(\d+)_(q|k|v)\.npy", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None

def verify_experiment(base_dir, exp_name):
    """Verify which layers are saved vs which should be frozen."""
    exp_dir = Path(base_dir) / exp_name
    epoch0_dir = exp_dir / "epoch_weights" / "checkpoint-epoch-0" / "numpy_weights"
    
    if not epoch0_dir.exists():
        print(f"⚠️  {exp_name}: epoch_weights not found")
        return
    
    frozen = get_frozen_layers_from_experiment_name(exp_name)
    
    # Get all saved files
    saved_files = list(epoch0_dir.glob("*.npy"))
    saved_layers = {}
    for f in saved_files:
        parsed = parse_filename(f.name)
        if parsed:
            layer, proj = parsed
            if layer not in saved_layers:
                saved_layers[layer] = set()
            saved_layers[layer].add(proj)
    
    # Check each layer and projection
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Expected frozen: Q={frozen['q']}, K={frozen['k']}, V={frozen['v']}")
    print(f"{'='*80}")
    
    issues = []
    for layer in range(28):
        for proj in ['q', 'k', 'v']:
            should_be_frozen = layer in frozen[proj]
            is_saved = layer in saved_layers and proj in saved_layers[layer]
            
            if should_be_frozen and is_saved:
                issues.append(f"  ❌ Layer {layer} {proj.upper()}: Should be FROZEN but FILE EXISTS!")
            elif should_be_frozen and not is_saved:
                print(f"  ✅ Layer {layer} {proj.upper()}: FROZEN (no file) - CORRECT")
            elif not should_be_frozen and not is_saved:
                issues.append(f"  ⚠️  Layer {layer} {proj.upper()}: Should NOT be frozen but NO FILE!")
            # elif not should_be_frozen and is_saved:  # This is correct, no issue
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ All frozen layers correctly have no files, all unfrozen layers have files")
    
    # Summary
    total_should_be_frozen = len(frozen['q']) + len(frozen['k']) + len(frozen['v'])
    total_saved = len(saved_files)
    total_should_be_unfrozen = 28 * 3 - total_should_be_frozen
    
    print(f"\n📊 Summary:")
    print(f"  Expected frozen layers: {total_should_be_frozen}")
    print(f"  Expected unfrozen layers: {total_should_be_unfrozen}")
    print(f"  Actual files saved: {total_saved}")
    print(f"  Expected files: {total_should_be_unfrozen}")

if __name__ == "__main__":
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
    
    for exp in experiments:
        verify_experiment(base_dir, exp)
