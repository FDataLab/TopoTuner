#!/usr/bin/env python3
"""
Compare numpy weight checkpoints between epochs.
Adapted from user's helper code to work with .npy files in directories.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_npy_directory(directory: Path) -> Dict[str, np.ndarray]:
    """
    Load all .npy files from a directory into a dictionary.
    
    Args:
        directory: Path to directory containing .npy files
    
    Returns:
        Dictionary mapping filename (without .npy) to numpy array
    """
    weights = {}
    if not directory.exists():
        return weights
    
    for npy_file in sorted(directory.glob("*.npy")):
        key = npy_file.stem  # filename without .npy extension
        weights[key] = np.load(npy_file)
    
    return weights


def compare_checkpoints(
    p1: Path, 
    p2: Path, 
    atol: float = 0.0,
    verbose: bool = False
) -> Dict:
    """
    Compare two checkpoint directories containing .npy files.
    
    Args:
        p1: Path to first checkpoint directory
        p2: Path to second checkpoint directory
        atol: Absolute tolerance for considering values as changed
        verbose: If True, print detailed per-file differences
    
    Returns:
        Dictionary with comparison statistics
    """
    w1 = load_npy_directory(p1)
    w2 = load_npy_directory(p2)
    
    if not w1:
        return {"error": f"No weights found in {p1}"}
    if not w2:
        return {"error": f"No weights found in {p2}"}
    
    # Find common keys
    common_keys = set(w1.keys()) & set(w2.keys())
    only_p1 = set(w1.keys()) - set(w2.keys())
    only_p2 = set(w2.keys()) - set(w1.keys())
    
    if only_p1 or only_p2:
        if verbose:
            print(f"Warning: Key mismatch")
            if only_p1:
                print(f"  Only in {p1}: {sorted(only_p1)}")
            if only_p2:
                print(f"  Only in {p2}: {sorted(only_p2)}")
    
    if not common_keys:
        return {"error": "No common parameter keys found"}
    
    total_elems = 0
    changed_elems = 0
    max_diff = 0.0
    mean_diffs = []
    file_stats = []
    
    for k in sorted(common_keys):
        a = w1[k]
        b = w2[k]
        
        if a.shape != b.shape:
            if verbose:
                print(f"Shape mismatch for {k}: {a.shape} vs {b.shape}")
            continue
        
        diff = np.abs(a - b)
        
        file_total = diff.size
        file_changed = np.count_nonzero(diff > atol)
        file_max = diff.max()
        file_mean = diff.mean()
        
        total_elems += file_total
        changed_elems += file_changed
        max_diff = max(max_diff, file_max)
        mean_diffs.append(file_mean)
        
        if verbose:
            file_stats.append({
                "key": k,
                "shape": a.shape,
                "max_diff": file_max,
                "mean_diff": file_mean,
                "pct_changed": 100 * file_changed / file_total if file_total > 0 else 0
            })
    
    result = {
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(np.mean(mean_diffs)) if mean_diffs else 0.0,
        "pct_changed": 100 * changed_elems / total_elems if total_elems > 0 else 0.0,
        "total_elements": total_elems,
        "changed_elements": changed_elems,
        "num_files": len(common_keys),
        "only_in_p1": len(only_p1),
        "only_in_p2": len(only_p2)
    }
    
    if verbose and file_stats:
        result["file_details"] = file_stats
    
    return result


def layerwise_diff(
    p1: Path,
    p2: Path,
    topk: int = 30
) -> None:
    """
    Show layer-wise differences sorted by mean absolute difference.
    
    Args:
        p1: Path to first checkpoint directory
        p2: Path to second checkpoint directory
        topk: Number of top changed layers to display
    """
    w1 = load_npy_directory(p1)
    w2 = load_npy_directory(p2)
    
    if not w1:
        print(f"Error: No weights found in {p1}")
        return
    if not w2:
        print(f"Error: No weights found in {p2}")
        return
    
    # Find common keys
    common_keys = set(w1.keys()) & set(w2.keys())
    
    if not common_keys:
        print("Error: No common parameter keys found")
        return
    
    diffs = []
    for k in common_keys:
        a = w1[k]
        b = w2[k]
        
        if a.shape != b.shape:
            continue
        
        diff = np.abs(a - b).mean()
        diffs.append((k, diff))
    
    # Sort by difference (highest first)
    diffs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top changed layers:")
    for k, d in diffs[:topk]:
        print(f"{k:70s} {d:.3e}")
    
    print("\nFrozen layers (exact zero diff):")
    frozen = [k for k, d in diffs if d == 0.0]
    print(f"count={len(frozen)}")
    for k in frozen[:topk]:
        print(k)


def compare_epochs(
    base_dir: Path,
    exp_name: str,
    epoch1: int,
    epoch2: int,
    atol: float = 0.0,
    verbose: bool = False
) -> Dict:
    """
    Compare two epochs for a specific experiment.
    
    Args:
        base_dir: Base directory (e.g., numpy_weights/hotpotqa/llama32_3b)
        exp_name: Experiment name (e.g., k_o_lowest3)
        epoch1: First epoch number
        epoch2: Second epoch number
        atol: Absolute tolerance
        verbose: Print detailed stats
    
    Returns:
        Comparison statistics dictionary
    """
    ckpt1_dir = base_dir / exp_name / "epoch_weights" / f"checkpoint-epoch-{epoch1}" / "numpy_weights"
    ckpt2_dir = base_dir / exp_name / "epoch_weights" / f"checkpoint-epoch-{epoch2}" / "numpy_weights"
    
    if not ckpt1_dir.exists():
        return {"error": f"Checkpoint {epoch1} not found: {ckpt1_dir}"}
    if not ckpt2_dir.exists():
        return {"error": f"Checkpoint {epoch2} not found: {ckpt2_dir}"}
    
    result = compare_checkpoints(ckpt1_dir, ckpt2_dir, atol=atol, verbose=verbose)
    result["epoch1"] = epoch1
    result["epoch2"] = epoch2
    result["exp_name"] = exp_name
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare numpy weight checkpoints")
    parser.add_argument("--base-dir", type=str, 
                       default="numpy_weights/hotpotqa/llama32_3b",
                       help="Base directory for experiments")
    parser.add_argument("--exp", type=str, required=True,
                       help="Experiment name (e.g., k_o_lowest3)")
    parser.add_argument("--epoch1", type=int, required=True,
                       help="First epoch number")
    parser.add_argument("--epoch2", type=int, required=True,
                       help="Second epoch number")
    parser.add_argument("--atol", type=float, default=0.0,
                       help="Absolute tolerance for considering values changed")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed per-file statistics")
    parser.add_argument("--layerwise", action="store_true",
                       help="Show layer-wise differences sorted by change amount")
    parser.add_argument("--topk", type=int, default=30,
                       help="Number of top changed layers to show (for --layerwise)")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    ckpt1_dir = base_path / args.exp / "epoch_weights" / f"checkpoint-epoch-{args.epoch1}" / "numpy_weights"
    ckpt2_dir = base_path / args.exp / "epoch_weights" / f"checkpoint-epoch-{args.epoch2}" / "numpy_weights"
    
    if args.layerwise:
        print(f"\n=== Layer-wise Diff: {args.exp} (Epoch {args.epoch1} → {args.epoch2}) ===\n")
        layerwise_diff(ckpt1_dir, ckpt2_dir, topk=args.topk)
    else:
        result = compare_epochs(
            base_path, 
            args.exp, 
            args.epoch1, 
            args.epoch2,
            atol=args.atol,
            verbose=args.verbose
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n=== Comparison: {args.exp} (Epoch {args.epoch1} → {args.epoch2}) ===")
            print(f"Max absolute difference: {result['max_abs_diff']:.9f}")
            print(f"Mean absolute difference: {result['mean_abs_diff']:.9f}")
            print(f"Percentage changed: {result['pct_changed']:.6f}%")
            print(f"Total elements: {result['total_elements']:,}")
            print(f"Changed elements: {result['changed_elements']:,}")
            print(f"Number of files: {result['num_files']}")
            
            if result['max_abs_diff'] < 1e-6:
                print("\n⚠️  WARNING: Weights are IDENTICAL (converged)")
            elif result['pct_changed'] < 0.01:
                print("\n⚠️  WARNING: Very few elements changed (likely converged)")
            
            if args.verbose and "file_details" in result:
                print("\n--- Per-file details ---")
                for fstat in result["file_details"][:10]:  # Show first 10
                    print(f"{fstat['key']:20s} | max_diff={fstat['max_diff']:.9f} | "
                          f"mean_diff={fstat['mean_diff']:.9f} | "
                          f"pct_changed={fstat['pct_changed']:.4f}%")
                if len(result["file_details"]) > 10:
                    print(f"... and {len(result['file_details']) - 10} more files")

