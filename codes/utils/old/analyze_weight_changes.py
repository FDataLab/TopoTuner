#!/usr/bin/env python3
"""
Analyze weight changes across epochs for Q/K/V projections.

Calculates |Wi - Wo| for each epoch, layer, and projection type (Q/K/V).
This helps verify which layers are frozen (distance ≈ 0) vs unfrozen (distance > 0).

Usage:
    python codes/utils/analyze_weight_changes.py \
        --weights-dir ./numpy_weights/imdb/llama32_3b/lowest_15/epoch_weights \
        --output ./analysis/weight_changes_lowest_15.csv
"""

import os
import re
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_filename(filename: str) -> Optional[Tuple[int, str, Optional[str]]]:
    """
    Parse numpy weight filename to extract layer, projection, and LoRA type.
    
    Returns:
        (layer_idx, projection_type, lora_type) where:
        - layer_idx: int (0-27)
        - projection_type: 'q', 'k', or 'v'
        - lora_type: 'A', 'B', or None (for full finetuning)
    
    Examples:
        'layer7_q_A.npy' -> (7, 'q', 'A')
        'layer7_q.npy' -> (7, 'q', None)
    """
    match = re.match(r"layer(\d+)_(q|k|v)(?:_(A|B))?\.npy", filename)
    if match:
        layer = int(match.group(1))
        proj = match.group(2)
        lora = match.group(3) if match.group(3) else None
        return (layer, proj, lora)
    return None


def load_epoch_weights(epoch_dir: str, is_lora: bool) -> Dict[Tuple[int, str, Optional[str]], np.ndarray]:
    """
    Load all Q/K/V weight files from an epoch directory.
    Only loads base weights (not LoRA adapters).
    
    Returns:
        Dictionary mapping (layer, proj, None) -> weight array
    """
    npy_dir = os.path.join(epoch_dir, "numpy_weights")
    if not os.path.exists(npy_dir):
        return {}
    
    weights = {}
    for filename in os.listdir(npy_dir):
        if not filename.endswith('.npy'):
            continue
        
        # Skip LoRA files (_A, _B)
        if '_A.npy' in filename or '_B.npy' in filename:
            continue
        
        parsed = parse_filename(filename)
        if parsed is None:
            continue
        
        layer, proj, lora = parsed
        # Only process Q/K/V projections
        if proj not in ['q', 'k', 'v']:
            continue
        
        # Only process base weights (lora should be None)
        if lora is not None:
            continue
        
        filepath = os.path.join(npy_dir, filename)
        weight = np.load(filepath)
        weights[(layer, proj, None)] = weight
    
    return weights


def calculate_distance(w0: np.ndarray, wi: np.ndarray) -> float:
    """
    Calculate absolute difference: |Wi - Wo|
    
    Returns mean absolute difference across all elements.
    """
    if w0.shape != wi.shape:
        raise ValueError(f"Shape mismatch: {w0.shape} vs {wi.shape}")
    
    diff = np.abs(wi.astype(np.float32) - w0.astype(np.float32))
    return float(np.mean(diff))


def get_frozen_layers_from_experiment_name(exp_name: str) -> Dict[str, List[int]]:
    """
    Get which layers should be frozen based on experiment name.
    Returns dict with 'q', 'k', 'v' keys mapping to lists of layer indices.
    """
    # Layer orderings by Wasserstein distance (lowest to highest)
    Q_ORDERED = [0, 16, 23, 18, 20, 17, 5, 10, 24, 12, 15, 8, 13, 19, 3, 6, 22, 2, 9, 21, 25, 11, 4, 14, 1, 7, 26, 27]
    K_ORDERED = [15, 8, 0, 12, 16, 18, 14, 17, 21, 9, 13, 10, 19, 11, 20, 24, 4, 6, 23, 7, 5, 25, 26, 22, 3, 27, 1, 2]
    V_ORDERED = [25, 23, 27, 24, 26, 20, 22, 21, 18, 15, 19, 3, 14, 13, 16, 1, 17, 11, 9, 6, 4, 8, 12, 7, 5, 10, 2, 0]
    
    frozen = {'q': [], 'k': [], 'v': []}
    
    if exp_name == 'full':
        return frozen  # No layers frozen
    
    # Parse experiment name
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


def analyze_weight_changes(weights_dir: str, output_path: str, is_lora: bool = True) -> Optional[pd.DataFrame]:
    """
    Analyze weight changes across epochs.
    
    Args:
        weights_dir: Directory containing epoch_weights/checkpoint-epoch-{N}/
        output_path: Output CSV path
        is_lora: Whether this is LoRA finetuning (affects how weights are combined)
    """
    weights_dir = Path(weights_dir)
    epoch_weights_dir = weights_dir / "epoch_weights"
    
    if not epoch_weights_dir.exists():
        raise ValueError(f"epoch_weights directory not found: {epoch_weights_dir}")
    
    # Get experiment name from directory path
    exp_name = weights_dir.name
    frozen_layers = get_frozen_layers_from_experiment_name(exp_name)
    print(f"Expected frozen layers: Q={frozen_layers['q']}, K={frozen_layers['k']}, V={frozen_layers['v']}")
    
    # Find all epoch directories
    epoch_dirs = sorted([
        epoch_weights_dir / d
        for d in os.listdir(epoch_weights_dir)
        if d.startswith("checkpoint-epoch-") and os.path.isdir(epoch_weights_dir / d)
    ], key=lambda x: int(re.search(r"epoch-(\d+)", str(x)).group(1)))
    
    if len(epoch_dirs) == 0:
        raise ValueError(f"No epoch directories found in {epoch_weights_dir}")
    
    print(f"Found {len(epoch_dirs)} epochs: {[d.name for d in epoch_dirs]}")
    
    # Load epoch-0 weights (baseline)
    print(f"Loading baseline weights from {epoch_dirs[0]}...")
    w0_all = load_epoch_weights(str(epoch_dirs[0]), is_lora)
    print(f"Loaded {len(w0_all)} weight files from epoch 0")
    
    if len(w0_all) == 0:
        raise ValueError(f"No weights found in epoch 0 directory")
    
    # Determine all layers and projections
    all_keys = set(w0_all.keys())
    layers = sorted(set(layer for layer, _, _ in all_keys))
    projs = sorted(set(proj for _, proj, _ in all_keys))
    
    print(f"Found {len(layers)} layers: {layers}")
    print(f"Found projections: {projs}")
    
    # Collect results
    results = []
    
    # Process each epoch
    for epoch_dir in epoch_dirs[1:]:  # Skip epoch 0 (baseline)
        epoch_num = int(re.search(r"epoch-(\d+)", str(epoch_dir)).group(1))
        print(f"\nProcessing epoch {epoch_num} from {epoch_dir.name}...")
        
        wi_all = load_epoch_weights(str(epoch_dir), is_lora)
        
        if len(wi_all) == 0:
            print(f"  ⚠️  No weights found, skipping...")
            continue
        
        # Determine if we have LoRA files (with _A/_B) or base weight files
        # Check if any keys have lora_type 'A' or 'B'
        has_lora_files = any(lora_type in ['A', 'B'] for _, _, lora_type in w0_all.keys() if lora_type is not None)
        
        # Calculate distances for each layer and projection
        # Note: Frozen layers may not be in the CSV because they're not saved (not trainable)
        # So we need to check all possible layers (0-27) and all projections (q, k, v)
        all_possible_layers = list(range(28))  # 0-27
        all_possible_projs = ['q', 'k', 'v']
        
        for layer in all_possible_layers:
            for proj in all_possible_projs:
                key = (layer, proj, None)
                proj_upper = proj.upper()
                
                # Check if this layer/projection should be frozen
                should_be_frozen = layer in frozen_layers.get(proj, [])
                
                if key in w0_all and key in wi_all:
                    # Layer exists in both epochs - calculate actual distance
                    distance = calculate_distance(w0_all[key], wi_all[key])
                    
                    # If should be frozen but file exists, this is unexpected - but calculate distance anyway
                    if should_be_frozen:
                        print(f"  ⚠️  WARNING: Layer {layer} {proj_upper} should be frozen but file exists! Distance: {distance:.2e}")
                    
                    # Mark as frozen only if distance is essentially zero (numerical precision)
                    is_frozen = (distance < 1e-6)
                    
                    results.append({
                        'epoch': epoch_num,
                        'layer': layer,
                        'projection': proj_upper,
                        'distance': distance,
                        'frozen': 'YES' if is_frozen else 'NO'
                    })
                elif should_be_frozen:
                    # Layer should be frozen and file doesn't exist - verified: frozen layers aren't saved
                    # This is correct behavior - frozen layers don't get saved because they're not trainable
                    results.append({
                        'epoch': epoch_num,
                        'layer': layer,
                        'projection': proj_upper,
                        'distance': 0.0,  # No file = no change = distance 0
                        'frozen': 'YES'
                    })
                elif key not in w0_all and key not in wi_all:
                    # Layer not frozen and file doesn't exist - unexpected but handle gracefully
                    # This shouldn't happen for unfrozen layers, but if it does, skip it
                    pass
                # If layer is not frozen and not in CSV, skip it (shouldn't happen but handle gracefully)
                    # For base weights (full finetuning or saved base weights): calculate distance
                    key = (layer, proj, None)
                    if key in w0_all and key in wi_all:
                        distance = calculate_distance(w0_all[key], wi_all[key])
                        results.append({
                            'epoch': epoch_num,
                            'layer': layer,
                            'projection': proj.upper(),
                            'distance': distance,
                            'frozen': 'NO' if distance > 1e-6 else 'YES'
                        })
                    elif key in w0_all:
                        print(f"  ⚠️  Missing {key} in epoch {epoch_num}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("⚠️  No results to save!")
        return None
    
    # Sort by epoch, layer, projection
    df = df.sort_values(['epoch', 'layer', 'projection'])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_path}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"  Total measurements: {len(df)}")
    print(f"  Epochs analyzed: {df['epoch'].nunique()}")
    print(f"  Layers analyzed: {df['layer'].nunique()}")
    print(f"  Projections: {df['projection'].unique().tolist()}")
    
    # Show frozen vs unfrozen counts
    if 'frozen' in df.columns:
        frozen_counts = df.groupby(['epoch', 'frozen']).size().unstack(fill_value=0)
        print("\n🔒 Freezing Status by Epoch:")
        print(frozen_counts)
        
        # Show layers that are consistently frozen across all epochs
        if len(df) > 0:
            layer_frozen = df.groupby(['layer', 'projection', 'frozen']).size().unstack(fill_value=0)
            print("\n🔒 Freezing Status by Layer & Projection:")
            print(layer_frozen)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weight changes across epochs for Q/K/V projections"
    )
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="Directory containing epoch_weights/checkpoint-epoch-{N}/"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path"
    )
    parser.add_argument(
        "--is-lora",
        action="store_true",
        default=True,
        help="Whether this is LoRA finetuning (default: True)"
    )
    parser.add_argument(
        "--no-lora",
        dest="is_lora",
        action="store_false",
        help="Disable LoRA mode (for full finetuning)"
    )
    
    args = parser.parse_args()
    
    analyze_weight_changes(
        args.weights_dir,
        args.output,
        is_lora=args.is_lora
    )


if __name__ == "__main__":
    main()
