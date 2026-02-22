#!/usr/bin/env python3
"""
Order layers by Wasserstein distance for Q, K, V projections.

This script loads Wasserstein distance data and orders all layers
from lowest to highest average distance for each projection type (Q, K, V).

The "lowest" layers have the smallest Wasserstein distance (least change),
and "highest" layers have the largest Wasserstein distance (most change).
"""

import os
import sys
import pandas as pd

# Add parent directory to path to import from codes.tda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.tda.generate_wasserstein_boxplots_plotly_prior import (
    process_all_data,
    calculate_combined_averages,
    MODELS,
    DATASETS,
    MATRIX_TYPES,
    WASSERSTEIN_TYPES,
    COMBINED_TYPES,
    EPOCHS,
)


def order_layers_by_distance(combo_data: pd.DataFrame, matrix_type: str, wasserstein_type: str):
    """
    Order all layers by average Wasserstein distance (lowest to highest).
    
    Args:
        combo_data: DataFrame with Wasserstein distance data
        matrix_type: 'q', 'k', or 'v' (lowercase)
        wasserstein_type: 'H0' or 'H1'
    
    Returns:
        List of layer indices ordered from lowest to highest average distance
    """
    # Filter to specific matrix type and Wasserstein type
    filtered = combo_data[
        (combo_data["MatrixType"] == matrix_type.lower()) &
        (combo_data["WassersteinType"] == wasserstein_type)
    ].copy()
    
    if filtered.empty:
        print(f"  Warning: No data for {matrix_type.upper()}-{wasserstein_type}")
        return []
    
    # Group by layer and compute mean distance across all datasets
    layer_mean = filtered.groupby("Layer")["AvgDistance"].mean().sort_values()
    
    # Return all layers ordered from lowest to highest distance
    return layer_mean.index.astype(int).tolist()


def main():
    """Generate layer orderings for Q, K, V based on Wasserstein distance for all models."""
    
    print("Loading Wasserstein distance data...")
    df = process_all_data()
    
    print(f"Loaded {len(df)} rows")
    print(f"Models found: {sorted(df['Model'].unique())}")
    
    # Process all models
    wasserstein_type = "H0"  # Using H0 as in layer_analysis.txt
    
    # Store results for all models
    all_results = {}
    
    print("\n" + "="*70)
    print("Layer Ordering by Wasserstein Distance (Lowest to Highest)")
    print("="*70)
    print(f"\nUsing WassersteinType: {wasserstein_type}")
    print("MatrixType: Q, K, V\n")
    
    # Process each model
    for model_key in sorted(MODELS.keys()):
        if model_key not in df['Model'].unique():
            print(f"⚠️  Warning: Model {model_key} not found in data, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_key}")
        print(f"{'='*70}")
        
        model_df = df[df["Model"] == model_key]
        if model_df.empty:
            print(f"  ⚠️  No data found for model {model_key}, skipping...")
            continue
        
        avg_data = calculate_combined_averages(model_df)
        
        if avg_data.empty:
            print(f"  ⚠️  No combined data after filtering for {model_key}, skipping...")
            continue
        
        # Order layers for Q, K, V
    q_ordered = order_layers_by_distance(avg_data, "q", wasserstein_type)
    k_ordered = order_layers_by_distance(avg_data, "k", wasserstein_type)
    v_ordered = order_layers_by_distance(avg_data, "v", wasserstein_type)
    
        if not q_ordered or not k_ordered or not v_ordered:
            print(f"  ⚠️  Incomplete data for {model_key}, skipping...")
            continue
        
        # Store results
        all_results[model_key] = {
            'q': q_ordered,
            'k': k_ordered,
            'v': v_ordered
        }
        
        print(f"\nQ layers (ordered lowest to highest distance):")
    print(f"  {q_ordered}")
    print(f"\nK layers (ordered lowest to highest distance):")
    print(f"  {k_ordered}")
    print(f"\nV layers (ordered lowest to highest distance):")
    print(f"  {v_ordered}")
    
        # Print shell script format
        print(f"\n--- Shell Script Format for {model_key} ---")
        print(f"Q_ORDERED_LAYERS=({' '.join(map(str, q_ordered))})")
    print(f"K_ORDERED_LAYERS=({' '.join(map(str, k_ordered))})")
    print(f"V_ORDERED_LAYERS=({' '.join(map(str, v_ordered))})")
    
    # Save all results to file
    output_file = os.path.join(os.path.dirname(__file__), "../../layer_orderings.txt")
    with open(output_file, "w") as f:
        f.write("# Layer orderings by Wasserstein distance (lowest to highest)\n")
        f.write("# Generated from Wasserstein distance analysis\n")
        f.write(f"# WassersteinType: {wasserstein_type}\n")
        f.write("# Format: Layers are ordered from lowest to highest Wasserstein distance\n")
        f.write("# Usage: Copy the arrays for your model into your shell script\n\n")
        
        for model_key in sorted(all_results.keys()):
            results = all_results[model_key]
            f.write(f"# ============================================================\n")
        f.write(f"# Model: {model_key}\n")
            f.write(f"# ============================================================\n\n")
            
            f.write(f"# {model_key} - Q projection layers (ordered lowest to highest)\n")
            f.write(f"Q_ORDERED_LAYERS_{model_key.upper().replace('-', '_')}=({' '.join(map(str, results['q']))})\n\n")
            
            f.write(f"# {model_key} - K projection layers (ordered lowest to highest)\n")
            f.write(f"K_ORDERED_LAYERS_{model_key.upper().replace('-', '_')}=({' '.join(map(str, results['k']))})\n\n")
            
            f.write(f"# {model_key} - V projection layers (ordered lowest to highest)\n")
            f.write(f"V_ORDERED_LAYERS_{model_key.upper().replace('-', '_')}=({' '.join(map(str, results['v']))})\n\n")
            
            # Also write in shell script format (without model suffix for easier copy-paste)
            f.write(f"# Shell script format (for direct copy-paste):\n")
            f.write(f"# Q_ORDERED_LAYERS=({' '.join(map(str, results['q']))})\n")
            f.write(f"# K_ORDERED_LAYERS=({' '.join(map(str, results['k']))})\n")
            f.write(f"# V_ORDERED_LAYERS=({' '.join(map(str, results['v']))})\n\n")
    
    print(f"\n{'='*70}")
    print(f"✅ Saved all layer orderings to: {output_file}")
    print(f"{'='*70}")
    print(f"\nProcessed {len(all_results)} model(s): {', '.join(sorted(all_results.keys()))}")


if __name__ == "__main__":
    main()
