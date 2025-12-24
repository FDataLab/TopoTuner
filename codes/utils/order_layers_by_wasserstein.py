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
    """Generate layer orderings for Q, K, V based on Wasserstein distance."""
    
    print("Loading Wasserstein distance data...")
    df = process_all_data()
    
    print(f"Loaded {len(df)} rows")
    print(f"Models found: {sorted(df['Model'].unique())}")
    
    # Focus on Llama-3.2-3B
    model_key = "llama32_3b"
    if model_key not in MODELS:
        print(f"Error: Model {model_key} not found in MODELS")
        return
    
    model_df = df[df["Model"] == model_key]
    if model_df.empty:
        print(f"Error: No data found for model {model_key}")
        return
    
    print(f"\nProcessing {model_key}...")
    avg_data = calculate_combined_averages(model_df)
    
    if avg_data.empty:
        print(f"  No combined data after filtering")
        return
    
    print("\n" + "="*70)
    print("Layer Ordering by Wasserstein Distance (Lowest to Highest)")
    print("="*70)
    print("\nUsing WassersteinType: H0 (as in layer_analysis.txt)")
    print("MatrixType: Q, K, V\n")
    
    # Order layers for Q, K, V using H0 (as in the analysis file)
    wasserstein_type = "H0"
    
    q_ordered = order_layers_by_distance(avg_data, "q", wasserstein_type)
    k_ordered = order_layers_by_distance(avg_data, "k", wasserstein_type)
    v_ordered = order_layers_by_distance(avg_data, "v", wasserstein_type)
    
    print(f"Q layers (ordered lowest to highest distance):")
    print(f"  {q_ordered}")
    print(f"\nK layers (ordered lowest to highest distance):")
    print(f"  {k_ordered}")
    print(f"\nV layers (ordered lowest to highest distance):")
    print(f"  {v_ordered}")
    
    # Also print in Python list format for easy copy-paste
    print("\n" + "="*70)
    print("Python List Format (for shell script):")
    print("="*70)
    print(f"\nQ_ORDERED_LAYERS=({' '.join(map(str, q_ordered))})")
    print(f"K_ORDERED_LAYERS=({' '.join(map(str, k_ordered))})")
    print(f"V_ORDERED_LAYERS=({' '.join(map(str, v_ordered))})")
    
    # Verify lowest 3 match the analysis file
    print("\n" + "="*70)
    print("Verification: Lowest 3 layers (should match layer_analysis.txt):")
    print("="*70)
    print(f"Q lowest 3: {q_ordered[:3]} (expected: [0, 16, 23])")
    print(f"K lowest 3: {k_ordered[:3]} (expected: [15, 8, 0])")
    print(f"V lowest 3: {v_ordered[:3]} (expected: [25, 23, 27])")
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), "../../layer_orderings.txt")
    with open(output_file, "w") as f:
        f.write("# Layer orderings by Wasserstein distance (lowest to highest)\n")
        f.write("# Generated from Wasserstein distance analysis\n")
        f.write(f"# Model: {model_key}\n")
        f.write(f"# WassersteinType: {wasserstein_type}\n\n")
        f.write(f"Q_ORDERED_LAYERS=({' '.join(map(str, q_ordered))})\n")
        f.write(f"K_ORDERED_LAYERS=({' '.join(map(str, k_ordered))})\n")
        f.write(f"V_ORDERED_LAYERS=({' '.join(map(str, v_ordered))})\n")
    
    print(f"\n✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
