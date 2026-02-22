#!/usr/bin/env python3
"""
Create a formatted table of IMDB evaluation results for 3 experiments.
"""

import pandas as pd
from pathlib import Path

# Read CSV files
csv_dir = Path("evaluation_results/imdb")

full_df = pd.read_csv(csv_dir / "imdb_llama32_3b_full.csv")
k_o_6_df = pd.read_csv(csv_dir / "k_o_lowest6.csv")
k_o_9_df = pd.read_csv(csv_dir / "k_o_lowest9.csv")

# Extract epoch numbers
def get_epoch(checkpoint_str):
    """Extract epoch number from checkpoint string."""
    if "epoch-" in checkpoint_str:
        return int(checkpoint_str.split("epoch-")[1])
    return -1

# Process each dataframe
full_df['epoch'] = full_df['checkpoint'].apply(get_epoch)
k_o_6_df['epoch'] = k_o_6_df['checkpoint'].apply(get_epoch)
k_o_9_df['epoch'] = k_o_9_df['checkpoint'].apply(get_epoch)

# Convert accuracy to percentage if needed
def to_percentage(acc):
    """Convert accuracy to percentage."""
    if acc < 1.0:
        return acc * 100
    return acc

full_df['acc_pct'] = full_df['acc'].apply(to_percentage)
k_o_6_df['acc_pct'] = k_o_6_df['acc'].apply(to_percentage)
k_o_9_df['acc_pct'] = k_o_9_df['acc'].apply(to_percentage)

# Create merged table
all_epochs = sorted(set(full_df['epoch'].tolist() + k_o_6_df['epoch'].tolist() + k_o_9_df['epoch'].tolist()))

# Create table data
table_data = []
for epoch in all_epochs:
    full_acc = full_df[full_df['epoch'] == epoch]['acc_pct'].values
    k_o_6_acc = k_o_6_df[k_o_6_df['epoch'] == epoch]['acc_pct'].values
    k_o_9_acc = k_o_9_df[k_o_9_df['epoch'] == epoch]['acc_pct'].values
    
    row = {
        'Epoch': epoch,
        'Full': f"{full_acc[0]:.2f}%" if len(full_acc) > 0 else "N/A",
        'K+O Lowest 6': f"{k_o_6_acc[0]:.2f}%" if len(k_o_6_acc) > 0 else "N/A",
        'K+O Lowest 9': f"{k_o_9_acc[0]:.2f}%" if len(k_o_9_acc) > 0 else "N/A",
    }
    table_data.append(row)

# Create DataFrame and print
result_df = pd.DataFrame(table_data)

print("=" * 80)
print("IMDB Evaluation Results: Accuracy by Epoch")
print("=" * 80)
print()
print(result_df.to_string(index=False))
print()
print("=" * 80)

# Also save to CSV
output_file = Path("logs/imdb_evaluation_table.csv")
result_df.to_csv(output_file, index=False)
print(f"\n✅ Table saved to: {output_file}")
