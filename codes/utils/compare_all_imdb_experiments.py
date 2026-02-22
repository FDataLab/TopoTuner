#!/usr/bin/env python3
"""
Compare checkpoints across all IMDB experiments and generate a comprehensive log.
Epoch-by-epoch and layer-by-layer analysis.
"""

import subprocess
import sys
from pathlib import Path

# Define the 3 experiments we want to analyze
base_dir = Path("numpy_weights/imdb/llama32_3b")
experiments = [
    "full",
    "k_o_lowest6",
    "k_o_lowest9",
]

# Filter to only include experiments that have epoch checkpoints
valid_experiments = []
for exp in experiments:
    epoch6_dir = base_dir / exp / "epoch_weights" / "checkpoint-epoch-6"
    if epoch6_dir.exists():
        valid_experiments.append(exp)
    else:
        print(f"⚠️  Skipping {exp} - no checkpoint-epoch-6 found")

experiments = valid_experiments
print(f"\nFound {len(experiments)} experiments with full epoch checkpoints:")
for exp in experiments:
    print(f"  ✅ {exp}")

log_file = Path('logs/imdb_checkpoint_comparison_all_experiments.log')

def run_command(cmd_args):
    """Run a command and return stdout and stderr."""
    result = subprocess.run(
        [sys.executable, 'codes/utils/compare_checkpoints.py'] + cmd_args,
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )
    return result.stdout, result.stderr

with open(log_file, 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('CHECKPOINT COMPARISON: All IMDB Experiments\n')
    f.write('Epoch-by-epoch and layer-by-layer analysis\n')
    f.write(f'Total experiments: {len(experiments)}\n')
    f.write('=' * 80 + '\n\n')
    
    for exp in experiments:
        f.write('\n' + '=' * 80 + '\n')
        f.write(f'EXPERIMENT: {exp}\n')
        f.write('=' * 80 + '\n\n')
        
        # Check which epochs exist
        exp_dir = base_dir / exp / "epoch_weights"
        epochs = []
        for epoch_dir in sorted(exp_dir.glob("checkpoint-epoch-*")):
            epoch_num = int(epoch_dir.name.split("-")[-1])
            epochs.append(epoch_num)
        
        if not epochs:
            f.write(f"  ⚠️  No epoch checkpoints found\n\n")
            continue
        
        epochs = sorted(epochs)
        f.write(f"Available epochs: {epochs}\n\n")
        
        # Compare consecutive epochs
        for i in range(len(epochs) - 1):
            epoch1 = epochs[i]
            epoch2 = epochs[i + 1]
            
            f.write(f'--- Epoch {epoch1} → {epoch2} ---\n')
            
            # Layer-wise comparison
            stdout, stderr = run_command([
                '--base-dir', str(base_dir),  # numpy_weights/imdb/llama32_3b
                '--exp', exp,
                '--epoch1', str(epoch1),
                '--epoch2', str(epoch2),
                '--layerwise',
                '--topk', '30'
            ])
            f.write(stdout)
            if stderr:
                f.write(f'STDERR: {stderr}\n')
            
            # Summary statistics
            stdout, stderr = run_command([
                '--base-dir', str(base_dir),
                '--exp', exp,
                '--epoch1', str(epoch1),
                '--epoch2', str(epoch2)
            ])
            f.write(stdout)
            if stderr:
                f.write(f'STDERR: {stderr}\n')
            
            f.write('\n')
        
        f.write('=' * 80 + '\n\n')

print(f'\n✅ Log saved to: {log_file}')
print(f'📄 View with: cat {log_file}')
