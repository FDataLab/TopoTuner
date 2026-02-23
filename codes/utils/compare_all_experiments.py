#!/usr/bin/env python3
"""
Compare checkpoints across all K+O experiments and generate a comprehensive log.
"""

import subprocess
import sys
from pathlib import Path

experiments = ['k_o_lowest3', 'k_o_highest3', 'k_o_lowest15', 'k_o_highest15']
log_file = Path('logs/checkpoint_comparison_all_experiments.log')

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
    f.write('CHECKPOINT COMPARISON: All K+O Experiments\n')
    f.write('Checking for convergence (identical weights after epoch 3)\n')
    f.write('=' * 80 + '\n\n')
    
    for exp in experiments:
        f.write('\n' + '=' * 80 + '\n')
        f.write(f'EXPERIMENT: {exp}\n')
        f.write('=' * 80 + '\n\n')
        
        # Epoch 2→3 (should show changes)
        f.write('--- Epoch 2 → 3 (Before Convergence) ---\n')
        stdout, stderr = run_command(['--exp', exp, '--epoch1', '2', '--epoch2', '3', 
                                     '--layerwise', '--topk', '30'])
        f.write(stdout)
        if stderr:
            f.write(f'STDERR: {stderr}\n')
        f.write('\n')
        
        # Epoch 3→4 (should show convergence)
        f.write('--- Epoch 3 → 4 (Convergence Check) ---\n')
        stdout, stderr = run_command(['--exp', exp, '--epoch1', '3', '--epoch2', '4', 
                                     '--layerwise', '--topk', '30'])
        f.write(stdout)
        if stderr:
            f.write(f'STDERR: {stderr}\n')
        f.write('\n')
        
        # Summary statistics
        f.write('--- Summary Statistics (Epoch 3 → 4) ---\n')
        stdout, stderr = run_command(['--exp', exp, '--epoch1', '3', '--epoch2', '4'])
        f.write(stdout)
        if stderr:
            f.write(f'STDERR: {stderr}\n')
        f.write('\n' + '=' * 80 + '\n\n')

print(f'✅ Log saved to: {log_file}')
print(f'📄 View with: cat {log_file}')
