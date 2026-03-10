#!/usr/bin/env python3
"""
Batch analyze weight changes for all freezing experiments.

Calculates |Wi - Wo| for each epoch, layer, and Q/K/V projection
for all experiments: full, lowest_3/6/9/12/15, highest_3/6/9/12/15.

Usage:
    python codes/utils/batch_analyze_weight_changes.py \
        --base-dir ./numpy_weights/imdb/llama32_3b \
        --output-dir ./analysis/weight_changes
"""

import os
import sys
import argparse
from pathlib import Path
from analyze_weight_changes import analyze_weight_changes


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze weight changes for all freezing experiments"
    )
    parser.add_argument(
        "--base-dir",
        default="./numpy_weights/imdb/llama32_3b",
        help="Base directory containing experiment folders"
    )
    parser.add_argument(
        "--output-dir",
        default="./analysis/weight_changes",
        help="Output directory for CSV files and logs"
    )
    parser.add_argument(
        "--is-lora",
        action="store_true",
        default=False,
        help="Whether experiments use LoRA (default: False, auto-detect from files)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of experiments to analyze
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
    
    print("=" * 80)
    print(f"Analyzing weight changes for {len(experiments)} experiments")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LoRA mode: {args.is_lora}")
    print("")
    
    results = []
    
    for exp in experiments:
        weights_dir = base_dir / exp
        output_csv = output_dir / f"{exp}_weight_changes.csv"
        log_file = output_dir / f"{exp}_analysis.log"
        
        print(f"[INFO] Processing: {exp}")
        print(f"  Weights dir: {weights_dir}")
        print(f"  Output: {output_csv}")
        
        if not weights_dir.exists():
            print(f"  ⚠️  Directory not found, skipping...")
            print("")
            results.append((exp, "SKIPPED", "Directory not found"))
            continue
        
        if not (weights_dir / "epoch_weights").exists():
            print(f"  ⚠️  epoch_weights directory not found, skipping...")
            print("")
            results.append((exp, "SKIPPED", "epoch_weights not found"))
            continue
        
        # We're analyzing base weights, not LoRA adapters
        # Run analysis
        try:
            df = analyze_weight_changes(
                str(weights_dir),
                str(output_csv),
                is_lora=False  # Always False - we only analyze base weights
            )
            
            if df is not None and len(df) > 0:
                print(f"  ✅ Analysis completed: {output_csv} ({len(df)} rows)")
                results.append((exp, "SUCCESS", f"{len(df)} rows"))
            else:
                print(f"  ⚠️  Analysis completed but no results")
                results.append((exp, "WARNING", "No results"))
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            results.append((exp, "FAILED", str(e)))
            # Write error to log file
            with open(log_file, "w") as f:
                f.write(f"Error analyzing {exp}:\n{str(e)}\n")
        
        print("")
    
    # Print summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    for exp, status, message in results:
        status_symbol = {
            "SUCCESS": "✅",
            "SKIPPED": "⚠️ ",
            "WARNING": "⚠️ ",
            "FAILED": "❌"
        }.get(status, "  ")
        print(f"{status_symbol} {exp:15s} - {status:8s} - {message}")
    
    print("")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    
    # Count successes
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    print(f"\n✅ Successfully analyzed {success_count}/{len(experiments)} experiments")


if __name__ == "__main__":
    main()
