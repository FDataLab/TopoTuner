#!/usr/bin/env python3
"""
Generate boxplots with line plots overlaid for Wasserstein distances using Plotly.
For each model and configuration (Full, LoRA, Combined), creates plots with 6 subplots
(one for each combination of matrix type: q/k/v and Wasserstein type: H0/H1).
Matches the style of existing plotly scripts in codes/tda/.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import os
import re
from pathlib import Path
import glob

# Configuration
DATASETS = ['imdb', 'sst2', 'mmlu']
MODELS = {
    'llama31_8b': 'llama31_8b',
    'llama32_3b': 'llama32_3b',
    'mistral7b_v03': 'mistral7b_v03',
    'qwen_8b_base': 'qwen_8b_base'
}
MATRIX_TYPES = ['q', 'k', 'v']
WASSERSTEIN_TYPES = ['H0', 'H1']
EPOCHS = list(range(7))  # 0-6
CONFIGURATIONS = {
    'Full': 'Baseline vs Full Finetuned',
    'LoRA': 'Baseline vs LoRA-final',
    'Combined': None  # Will combine both Full and LoRA
}

RESULTS_DIR = 'wasserstein_results'
OUTPUT_DIR = 'wasserstein_boxplots_log'  # Log scale version
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset colors matching the style
DATASET_COLORS = {
    'imdb': '#1f77b4',  # blue
    'sst2': '#2ca02c',  # green
    'mmlu': '#d62728',  # red
    'gsm8k': '#ff7f0e'  # orange
}

DATASET_MARKERS = {
    'imdb': 'circle',
    'sst2': 'square',
    'mmlu': 'triangle-up',
    'gsm8k': 'diamond'
}

# Models that have gsm8k data
GSM8K_MODELS = ['llama31_8b', 'llama32_3b']


def extract_layer_and_matrix(filename):
    """Extract layer number and matrix type from filename."""
    match = re.match(r'layer(\d+)_([qkv])\.pkl', filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def load_and_process_data(dataset, model_key, model_pattern):
    """Load CSV file and process data."""
    csv_path = f"{RESULTS_DIR}/wasserstein_{dataset}_{model_pattern}.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Extract layer and matrix type
    df[['Layer', 'MatrixType']] = df['File'].apply(
        lambda x: pd.Series(extract_layer_and_matrix(x))
    )
    
    # Filter out invalid rows
    df = df.dropna(subset=['Layer', 'MatrixType'])
    
    # Add dataset column
    df['Dataset'] = dataset
    
    return df


def process_all_data():
    """Load and process all CSV files."""
    all_data = []
    
    # Process standard datasets for all models
    for dataset in DATASETS:
        for model_key, model_pattern in MODELS.items():
            df = load_and_process_data(dataset, model_key, model_pattern)
            if df is not None:
                df['Model'] = model_key
                all_data.append(df)
    
    # Process gsm8k only for specific models
    for model_key in GSM8K_MODELS:
        model_pattern = MODELS[model_key]
        df = load_and_process_data('gsm8k', model_key, model_pattern)
        if df is not None:
            df['Model'] = model_key
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def calculate_averages(df, config_type, model_key=None):
    """Calculate averages across epochs 0-6 for each layer, dataset, matrix type, and Wasserstein type."""
    # Filter by configuration type
    if config_type == 'Full':
        df_filtered = df[df['Type'] == CONFIGURATIONS['Full']].copy()
    elif config_type == 'LoRA':
        df_filtered = df[df['Type'] == CONFIGURATIONS['LoRA']].copy()
    elif config_type == 'Combined':
        # Combine both Full and LoRA
        df_filtered = df[df['Type'].isin([CONFIGURATIONS['Full'], CONFIGURATIONS['LoRA']])].copy()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Filter epochs 0-6
    df_filtered = df_filtered[df_filtered['Epoch'].isin(EPOCHS)].copy()
    
    # Get datasets for this model
    datasets_to_process = get_datasets_for_model(model_key) if model_key else DATASETS
    
    # Calculate averages for H0 and H1 separately
    results = []
    
    for wasserstein_type in WASSERSTEIN_TYPES:
        col_name = f'Wasserstein {wasserstein_type}'
        
        for matrix_type in MATRIX_TYPES:
            for dataset in datasets_to_process:
                for layer in sorted(df_filtered['Layer'].unique()):
                    # Filter data
                    mask = (
                        (df_filtered['MatrixType'] == matrix_type) &
                        (df_filtered['Dataset'] == dataset) &
                        (df_filtered['Layer'] == layer)
                    )
                    
                    layer_data = df_filtered[mask]
                    
                    if len(layer_data) > 0:
                        # Average across epochs
                        avg_distance = layer_data[col_name].mean()
                        
                        results.append({
                            'Layer': layer,
                            'MatrixType': matrix_type,
                            'WassersteinType': wasserstein_type,
                            'Dataset': dataset,
                            'AvgDistance': avg_distance
                        })
    
    return pd.DataFrame(results)


def get_datasets_for_model(model_key):
    """Get list of datasets for a specific model."""
    datasets = list(DATASETS)  # Always include standard datasets
    if model_key in GSM8K_MODELS:
        datasets.append('gsm8k')
    return datasets


def find_lowest_layers_per_dataset(combo_data, n_layers=5, model_key=None):
    """Find the n layers with the lowest average Wasserstein distance for each dataset."""
    lowest_by_dataset = {}
    datasets_to_check = get_datasets_for_model(model_key) if model_key else DATASETS + ['gsm8k']
    
    for dataset in datasets_to_check:
        dataset_data = combo_data[combo_data['Dataset'] == dataset]
        if len(dataset_data) > 0:
            layer_avg = dataset_data.groupby('Layer')['AvgDistance'].mean().sort_values()
            lowest_by_dataset[dataset] = set(layer_avg.head(n_layers).index.tolist())
        else:
            lowest_by_dataset[dataset] = set()
    return lowest_by_dataset


def find_consistently_low_layers(lowest_by_dataset, num_datasets):
    """Find layers that are in the lowest set for all datasets (or most datasets)."""
    all_layers = set()
    for layers in lowest_by_dataset.values():
        all_layers.update(layers)
    
    consistently_low = {}
    for layer in all_layers:
        count = sum(1 for layers in lowest_by_dataset.values() if layer in layers)
        consistently_low[layer] = count
    
    # Layers that appear in all datasets
    all_datasets_low = {layer: count for layer, count in consistently_low.items() if count == num_datasets}
    # Layers that appear in at least 2 datasets (or at least half if more than 3 datasets)
    min_datasets = max(2, num_datasets // 2) if num_datasets > 3 else 2
    most_datasets_low = {layer: count for layer, count in consistently_low.items() if count >= min_datasets}
    
    return all_datasets_low, most_datasets_low, consistently_low


def create_boxplot_with_lines_plotly(model_key, config_type, avg_data, analysis_file=None, global_ranges=None):
    """Create boxplot with line plots overlaid using Plotly."""
    # Initialize analysis results storage
    create_boxplot_with_lines_plotly.analysis_results = []
    
    # Create subplots: H0 on top row, H1 on bottom row
    # Top row: Q-H0, K-H0, V-H0
    # Bottom row: Q-H1, K-H1, V-H1
    subplot_titles = []
    for wasserstein_type in WASSERSTEIN_TYPES:
        for matrix_type in MATRIX_TYPES:
            subplot_titles.append(f'{matrix_type.upper()}-{wasserstein_type}')
    
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    plot_idx = 0
    for wasserstein_type in WASSERSTEIN_TYPES:
        for matrix_type in MATRIX_TYPES:
            # H0 on top row (row 1), H1 on bottom row (row 2)
            row = 1 if wasserstein_type == 'H0' else 2
            # Q=col 1, K=col 2, V=col 3
            col = MATRIX_TYPES.index(matrix_type) + 1
            
            # Filter data for this combination
            combo_data = avg_data[
                (avg_data['MatrixType'] == matrix_type) &
                (avg_data['WassersteinType'] == wasserstein_type)
            ].copy()
            
            if len(combo_data) == 0:
                plot_idx += 1
                continue
            
            # Prepare data for boxplot: for each layer, get distribution across datasets
            layers = sorted(combo_data['Layer'].unique())
            
            # Find lowest layers per dataset and consistently low layers
            lowest_by_dataset = find_lowest_layers_per_dataset(combo_data, n_layers=5, model_key=model_key)
            num_datasets = len([d for d in lowest_by_dataset.values() if len(d) > 0])
            all_datasets_low, most_datasets_low, consistently_low = find_consistently_low_layers(lowest_by_dataset, num_datasets)
            
            # Store analysis for printing later (we'll print all at once per model/config)
            if not hasattr(create_boxplot_with_lines_plotly, 'analysis_results'):
                create_boxplot_with_lines_plotly.analysis_results = []
            
            create_boxplot_with_lines_plotly.analysis_results.append({
                'matrix': matrix_type.upper(),
                'wasserstein': wasserstein_type,
                'lowest_by_dataset': lowest_by_dataset,
                'all_datasets_low': all_datasets_low,
                'most_datasets_low': most_datasets_low
            })
            
            # Create boxplot data: for each layer, collect values from all datasets
            # We'll create separate traces for normal, lowest, and consistently low layers
            box_x_normal = []
            box_y_normal = []
            box_x_lowest = []
            box_y_lowest = []
            box_x_consistent = []
            box_y_consistent = []
            
            for layer in layers:
                layer_data = combo_data[combo_data['Layer'] == layer]
                distances = layer_data['AvgDistance'].values
                if len(distances) > 0:
                    # Prioritize: consistently low > lowest > normal
                    if layer in all_datasets_low:
                        box_x_consistent.extend([layer] * len(distances))
                        box_y_consistent.extend(distances)
                    elif layer in most_datasets_low:
                        box_x_consistent.extend([layer] * len(distances))
                        box_y_consistent.extend(distances)
                    else:
                        # Check if it's in any dataset's lowest
                        in_any_lowest = any(layer in layers for layers in lowest_by_dataset.values())
                        if in_any_lowest:
                            box_x_lowest.extend([layer] * len(distances))
                            box_y_lowest.extend(distances)
                        else:
                            box_x_normal.extend([layer] * len(distances))
                            box_y_normal.extend(distances)
            
            # Add boxplot trace for normal layers
            if box_x_normal:
                fig.add_trace(
                    go.Box(
                        x=box_x_normal,
                        y=box_y_normal,
                        name='Distribution',
                        boxmean='sd',
                        showlegend=False,
                        marker_color='lightblue',
                        line=dict(color='rgba(0,0,0,0.5)', width=1),
                        fillcolor='rgba(173, 216, 230, 0.3)',
                    ),
                    row=row,
                    col=col
                )
            
            # Add boxplot trace for consistently low layers (highlighted most)
            if box_x_consistent:
                fig.add_trace(
                    go.Box(
                        x=box_x_consistent,
                        y=box_y_consistent,
                        name='Consistently Low',
                        boxmean='sd',
                        showlegend=(plot_idx == 0),
                        marker_color='red',
                        line=dict(color='rgba(255,0,0,0.8)', width=2),
                        fillcolor='rgba(255, 0, 0, 0.4)',
                    ),
                    row=row,
                    col=col
                )
            
            # Add boxplot trace for lowest layers (highlighted)
            if box_x_lowest:
                fig.add_trace(
                    go.Box(
                        x=box_x_lowest,
                        y=box_y_lowest,
                        name='Lowest (Some Datasets)',
                        boxmean='sd',
                        showlegend=(plot_idx == 0),
                        marker_color='orange',
                        line=dict(color='rgba(255,140,0,0.8)', width=2),
                        fillcolor='rgba(255, 165, 0, 0.5)',
                    ),
                    row=row,
                    col=col
                )
            
            # Add line plots for each dataset
            datasets_to_plot = get_datasets_for_model(model_key)
            for dataset in datasets_to_plot:
                dataset_data = combo_data[combo_data['Dataset'] == dataset]
                if len(dataset_data) > 0:
                    dataset_data = dataset_data.sort_values('Layer')
                    fig.add_trace(
                        go.Scatter(
                            x=dataset_data['Layer'],
                            y=dataset_data['AvgDistance'],
                            mode='lines+markers',
                            name=dataset.upper(),
                            line=dict(
                                color=DATASET_COLORS[dataset],
                                width=2
                            ),
                            marker=dict(
                                symbol=DATASET_MARKERS[dataset],
                                size=6,
                                color=DATASET_COLORS[dataset]
                            ),
                            legendgroup=dataset,
                            showlegend=(plot_idx == 0),  # Only show legend in first subplot
                        ),
                        row=row,
                        col=col
                    )
            
            # Update axes
            if layers:
                fig.update_xaxes(
                    title_text="Layer",
                    range=[min(layers) - 1, max(layers) + 1],
                    row=row,
                    col=col
                )
                
                # Use global range if available (for normalization across models)
                if global_ranges is not None:
                    key = f"{matrix_type}_{wasserstein_type}"
                    if key in global_ranges:
                        y_min, y_max = global_ranges[key]
                    else:
                        # Fallback to local range if global not found
                        y_values = combo_data['AvgDistance'].values
                        if len(y_values) > 0:
                            # For log scale, ensure minimum is positive and handle zeros
                            y_values_positive = y_values[y_values > 0]
                            if len(y_values_positive) > 0:
                                y_min = np.min(y_values_positive) * 0.9  # 10% padding below
                                y_max = np.max(y_values) * 1.1  # 10% padding above
                            else:
                                y_min, y_max = 0.001, 1.0
                        else:
                            y_min, y_max = 0.001, 1.0
                else:
                    # Calculate y-axis range locally (original behavior)
                    y_values = combo_data['AvgDistance'].values
                    if len(y_values) > 0:
                        # For log scale, ensure minimum is positive and handle zeros
                        y_values_positive = y_values[y_values > 0]
                        if len(y_values_positive) > 0:
                            y_min = np.min(y_values_positive) * 0.9  # 10% padding below
                            y_max = np.max(y_values) * 1.1  # 10% padding above
                        else:
                            y_min, y_max = 0.001, 1.0
                    else:
                        y_min, y_max = 0.001, 1.0
                
                # Use log scale for y-axis
                fig.update_yaxes(
                    title_text=f"Wasserstein {wasserstein_type} (log scale)",
                    range=[np.log10(y_min), np.log10(y_max)],
                    type="log",
                    row=row,
                    col=col
                )
            
            plot_idx += 1
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1800,
        template="plotly_white",
        title=f"{model_key} - {config_type} Finetuning",
        legend_title="Dataset",
        font=dict(size=13),
        margin=dict(l=60, r=40, t=100, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save figure
    output_path = f"{OUTPUT_DIR}/{model_key}_{config_type}_boxplots.html"
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    
    # Print analysis results
    datasets_for_model = get_datasets_for_model(model_key)
    if analysis_file:
        analysis_file.write(f"\n  Analysis for {model_key} - {config_type}:\n")
        for result in create_boxplot_with_lines_plotly.analysis_results:
            analysis_file.write(f"\n    {result['matrix']}-{result['wasserstein']}:\n")
            analysis_file.write(f"      Lowest layers per dataset:\n")
            for dataset in datasets_for_model:
                if dataset in result['lowest_by_dataset'] and len(result['lowest_by_dataset'][dataset]) > 0:
                    analysis_file.write(f"        {dataset.upper()}: {sorted(result['lowest_by_dataset'][dataset])}\n")
            analysis_file.write(f"      Layers consistently low in ALL datasets: {sorted(result['all_datasets_low'].keys())}\n")
            analysis_file.write(f"      Layers consistently low in MOST datasets (≥2): {sorted(result['most_datasets_low'].keys())}\n")
    
    # Also print to console
    print(f"\n  Analysis for {model_key} - {config_type}:")
    for result in create_boxplot_with_lines_plotly.analysis_results:
        print(f"    {result['matrix']}-{result['wasserstein']}:")
        print(f"      Lowest layers per dataset:")
        for dataset in datasets_for_model:
            if dataset in result['lowest_by_dataset'] and len(result['lowest_by_dataset'][dataset]) > 0:
                print(f"        {dataset.upper()}: {sorted(result['lowest_by_dataset'][dataset])}")
        print(f"      Layers consistently low in ALL datasets: {sorted(result['all_datasets_low'].keys())}")
        print(f"      Layers consistently low in MOST datasets (≥2): {sorted(result['most_datasets_low'].keys())}")


def calculate_global_y_ranges(df):
    """Calculate global y-axis ranges for each matrix-wasserstein combination across all models."""
    global_ranges = {}
    
    for config_type in CONFIGURATIONS.keys():
        global_ranges[config_type] = {}
        
        # Process all models to get max values
        all_avg_data = []
        for model_key in MODELS.keys():
            model_df = df[df['Model'] == model_key]
            if len(model_df) > 0:
                avg_data = calculate_averages(model_df, config_type, model_key=model_key)
                if len(avg_data) > 0:
                    all_avg_data.append(avg_data)
        
        if not all_avg_data:
            continue
        
        # Combine all models' data
        combined_avg = pd.concat(all_avg_data, ignore_index=True)
        
        # Calculate max for each combination
        for matrix_type in MATRIX_TYPES:
            for wasserstein_type in WASSERSTEIN_TYPES:
                combo_data = combined_avg[
                    (combined_avg['MatrixType'] == matrix_type) &
                    (combined_avg['WassersteinType'] == wasserstein_type)
                ]
                
                if len(combo_data) > 0:
                    y_values = combo_data['AvgDistance'].values
                    # For log scale, filter out zeros and ensure positive values
                    y_values_positive = y_values[y_values > 0]
                    if len(y_values_positive) > 0:
                        y_min = np.min(y_values_positive) * 0.9  # 10% padding below
                        y_max = np.max(y_values) * 1.1  # 10% padding above
                    else:
                        # If all values are zero or negative, use default
                        y_min, y_max = 0.001, 1.0
                    
                    key = f"{matrix_type}_{wasserstein_type}"
                    global_ranges[config_type][key] = (y_min, y_max)
                else:
                    key = f"{matrix_type}_{wasserstein_type}"
                    global_ranges[config_type][key] = (0.001, 1.0)
    
    return global_ranges


def main():
    """Main function to generate all boxplots."""
    print("Loading and processing data...")
    df = process_all_data()
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Models: {df['Model'].unique()}")
    print(f"Datasets: {df['Dataset'].unique()}")
    print(f"Layers: {sorted(df['Layer'].unique())}")
    
    # Calculate global y-axis ranges for normalization across models
    print("\nCalculating global y-axis ranges for normalization...")
    global_ranges = calculate_global_y_ranges(df)
    
    # Open analysis file for writing
    analysis_file = open(f"{OUTPUT_DIR}/layer_analysis.txt", "w")
    analysis_file.write("Analysis of Lowest Changing Layers Across Datasets\n")
    analysis_file.write("=" * 70 + "\n\n")
    
    # Generate plots for each model and configuration
    for model_key in MODELS.keys():
        model_df = df[df['Model'] == model_key]
        
        if len(model_df) == 0:
            print(f"Warning: No data for model {model_key}")
            continue
        
        analysis_file.write(f"\n{'='*70}\n")
        analysis_file.write(f"Model: {model_key}\n")
        analysis_file.write(f"{'='*70}\n\n")
        
        for config_type in CONFIGURATIONS.keys():
            print(f"\nProcessing {model_key} - {config_type}...")
            analysis_file.write(f"\nConfiguration: {config_type}\n")
            analysis_file.write("-" * 70 + "\n")
            
            avg_data = calculate_averages(model_df, config_type, model_key=model_key)
            
            if len(avg_data) > 0:
                # Get global ranges for this configuration
                config_ranges = global_ranges.get(config_type, {})
                create_boxplot_with_lines_plotly(model_key, config_type, avg_data, analysis_file, config_ranges)
            else:
                print(f"  No data for {model_key} - {config_type}")
                analysis_file.write(f"  No data for {model_key} - {config_type}\n")
    
    analysis_file.close()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print(f"Analysis saved to {OUTPUT_DIR}/layer_analysis.txt")


if __name__ == '__main__':
    main()

