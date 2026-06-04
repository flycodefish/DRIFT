"""Plotting module: Generate publication-quality training charts"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Set beautiful plot style
plt.style.use('seaborn-v0_8-paper')

# Publication-quality color scheme
COLORS = {
    'train_loss': '#1f77b4',      # Blue
    'val_loss': '#ff7f0e',        # Orange
    'val_acc': '#2ca02c',         # Green
    'comm_success': '#9467bd',    # Purple
    'comm_failure': '#d62728',    # Red
    'gossip_ops': '#17becf',      # Cyan
}


def setup_plot(font_size=12, fig_width=8, fig_height=5):
    """Setup publication-quality plot parameters"""
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size + 2
    plt.rcParams['axes.titlesize'] = font_size + 4
    plt.rcParams['xtick.labelsize'] = font_size
    plt.rcParams['ytick.labelsize'] = font_size
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.figsize'] = (fig_width, fig_height)
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def plot_loss_curve(round_metrics: pd.DataFrame, output_path: str):
    """Plot training and validation loss curves"""
    setup_plot()
    
    fig, ax = plt.subplots()
    
    ax.plot(round_metrics['round_id'], round_metrics['avg_train_loss'], 
            marker='o', linestyle='-', color=COLORS['train_loss'], 
            label='Training Loss')
    ax.plot(round_metrics['round_id'], round_metrics['val_loss'], 
            marker='s', linestyle='--', color=COLORS['val_loss'], 
            label='Validation Loss')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_curve(round_metrics: pd.DataFrame, output_path: str):
    """Plot validation accuracy curve"""
    setup_plot()
    
    fig, ax = plt.subplots()
    
    ax.plot(round_metrics['round_id'], round_metrics['val_acc'], 
            marker='^', linestyle='-', color=COLORS['val_acc'], 
            label='Validation Accuracy')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend(frameon=True, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add accuracy annotations (every 5 rounds)
    for i, row in round_metrics.iterrows():
        if i % 5 == 0:
            ax.text(row['round_id'], row['val_acc'] + 0.01, 
                    f'{row["val_acc"]:.2%}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_accuracy_combo(round_metrics: pd.DataFrame, output_path: str):
    """Plot combined chart: loss and accuracy"""
    setup_plot(fig_width=10, fig_height=5)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    
    # Left: Loss curves
    ax1.plot(round_metrics['round_id'], round_metrics['avg_train_loss'], 
             marker='o', linestyle='-', color=COLORS['train_loss'], 
             label='Training Loss')
    ax1.plot(round_metrics['round_id'], round_metrics['val_loss'], 
             marker='s', linestyle='--', color=COLORS['val_loss'], 
             label='Validation Loss')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Loss Curves')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Right: Accuracy curve
    ax2.plot(round_metrics['round_id'], round_metrics['val_acc'], 
             marker='^', linestyle='-', color=COLORS['val_acc'], 
             label='Validation Accuracy')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('(b) Accuracy Curve')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_communication_stats(round_metrics: pd.DataFrame, output_path: str):
    """Plot communication statistics"""
    setup_plot(fig_width=10, fig_height=6)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    
    # Communication success rate
    total_comm = round_metrics['comm_success_count'] + round_metrics['comm_failure_count']
    success_rate = round_metrics['comm_success_count'] / total_comm.replace(0, 1)
    
    ax1.plot(round_metrics['round_id'], success_rate * 100, 
             marker='o', linestyle='-', color=COLORS['comm_success'], 
             label='Success Rate')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Communication Success Rate')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 105)
    
    # Gossip operations
    if 'gossip_success_count' in round_metrics.columns:
        ax2.plot(round_metrics['round_id'], round_metrics['gossip_success_count'], 
                 marker='s', linestyle='-', color=COLORS['gossip_ops'], 
                 label='Gossip Success')
        ax2.plot(round_metrics['round_id'], round_metrics['gossip_failure_count'], 
                 marker='x', linestyle='--', color=COLORS['comm_failure'], 
                 label='Gossip Failure')
        ax2.set_ylabel('Operations')
        ax2.set_title('Gossip Communication Statistics')
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel('Round')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_client_stats(round_metrics: pd.DataFrame, output_path: str):
    """Plot client statistics"""
    setup_plot()
    
    fig, ax = plt.subplots()
    
    ax.plot(round_metrics['round_id'], round_metrics['selected_count'], 
            marker='o', linestyle='-', color='#1f77b4', 
            label='Selected Clients')
    ax.plot(round_metrics['round_id'], round_metrics['dropped_count'], 
            marker='x', linestyle='--', color='#d62728', 
            label='Dropped Clients')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Number of Clients')
    ax.set_title('Client Participation')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(round_metrics: pd.DataFrame, config: dict, output_dir: str):
    """Generate complete training report"""
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating training report...")
    
    # Plot all charts
    plot_loss_curve(round_metrics, os.path.join(output_dir, 'loss_curve.png'))
    plot_accuracy_curve(round_metrics, os.path.join(output_dir, 'accuracy_curve.png'))
    plot_loss_accuracy_combo(round_metrics, os.path.join(output_dir, 'loss_accuracy_combo.png'))
    plot_communication_stats(round_metrics, os.path.join(output_dir, 'communication_stats.png'))
    plot_client_stats(round_metrics, os.path.join(output_dir, 'client_stats.png'))
    
    # Generate summary statistics
    summary = {
        'algorithm': config.get('algorithm', 'unknown'),
        'topology': config.get('topology', 'unknown'),
        'num_clients': config.get('num_clients', 0),
        'rounds': config.get('rounds', 0),
        'final_val_acc': round_metrics['val_acc'].iloc[-1] if len(round_metrics) > 0 else 0,
        'final_val_loss': round_metrics['val_loss'].iloc[-1] if len(round_metrics) > 0 else 0,
        'min_val_loss': round_metrics['val_loss'].min() if len(round_metrics) > 0 else 0,
        'max_val_acc': round_metrics['val_acc'].max() if len(round_metrics) > 0 else 0,
        'avg_train_loss': round_metrics['avg_train_loss'].mean() if len(round_metrics) > 0 else 0,
    }
    
    # Save summary to JSON
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate text report
    report_text = f"""Training Report
===============

1. Experiment Configuration
---------------------------
Algorithm: {summary['algorithm']}
Topology: {summary['topology']}
Number of Clients: {summary['num_clients']}
Number of Rounds: {summary['rounds']}

2. Experiment Results
---------------------
Final Validation Accuracy: {summary['final_val_acc']:.2%}
Final Validation Loss: {summary['final_val_loss']:.4f}
Minimum Validation Loss: {summary['min_val_loss']:.4f}
Maximum Validation Accuracy: {summary['max_val_acc']:.2%}
Average Training Loss: {summary['avg_train_loss']:.4f}

3. Chart Files
--------------
1. loss_curve.png - Training and validation loss curves
2. accuracy_curve.png - Validation accuracy curve
3. loss_accuracy_combo.png - Combined loss and accuracy chart
4. communication_stats.png - Communication statistics
5. client_stats.png - Client participation statistics

4. Data Files
-------------
1. round_metrics.csv - Per-round metrics
2. comm_events.csv - Communication events
3. summary.json - Experiment summary

===============
Report generated successfully
"""
    
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report_text)
    
    print(f"Training report generated to: {output_dir}")
    print(f"Generated files:")
    print(f"  - loss_curve.png")
    print(f"  - accuracy_curve.png")
    print(f"  - loss_accuracy_combo.png")
    print(f"  - communication_stats.png")
    print(f"  - client_stats.png")
    print(f"  - summary.json")
    print(f"  - report.txt")
    
    return summary


def load_metrics_and_generate_report(metrics_path: str, config: dict, output_dir: str):
    """Load metrics from CSV and generate report"""
    round_metrics = pd.read_csv(metrics_path)
    return generate_summary_report(round_metrics, config, output_dir)
