"""
Visualization code to generate the two graphs from zero-shot evaluation results
Add this to your existing code or run separately after evaluation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_accuracy_gains_bar(results):
    """
    Plot the bar chart showing Max, Mean, Min accuracy gains
    Left side of your image
    """
    # Extract CLIP baseline and CoN-CLIP results for each architecture
    # architectures = ['ViT-L/14', 'ViT-B/32', 'ViT-B/16']
    architectures = ['ViT-L/14', 'ViT-B/32', 'ViT-B/16']
    
    gains_data = {}
    
    for arch in architectures:
        clip_key = f"CLIP-{arch}"
        conclip_key = f"CoN-CLIP-{arch}"
        
        if clip_key in results and conclip_key in results:
            # Get matching pairs where both CLIP and CoN-CLIP have values
            gains = []
            for dataset in results[clip_key].keys():
                clip_val = results[clip_key][dataset]
                conclip_val = results[conclip_key][dataset]
                
                # Only calculate gain if both values exist
                if clip_val is not None and conclip_val is not None:
                    gains.append(conclip_val - clip_val)
            
            # Only add to gains_data if we have at least one valid gain
            if gains:
                gains_data[arch] = {
                    'max': max(gains),
                    'mean': np.mean(gains),
                    'min': min(gains)
                }
    
    # Filter architectures to only those with data
    valid_architectures = [arch for arch in architectures if arch in gains_data]
    
    if not valid_architectures:
        print("No valid data to plot!")
        return None
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(valid_architectures))
    width = 0.25
    
    # Create bars
    max_bars = ax.bar(x - width, [gains_data[arch]['max'] for arch in valid_architectures], 
                       width, label='Max', color='#2ecc71', alpha=0.8)
    mean_bars = ax.bar(x, [gains_data[arch]['mean'] for arch in valid_architectures], 
                        width, label='Mean', color='#e74c3c', alpha=0.8)
    min_bars = ax.bar(x + width, [gains_data[arch]['min'] for arch in valid_architectures], 
                       width, label='Min', color='#3498db', alpha=0.8)
    
    # Add value labels on bars
    for bars in [max_bars, mean_bars, min_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 accuracy gain (%)', fontsize=12, fontweight='bold')
    ax.set_title('Zero-shot classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_architectures)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('accuracy_gains_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_radar_chart(results, architecture='ViT-B/32'):
    """
    Plot the radar chart comparing CLIP and CoN-CLIP
    Right side of your image
    """
    # Select which architecture to plot
    clip_key = f"CLIP-{architecture}"
    conclip_key = f"CoN-CLIP-{architecture}"
    
    # Check if the architecture exists in results
    if clip_key not in results or conclip_key not in results:
        print(f"Architecture {architecture} not found in results!")
        return None
    
    # Define the specific datasets to include (matching your code)
    dataset_order = ['Caltech-101', 'CIFAR-10', 'Flowers-102', 'CIFAR-100', 'Oxford Pets']
    
    # Get accuracies only for datasets where BOTH models have non-None values
    clip_scores = []
    conclip_scores = []
    datasets = []
    
    for d in dataset_order:
        if d in results[clip_key] and d in results[conclip_key]:
            clip_val = results[clip_key][d]
            conclip_val = results[conclip_key][d]
            
            # Only include if both values are not None
            if clip_val is not None and conclip_val is not None:
                clip_scores.append(clip_val)
                conclip_scores.append(conclip_val)
                datasets.append(d)
    
    # Check if we have any valid data
    if not datasets:
        print(f"No valid datasets with complete data for {architecture}")
        return None
    
    # Number of variables
    num_vars = len(datasets)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    clip_scores += clip_scores[:1]
    conclip_scores += conclip_scores[:1]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, clip_scores, 'o-', linewidth=2, label='CLIP', color='#3498db')
    ax.fill(angles, clip_scores, alpha=0.15, color='#3498db')
    
    ax.plot(angles, conclip_scores, 'o-', linewidth=2, label='CoN-CLIP', color='#2ecc71')
    ax.fill(angles, conclip_scores, alpha=0.15, color='#2ecc71')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim(50, 100)
    
    # Add title and legend
    ax.set_title(f'Top-1 accuracy using {architecture} on {num_vars} datasets', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def generate_both_plots(results):
    """
    Generate both plots from results dictionary
    """
    # Generate bar chart
    print("Generating accuracy gains bar chart...")
    plot_accuracy_gains_bar(results)
    
    # Generate radar chart
    print("Generating radar chart...")
    plot_radar_chart(results, architecture='ViT-B/32')
    
    print("Plots saved!")


# Usage: Manually add your results here
if __name__ == "__main__":
    # Manual results dictionary - ADD YOUR VALUES HERE
    results = {
    "CLIP-ViT-B/32": {
        "Caltech-101": 85.63,
        "CIFAR-10": 88.32,
        "Flowers-102": 63.73,
        "CIFAR-100": 64.43,
        "Oxford Pets": 85.09
    },
    "CoN-CLIP-ViT-B/32": {
        "Caltech-101": 87.81,
        "CIFAR-10": 89.93,
        "Flowers-102": 65.05,
        "CIFAR-100": 64.75,
        "Oxford Pets": 84.74
    },
    "CLIP-ViT-B/16": {
        "Caltech-101": 84.42,
        "CIFAR-10": 90.10,
        "Flowers-102": 67.70,
        "CIFAR-100": 68.41,
        "Oxford Pets": 88.23
    },
    "CoN-CLIP-ViT-B/16": {
        "Caltech-101": 89.13,
        "CIFAR-10": 91.55,
        "Flowers-102": 68.09,
        "CIFAR-100": 69.31,
        "Oxford Pets": 88.72
    },
    "CLIP-ViT-L/14": {
        "Caltech-101": 85.54,
        "CIFAR-10": 95.17,
        "Flowers-102": 74.52,
        "CIFAR-100": 77.17,
        "Oxford Pets": 93.08
    },
    "CoN-CLIP-ViT-L/14": {
        "Caltech-101": 88.96,  
        "CIFAR-10": 95.84,     
        "Flowers-102": 75.51,  
        "CIFAR-100": 78.97,    
        "Oxford Pets": 91.85  
    }
}
    
    # Generate plots
    generate_both_plots(results)