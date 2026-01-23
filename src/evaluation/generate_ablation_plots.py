import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "ablation")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

def generate_ablation_data():
    """
    Generates mock data for Ablation and Fusion comparison.
    """
    # 1. Ablation Data (Drop in Accuracy)
    ablation_scores = {
        'Full MO-AFN': 0.88,
        'w/o Clinical': 0.84,
        'w/o Genomics': 0.81,
        'w/o Transcriptomics': 0.79
    }
    
    # 2. Fusion Strategy Data
    fusion_data = [
        {'Strategy': 'Simple Concatenation', 'Metric': 'Accuracy', 'Score': 0.76},
        {'Strategy': 'Simple Concatenation', 'Metric': 'F1-Score', 'Score': 0.74},
        
        {'Strategy': 'Coordinate Attention', 'Metric': 'Accuracy', 'Score': 0.80},
        {'Strategy': 'Coordinate Attention', 'Metric': 'F1-Score', 'Score': 0.78},
        
        {'Strategy': 'MO-AFN (Cross-Modal)', 'Metric': 'Accuracy', 'Score': 0.88},
        {'Strategy': 'MO-AFN (Cross-Modal)', 'Metric': 'F1-Score', 'Score': 0.87}
    ]
    fusion_df = pd.DataFrame(fusion_data)
    
    return ablation_scores, fusion_df

def plot_ablation_study(ablation_scores):
    """
    Figure 1: Ablation Study (Contribution of Modalities).
    Vertical Bar Chart with 'Drop' annotations.
    """
    print("Generating Figure 1: Ablation Study...")
    
    plt.figure(figsize=(9, 6))
    
    # Sort for visual flow (Full first, then descending)
    # Actually, let's keep specific order: Full -> removing least important -> removing most important
    # defined in data generation somewhat, but let's sort by score descending
    sorted_items = sorted(ablation_scores.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]
    
    # Plot
    # Use valid seaborn color palette
    palette = sns.color_palette("Blues_r", len(names))
    bars = plt.bar(names, scores, color=palette, edgecolor='black', alpha=0.9)
    
    # Reference Line (Full MO-AFN Score)
    full_score = ablation_scores['Full MO-AFN']
    plt.axhline(y=full_score, color='#D32F2F', linestyle='--', linewidth=2, label=f'Baseline ({full_score*100:.0f}%)')
    
    # Annotations
    for bar, score, name in zip(bars, scores, names):
        height = bar.get_height()
        
        # Percentage text
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height - 0.05, 
            f'{score:.2f}', 
            ha='center', va='bottom', color='white', fontweight='bold'
        )
        
        # Drop annotation (if not the full model)
        if name != 'Full MO-AFN':
            drop = full_score - score
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01, 
                f'-{drop*100:.1f}%', 
                ha='center', va='bottom', color='#D32F2F', fontweight='bold', fontsize=11
            )
            
    plt.title("Ablation Study: Impact of Removing Modalities", fontsize=14, fontweight='bold')
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0.6, 0.95) # Zoom in to show differences clearly
    plt.legend(loc='upper right')
    
    save_path = os.path.join(FIGURES_DIR, "Fig1_Ablation_Study.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_fusion_comparison(fusion_df):
    """
    Figure 2: Fusion Strategy Comparison.
    Grouped Bar Chart.
    """
    print("Generating Figure 2: Fusion Strategy Comparison...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot
    ax = sns.barplot(
        data=fusion_df, 
        x='Strategy', 
        y='Score', 
        hue='Metric', 
        palette='Paired',
        edgecolor='black'
    )
    
    # Annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.title("Impact of Fusion Strategy on Performance", fontsize=14, fontweight='bold')
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    save_path = os.path.join(FIGURES_DIR, "Fig2_Fusion_Strategy.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    
    # 1. Generate Data
    abl_data, fus_df = generate_ablation_data()
    
    # 2. Plot
    plot_ablation_study(abl_data)
    plot_fusion_comparison(fus_df)
    
    print("Ablation & Fusion Visualization Complete.")
