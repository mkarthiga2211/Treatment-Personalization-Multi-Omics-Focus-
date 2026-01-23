import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne_connectivity.viz import plot_connectivity_circle

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "interpretability")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def generate_interpretability_data():
    """
    Simulates Bio-Interpretability Data:
    1. Gene SHAP Scores
    2. SNP-Gene Interaction Matrix
    """
    np.random.seed(42)
    
    # --- 1. Gene SHAP Data ---
    # Top known depression markers
    key_genes = ["BDNF", "SLC6A4", "TPH2", "FKBP5", "COMT", "MAOA", "DISC1", "NR3C1", "HTR2A", "CACNA1C"]
    
    # 90 Random genes
    random_genes = [f"Gene_{i}" for i in range(90)]
    all_genes = key_genes + random_genes
    
    # Assign scores
    # Key genes get high scores (0.6 - 1.0)
    # Random genes get low scores (0.0 - 0.2)
    scores = []
    for g in all_genes:
        if g in key_genes:
            scores.append(np.random.uniform(0.6, 1.0))
        else:
            scores.append(np.random.uniform(0.0, 0.2))
            
    shap_df = pd.DataFrame({'Gene': all_genes, 'Mean_Abs_SHAP': scores})
    shap_df = shap_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    
    # --- 2. Cross-Modal Interaction Matrix (10 SNPs x 10 Genes) ---
    # We will visualize connectivity between 10 SNPs and 10 Genes (20 nodes total)
    snps = [f"rs{i}" for i in range(1234, 1244)] # rs1234..rs1243
    genes_sub = key_genes 
    
    node_names = snps + genes_sub
    n_nodes = len(node_names)
    con_matrix = np.zeros((n_nodes, n_nodes))
    
    # Create specific high-strength links
    # Imagine SNP rs1234 regulates BDNF
    links = [
        (0, 10), # rs1234 -> BDNF (idx 10 is BDNF)
        (1, 11), # rs1235 -> SLC6A4
        (0, 15), # rs1234 -> MAOA (Pleiotropy)
        (5, 10), # rs1239 -> BDNF (Polygenic regulation)
        (9, 19), # rs1243 -> CACNA1C
        (2, 12),
        (3, 13)
    ]
    
    for (src, dst) in links:
        # Symmetric connectivity for visualization
        con_matrix[src, dst] = 0.95
        con_matrix[dst, src] = 0.95
        
    return shap_df, con_matrix, node_names

def plot_gene_shap_summary(shap_df):
    """
    Figure 1: Gene Contribution Score (Horizontal Bar Chart)
    """
    print("Generating Figure 1: Gene SHAP Contribution...")
    
    plt.figure(figsize=(8, 10))
    
    # Top 20 Genes
    top_20 = shap_df.head(20)
    
    sns.barplot(
        data=top_20,
        x='Mean_Abs_SHAP',
        y='Gene',
        palette='magma'
    )
    
    plt.title("Top 20 Genes Driving 'Responder' Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("mean(|SHAP value|) - Impact on Model Output", fontsize=12)
    plt.ylabel("Gene Name", fontsize=12)
    
    # Highlight specific biological markers via annotation or check logic
    # (Visual check essentially)
    
    save_path = os.path.join(FIGURES_DIR, "Fig1_Gene_SHAP_Contribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_cross_modal_circos(con_matrix, node_names):
    """
    Figure 2: Cross-Modal Correlation Plot (Circos-Style) using MNE.
    """
    print("Generating Figure 2: Cross-Modal Interaction Circos...")
    
    # Define colors
    # First 10 are SNPs (Green), Next 10 are Genes (Blue)
    node_colors = ['#66BB6A'] * 10 + ['#42A5F5'] * 10
    
    # Create the plot
    # plot_connectivity_circle returns (fig, axes)
    fig, ax = plot_connectivity_circle(
        con_matrix, 
        node_names, 
        node_colors=node_colors,
        node_edgecolor='white',
        textcolor='black',
        facecolor='white',
        colormap='Reds', # Color of the connecting lines
        vmin=0.0, vmax=1.0, # Range for line strength
        linewidth=2.5,
        title="Cross-Modal Interaction (SNPs \u2194 Genes)",
        show=False,
        interactive=False
    )
    
    # MNE creates a figure with specific figsize, we can adjust or just save
    save_path = os.path.join(FIGURES_DIR, "Fig2_Cross_Modal_Circos.png")
    fig.savefig(save_path, dpi=300, facecolor='white')
    plt.close(fig) # Close explicitly
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    
    # 1. Generate Data
    shap_df, con_matrix, node_names = generate_interpretability_data()
    
    # 2. Plot
    plot_gene_shap_summary(shap_df)
    plot_cross_modal_circos(con_matrix, node_names)
    
    print("Interpretability Visualization Complete.")
