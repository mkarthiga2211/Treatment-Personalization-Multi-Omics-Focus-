import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to MO_AFN_Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "evaluation")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style Settings for Publication Quality
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'

def generate_fusion_evidence_data():
    """
    Generates mock data strictly for visualizing Phase A deliverables.
    """
    np.random.seed(42)
    
    # 1. Attention Weights (50 Patients, 3 Modalities)
    # Pattern: 
    # Patients 0-24: Favor Genomics (Col 0)
    # Patients 25-49: Favor Clinical (Col 2)
    attention_weights = np.zeros((50, 3))
    
    # Group 1: Genomic driven (High prob on idx 0)
    # Generate base [0.7, 0.15, 0.15] with some noise
    for i in range(25):
        noise = np.random.dirichlet([0.7, 0.15, 0.15])
        # Ensure dominant is still genomics for visualization clarity
        base = np.array([0.7, 0.15, 0.15])
        val = base * 0.8 + noise * 0.2
        attention_weights[i] = val / val.sum()
        
    # Group 2: Clinical driven (High prob on idx 2)
    for i in range(25, 50):
        noise = np.random.dirichlet([0.15, 0.15, 0.7])
        base = np.array([0.15, 0.15, 0.7])
        val = base * 0.8 + noise * 0.2
        attention_weights[i] = val / val.sum()
        
    # 2. Fused Latent Vectors (100 Samples, 64 Dims) & Labels
    # Create two clear clusters
    # Non-Responders (0): Mean 0, Std 1
    # Responders (1): Mean 4, Std 1.5 (Significant shift to ensure separation)
    
    n_samples = 100
    latent_dim = 64
    
    # Half non-responders, half responders
    labels = np.array([0] * 50 + [1] * 50)
    
    vectors_0 = np.random.normal(loc=0.0, scale=1.0, size=(50, latent_dim))
    vectors_1 = np.random.normal(loc=4.0, scale=1.5, size=(50, latent_dim))
    
    fused_latent_vectors = np.vstack([vectors_0, vectors_1])
    
    # Shuffle indices to mix them in the array (labels still match)
    idx = np.random.permutation(n_samples)
    fused_latent_vectors = fused_latent_vectors[idx]
    labels = labels[idx]
    
    return attention_weights, fused_latent_vectors, labels

def plot_modality_heatmap(attention_weights):
    """
    Figure 1: Modality Weight Heatmap
    """
    print("Generating Figure 1: Modality Weight Heatmap...")
    
    plt.figure(figsize=(8, 10))
    
    # Create DataFrame for better labeling
    df = pd.DataFrame(attention_weights, columns=['Genomics', 'Transcriptomics', 'Clinical'])
    df.index.name = 'Patient ID'
    
    # Heatmap
    # RdYlBu_r: Red (High) -> Blue (Low). 'r' reverses it so Red is High.
    ax = sns.heatmap(
        df, 
        cmap='RdYlBu_r', 
        linewidths=0.05, 
        linecolor='white',
        cbar_kws={'label': 'Attention Strength'}
    )
    
    plt.title('Modality Weight Heatmap (Attention Scores)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Patient ID', fontsize=12)
    plt.xlabel('Data Source', fontsize=12)
    
    # Save
    save_path = os.path.join(FIGURES_DIR, "Fig1_Modality_Weight_Heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_latent_umap(latent_vectors, labels):
    """
    Figure 2: Latent Space Separability Plot (UMAP)
    """
    print("Generating Figure 2: Latent UMAP Projection...")
    
    # 1. Run UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latent_vectors)
    
    # 2. Plot
    plt.figure(figsize=(10, 8))
    
    # Map labels to colors/names
    # 0 -> Non-Responder (Red), 1 -> Responder (Blue)
    label_map = {0: 'Non-Responder', 1: 'Responder'}
    mapped_labels = [label_map[l] for l in labels]
    
    palette = {'Non-Responder': '#E53935', 'Responder': '#1E88E5'} # Distinct Red/Blue
    
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=mapped_labels,
        palette=palette,
        s=100, # Marker size
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    plt.title('MO-AFN Fused Representation (UMAP Projection)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('UMAP Axis 1', fontsize=12)
    plt.ylabel('UMAP Axis 2', fontsize=12)
    plt.legend(title='Clinical Outcome', title_fontsize=11)
    
    # Save
    save_path = os.path.join(FIGURES_DIR, "Fig2_Latent_Space_UMAP.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    print(f"Generating figures in: {FIGURES_DIR}")
    
    # 1. Generate Data
    att_w, lat_vecs, y = generate_fusion_evidence_data()
    
    # 2. Plot Figures
    plot_modality_heatmap(att_w)
    plot_latent_umap(lat_vecs, y)
    
    print("Visualization Pipeline Complete.")
