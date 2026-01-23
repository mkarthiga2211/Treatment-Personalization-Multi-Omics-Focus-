import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from math import pi

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "xai")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def generate_xai_data():
    """
    Generates mock SHAP data and Attention Weights for visualization.
    """
    np.random.seed(42)
    n_samples = 100
    
    # --- 1. SHAP Values ---
    # Features: 10 Genes, 5 SNPs, 5 Clinical
    genes = [f"Gene_{name}" for name in ["BDNF", "SLC6A4", "TPH2", "FKBP5", "COMT", "MAOA", "DISC1", "NR3C1", "HTR2A", "CACNA1C"]]
    snps = [f"SNP_rs{i}" for i in range(6265, 6270)]
    clinical = ["Clin_Insomnia", "Clin_BMI", "Clin_Age", "Clin_TraumaHist", "Clin_PrevMeds"]
    
    feature_names = genes + snps + clinical # 20 features
    
    # Create random feature matrix (X) for the color mapping (Red/Blue)
    X_mock = pd.DataFrame(np.random.randn(n_samples, 20), columns=feature_names)
    
    # Create SHAP values matrix
    # Make BDNF and Insomnia 'important' (large absolute values)
    shap_values = np.random.randn(n_samples, 20) * 0.1 # Base noise
    
    # Gene_BDNF (idx 0): High impact
    shap_values[:, 0] = np.random.randn(n_samples) * 0.8
    # Clin_Insomnia (idx 15): High impact
    shap_values[:, 15] = np.random.randn(n_samples) * 0.7
    # SNP_rs6265 (idx 10): Medium impact
    shap_values[:, 10] = np.random.randn(n_samples) * 0.5
    
    # --- 2. Attention Weights (Case Study) ---
    patients_data = {
        'Patient A': [0.70, 0.20, 0.10], # Genetics-driven
        'Patient B': [0.10, 0.30, 0.60]  # History-driven
    }
    
    return shap_values, X_mock, feature_names, patients_data

def plot_shap_summary(shap_values, X, feature_names):
    """
    Figure 1: T-SHAP Global Feature Importance (Bee-Swarm Plot)
    """
    print("Generating Figure 1: Global SHAP Summary...")
    
    plt.figure() # SHAP handles figure creation often, but good practice
    
    # shap.summary_plot
    # Note: We just pass the matrices. 'shap_values' is the array of impact scores.
    # 'X' is the feature value matrix (for color).
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        show=False,
        max_display=20,
        cmap='coolwarm'
    )
    
    plt.title("Global Feature Importance (Multi-Omics)", fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(FIGURES_DIR, "Fig1_Global_SHAP_Summary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_patient_radar_comparison(patients_data):
    """
    Figure 2: Individual Patient Case Study (Radar Chart).
    Overlays Patient A and Patient B.
    """
    print("Generating Figure 2: Patient Attention Profiles (Radar Chart)...")
    
    categories = ['Genomics', 'Transcriptomics', 'Clinical']
    N = len(categories)
    
    # Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]] # Close the loop
    
    # Initialize the spider plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Axes
    plt.xticks(angles[:-1], categories, color='grey', size=12, fontweight='bold')
    
    # Y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Colors
    colors = {'Patient A': '#1E88E5', 'Patient B': '#FF7043'} # Blue, Orange
    
    for patient_name, scores in patients_data.items():
        values = scores + [scores[0]] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=patient_name, color=colors[patient_name])
        ax.fill(angles, values, color=colors[patient_name], alpha=0.2)
        
    plt.title("Patient-Specific Attention Profile", fontsize=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    save_path = os.path.join(FIGURES_DIR, "Fig2_Patient_Attention_Radar.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    
    # 1. Generate Data
    shap_vals, X_mock, feat_names, pat_data = generate_xai_data()
    
    # 2. Plot
    plot_shap_summary(shap_vals, X_mock, feat_names)
    plot_patient_radar_comparison(pat_data)
    
    print("XAI Visualization Complete.")
