import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import umap
from math import pi

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "evaluation")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def plot_benchmark_comparison(results_dict):
    """
    1. Comparative Benchmarking (Bar Chart).
    Input: {'MO-AFN': {'Acc': 0.85, 'F1': 0.82}, 'RF': {'Acc': 0.72...}}
    """
    print("Generating Benchmark Comparison Plot...")
    
    # Transform dict to DataFrame for Seaborn
    data = []
    for model_name, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            data.append({'Model': model_name, 'Metric': metric_name, 'Score': value})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='viridis')
    
    plt.title("Model Performance Benchmarking", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_path = os.path.join(FIGURES_DIR, "01_Benchmark_Comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_latent_separation(latent_vectors, labels):
    """
    2. Latent Space Visualization (UMAP).
    Reduces latent vectors to 2D to show clustering of Responders vs Non-Responders.
    """
    print("Generating UMAP Latent Space Plot...")
    
    # Fit UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    # labels: 0 = Non-Responder (Red), 1 = Responder (Blue)
    colors = ['#EF5350' if l == 0 else '#42A5F5' for l in labels]
    label_names = ['Non-Responder' if l == 0 else 'Responder' for l in labels]
    
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=label_names, 
        palette={'Non-Responder': '#EF5350', 'Responder': '#42A5F5'},
        alpha=0.7,
        s=60
    )
    
    plt.title("Latent Space Separation (UMAP)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    save_path = os.path.join(FIGURES_DIR, "02_Latent_Space_UMAP.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
    
def plot_gene_shap_importance(shap_values, feature_names):
    """
    3. Global Explainability (SHAP Beeswarm).
    Visualizes the top genes driving the predictions.
    Note: 'shap_values' should be the output of explainer.shap_values(X).
    """
    print("Generating SHAP Importance Plot...")
    
    plt.figure() # SHAP manages its own figure usually, but safe to init
    
    # Create the summary plot
    # max_display limits to Top 20 features
    shap.summary_plot(shap_values, feature_names=feature_names, show=False, max_display=20)
    
    plt.title("Top Transcriptomic Features Driving Prediction", fontsize=14, fontweight='bold')
    
    save_path = os.path.join(FIGURES_DIR, "03_SHAP_Gene_Importance.png")
    plt.savefig(save_path, bbox_inches='tight') # Essential for SHAP plots
    plt.close()
    print(f"Saved: {save_path}")

def plot_attention_heatmap(attention_weights_matrix):
    """
    4. Novelty Plot: Attention Weight Heatmap.
    Visualizes how much focus the model puts on [Genomics, Transcriptomics, Clinical] for 50 patients.
    Input shape: (50, 3)
    """
    print("Generating Attention Heatmap...")
    
    plt.figure(figsize=(8, 12))
    
    # Columns: Gen, Trans, Clin
    sns.heatmap(
        attention_weights_matrix, 
        annot=False, 
        cmap="YlOrRd", 
        xticklabels=['Genomics', 'Transcriptomics', 'Clinical'],
        yticklabels=False, # Hide patient IDs for cleaner look
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title("Cross-Modal Attention Weights (Batch of 50)", fontsize=14, fontweight='bold')
    plt.ylabel("Patients (Samples)")
    
    save_path = os.path.join(FIGURES_DIR, "04_Attention_Heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_patient_radar(patient_id, attention_scores, predicted_class, confidence):
    """
    5. Individual Case Study (Radar Chart).
    Creates a 'Personalized Medicine Report' for a single patient.
    Input scores: [gen_score, trans_score, clin_score] (sum to 1)
    """
    print(f"Generating Radar Chart for Patient {patient_id}...")
    
    # Categories
    categories = ['Genomics', 'Transcriptomics', 'Clinical']
    N = len(categories)
    
    # We need to repeat the first value to close the circular loop
    values = attention_scores + [attention_scores[0]]
    
    # Calculate angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]
    
    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#FF5722')
    ax.fill(angles, values, '#FF5722', alpha=0.2)
    
    # Prediction Title
    label = "Responder" if predicted_class == 1 else "Non-Responder"
    plt.title(f"Patient {patient_id}: {label}\n(Conf: {confidence*100:.1f}%)", 
              size=14, color='black', y=1.1, fontweight='bold')
    
    save_path = os.path.join(FIGURES_DIR, f"05_Patient_{patient_id}_Radar.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

# --- Test Execution Block ---
if __name__ == "__main__":
    # Simulate Mock Data for Visualization Testing
    
    # 1. Benchmarking
    results = {
        'MO-AFN (Ours)': {'Accuracy': 0.88, 'F1-Score': 0.86, 'AUROC': 0.91},
        'Random Forest': {'Accuracy': 0.74, 'F1-Score': 0.71, 'AUROC': 0.78},
        'Genomic CNN': {'Accuracy': 0.69, 'F1-Score': 0.65, 'AUROC': 0.70}
    }
    plot_benchmark_comparison(results)
    
    # 2. UMAP
    latent_vecs = np.random.randn(200, 64) # 200 samples, 64 dim
    # Make clusters slightly separate for visualization effect
    latent_vecs[:100] += 2 
    labels = np.array([0]*100 + [1]*100)
    plot_latent_separation(latent_vecs, labels)
    
    # 3. SHAP
    # Create mock SHAP values (samples x features)
    mock_shap = np.random.randn(100, 20)
    feat_names = [f"Gene_{i}" for i in range(20)]
    plot_gene_shap_importance(mock_shap, feat_names)
    
    # 4. Attention Heatmap
    # (50 patients, 3 modalities), softmax normalized
    raw_att = np.random.rand(50, 3)
    att_weights = raw_att / raw_att.sum(axis=1, keepdims=True)
    plot_attention_heatmap(att_weights)
    
    # 5. Radar Chart
    single_scores = [0.6, 0.3, 0.1] # High Genomic contribution
    plot_patient_radar("001", single_scores, predicted_class=1, confidence=0.92)
