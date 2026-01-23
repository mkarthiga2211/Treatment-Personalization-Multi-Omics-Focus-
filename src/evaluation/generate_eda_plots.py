import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
# Path to the script's directory (src/evaluation)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to MO_AFN_Project (src/evaluation -> src -> MO_AFN_Project)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "eda")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set global style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300

def generate_mock_data(n_samples=200, n_genes=50, n_snps=50):
    """
    Generates usage mock data for Genomics, Transcriptomics, and Clinical modalities.
    """
    np.random.seed(42)
    
    # 1. Genomics: Minor Allele Frequencies (MAF) for SNPs
    # Most SNPs have low MAF, so we simulate a skewed distribution
    maf_values = np.random.beta(a=1, b=5, size=n_snps * 10) 
    # Filter to realistic range [0, 0.5]
    maf_values = maf_values[maf_values <= 0.5]
    genomics_df = pd.DataFrame({'MAF': maf_values})
    
    # 2. Transcriptomics: Gene Expression
    # Raw data is often log-normal (skewed)
    raw_expression = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    transcriptomics_df = pd.DataFrame(raw_expression, columns=[f"Gene_{i}" for i in range(n_genes)])
    
    # 3. Clinical: Responder Status
    # Binary classification with some imbalance
    responses = np.random.choice(['Responder', 'Non-Responder'], size=n_samples, p=[0.35, 0.65])
    clinical_df = pd.DataFrame({'Status': responses})
    
    # 4. Correlation Stub
    # Create valid numeric data for correlation
    # Let's make some genes correlated with some random 'SNP' scores
    snp_scores = np.random.randint(0, 3, size=(n_samples, 10)) # 0, 1, 2 alleles
    snp_df = pd.DataFrame(snp_scores, columns=[f"SNP_{i}" for i in range(10)])
    
    # Combine a subset for correlation
    correlation_df = pd.concat([transcriptomics_df.iloc[:, :10], snp_df], axis=1)
    
    return genomics_df, transcriptomics_df, clinical_df, correlation_df

def plot_genomics_qc(genomics_df):
    """
    Visualizes the distribution of Minor Allele Frequencies (MAF).
    Biological Insight: Helps identify rare variants that might need filtering to reduce noise/overfitting.
    Standard QC often filters SNPs with MAF < 0.01 or 0.05.
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=genomics_df, x='MAF', bins=30, kde=True, color='teal', edgecolor='black')
    plt.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='QC Cutoff (0.05)')
    
    plt.title('Genomics QC: Minor Allele Frequency Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Minor Allele Frequency (MAF)')
    plt.ylabel('Count of SNPs')
    plt.legend()
    
    save_path = os.path.join(FIGURES_DIR, "01_Genomics_MAF_Distribution.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_transcriptomics_qc(transcriptomics_df):
    """
    Visualizes Raw vs Log2 Transformed Gene Expression.
    Biological Insight: Transcriptomic data is often highly skewed. Log2 transformation usually 
    normalizes the distribution, stabilizing variance for downstream statistical tests.
    """
    # Create Log2 transformed data
    log2_expression = np.log2(transcriptomics_df + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Raw Expression (Subset of first 5 genes for clarity)
    subset_cols = transcriptomics_df.columns[:5]
    sns.boxplot(data=transcriptomics_df[subset_cols], ax=axes[0], palette="viridis")
    axes[0].set_title('Raw Gene Expression (Highly Skewed)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Expression Counts')
    
    # Plot 2: Log2 Normalized
    sns.boxplot(data=log2_expression[subset_cols], ax=axes[1], palette="magma")
    axes[1].set_title('Log2 Normalized Expression (Centered)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Log2(Counts + 1)')
    
    plt.suptitle('Transcriptomics QC: Normalization Check', fontsize=16)
    
    save_path = os.path.join(FIGURES_DIR, "02_Transcriptomics_Normalization.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_clinical_balance(clinical_df):
    """
    Visualizes the class balance between Responders and Non-Responders.
    Biological Insight: Severe class imbalance can bias the model towards the majority class.
    Knowing the ratio is crucial for choosing metrics (F1 vs Accuracy) and loss functions (Weighted CE).
    """
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(data=clinical_df, x='Status', palette="coolwarm", order=['Non-Responder', 'Responder'])
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12, fontweight='bold')
        
    plt.title('Clinical Label Balance: Responder vs Non-Responder', fontsize=14, fontweight='bold')
    plt.xlabel('Treatment Outcome')
    plt.ylabel('Patient Count')
    
    save_path = os.path.join(FIGURES_DIR, "03_Clinical_Class_Balance.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_modality_correlation(correlation_df):
    """
    Visualizes correlations between a subset of Genes and SNPs.
    Biological Insight: High correlation between a SNP and a Gene might indicate a localized 
    genetic regulation (eQTL effect), which is a key biological feature our model should capture.
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Multi-Omics Correlation Heatmap (Top Genes vs Top SNPs)', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(FIGURES_DIR, "04_Modality_Correlation_Heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    print("Generating mock multi-omics data...")
    gen_df, trans_df, clin_df, corr_df = generate_mock_data()
    
    print("Generating EDA Plots...")
    plot_genomics_qc(gen_df)
    plot_transcriptomics_qc(trans_df)
    plot_clinical_balance(clin_df)
    plot_modality_correlation(corr_df)
    
    print("All EDA visualization complete.")
