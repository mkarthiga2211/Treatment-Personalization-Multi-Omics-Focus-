import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# -------------------------------------------------------------------
# 1. Data Simulation
# -------------------------------------------------------------------

def generate_mock_omics_data(n_samples=1000, n_snps=5000, n_genes=2000, n_clinical=10):
    """
    Generates realistic mock Multi-Omics data for testing the pipeline.
    """
    print(f"Generating mock data: {n_samples} samples...")
    np.random.seed(42)

    # Genomics: SNPs are typically 0, 1, 2 (number of minor alleles)
    # We simulate this using a binomial distribution
    # Probability p roughly corresponds to MAF (Minor Allele Frequency)
    mafs = np.random.uniform(0.005, 0.5, n_snps) # Some SNPs will fall below 0.01 threshold
    genomics_data = np.array([np.random.binomial(2, p, n_samples) for p in mafs]).T
    genomics_df = pd.DataFrame(genomics_data, columns=[f"SNP_{i}" for i in range(n_snps)])
    
    # Transcriptomics: Gene Expression (Log-normal distribution usually)
    transcriptomics_data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    transcriptomics_df = pd.DataFrame(transcriptomics_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    
    # Clinical Data
    clinical_data = np.random.normal(0, 1, size=(n_samples, n_clinical))
    clinical_df = pd.DataFrame(clinical_data, columns=[f"Clin_{i}" for i in range(n_clinical)])
    
    return genomics_df, transcriptomics_df, clinical_df

# -------------------------------------------------------------------
# 2. Modality-Specific Preprocessing
# -------------------------------------------------------------------

class OmicsPreprocessor:
    def __init__(self, maf_threshold=0.01):
        self.maf_threshold = maf_threshold
        self.gene_scaler = StandardScaler()
        self.snps_kept = None

    def preprocess_genomics(self, df):
        """
        QC for Genomics: Filter SNPs with MAF < threshold.
        MAF = (Count of Minor Alleles) / (2 * N_Samples)
        """
        print("Preprocessing Genomics Data...")
        # Calculate MAF for each SNP
        # Sum of cols gives total minor alleles. 2*N is total alleles.
        maf = df.sum(axis=0) / (2 * df.shape[0])
        
        # In real data, minor allele is defined as the less common one, so MAF is always <= 0.5
        # If calc > 0.5, it means "0" was the minor allele, so we take 1 - p. 
        # For simplicity here, we assume standard encoding relative to minor allele.
        
        keep_mask = maf >= self.maf_threshold
        self.snps_kept = keep_mask[keep_mask].index
        
        filtered_df = df.loc[:, keep_mask]
        print(f"  - Original SNPs: {df.shape[1]}")
        print(f"  - SNPs after MAF < {self.maf_threshold} filter: {filtered_df.shape[1]}")
        return filtered_df

    def preprocess_transcriptomics(self, df):
        """
        Standard pipeline: Log2 transformation -> Z-score Standardization.
        """
        print("Preprocessing Transcriptomics Data...")
        # 1. Log2 Transformation (log2(x + 1) to handle zeros)
        log2_df = np.log2(df + 1)
        
        # 2. StandardScaler (Z-score)
        scaled_data = self.gene_scaler.fit_transform(log2_df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        
        print("  - Log2 transformation and Standardization applied.")
        return scaled_df

# -------------------------------------------------------------------
# 3. Parallel Denoising Autoencoders (DAE)
# -------------------------------------------------------------------

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_ratio=0.5, dropout_rate=0.1):
        """
        Args:
            input_dim: Number of input features (genes or SNPs).
            latent_dim: Size of the compressed representation.
            hidden_ratio: Factor to determine intermediate hidden layer size.
            dropout_rate: Probability of zeroing an element (noise injection).
        """
        super(DenoisingAutoencoder, self).__init__()
        
        hidden_dim = int(input_dim * hidden_ratio)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Noise injection during training
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU() # Latent representation (ReLU often good for sparsity)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            # No final activation for reconstruction if inputs are standardized (unbounded)
            # Use Sigmoid if inputs are [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def get_latent(self, x):
        """Returns only the compressed representation."""
        with torch.no_grad():
            self.eval()
            encoded = self.encoder(x)
            # Switch back to train mode if needed elsewhere, usually safe to leave
            self.train() 
            return encoded

# Helper to train an AE
def train_autoencoder(model, data_tensor, epochs=5, batch_size=32, lr=1e-3, modality_name="Omics"):
    print(f"\nTraining DAE for {modality_name}...")
    
    # Create DataLoader
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            inputs = batch[0]
            
            # Forward
            # In a Denoising AE, we often apply noise explicitly to inputs here 
            # OR rely on the Dropout layer in the encoder. 
            # Pytorch Dropout is active only during model.train(), satisfying the DAE requirement.
            
            latent, reconstruction = model(inputs)
            
            loss = criterion(reconstruction, inputs)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(loader):.4f}")
        
    return model

# -------------------------------------------------------------------
# 4. Pipeline Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    
    # A. Generate Data
    gen_df, trans_df, clin_df = generate_mock_omics_data()

    # B. Preprocess
    preprocessor = OmicsPreprocessor(maf_threshold=0.01)
    
    # Genomics
    gen_clean = preprocessor.preprocess_genomics(gen_df)
    gen_tensor = torch.FloatTensor(gen_clean.values)
    
    # Transcriptomics
    trans_clean = preprocessor.preprocess_transcriptomics(trans_df)
    trans_tensor = torch.FloatTensor(trans_clean.values)
    
    # C. Initialize Autoencoders
    # Calculate dimensions after preprocessing
    dim_gen = gen_clean.shape[1]
    dim_trans = trans_clean.shape[1]
    latent_dim = 64
    
    # Create models
    dae_genomics = DenoisingAutoencoder(input_dim=dim_gen, latent_dim=latent_dim)
    dae_transcriptomics = DenoisingAutoencoder(input_dim=dim_trans, latent_dim=latent_dim)
    
    # D. Train
    train_autoencoder(dae_genomics, gen_tensor, epochs=5, modality_name="Genomics")
    train_autoencoder(dae_transcriptomics, trans_tensor, epochs=5, modality_name="Transcriptomics")
    
    # E. Extract Latent Representations
    print("\nExtracting Latent Representations...")
    latent_gen = dae_genomics.get_latent(gen_tensor)
    latent_trans = dae_transcriptomics.get_latent(trans_tensor)
    
    print(f"Final Genomics Latent Shape: {latent_gen.shape}")           # Should be (1000, 64)
    print(f"Final Transcriptomics Latent Shape: {latent_trans.shape}") # Should be (1000, 64)
    
    print("\nPhase 1 Data Engineering Complete.")
