import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Layer.
    Projects Genomic, Transcriptomic, and Clinical features into a shared space
    to learn non-linear interactions and importance weights.
    """
    def __init__(self, genomic_dim, transcriptomic_dim, clinical_dim, attention_dim=64):
        super(CrossModalAttention, self).__init__()
        
        # Projection layers to map inputs to shared dimension
        self.gen_proj = nn.Linear(genomic_dim, attention_dim)
        self.trans_proj = nn.Linear(transcriptomic_dim, attention_dim)
        self.clin_proj = nn.Linear(clinical_dim, attention_dim)
        
        # Attention scoring mechanism
        # Learnable vector to compute importance scores from the projected features
        self.attention_vector = nn.Parameter(torch.randn(attention_dim, 1))
        
        # Dropout for regularization within attention
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, gen_x, trans_x, clin_x):
        """
        Args:
            gen_x: Latent genomics tensor (Batch, Gen_Dim)
            trans_x: Latent transcriptomics tensor (Batch, Trans_Dim)
            clin_x: Clinical features tensor (Batch, Clin_Dim)
            
        Returns:
            fused_vector: Weighted combination of inputs (Batch, Attention_Dim)
            attention_weights: The calculated weights for visualization (Batch, 3)
        """
        # 1. Project Modalities to Shared Space (Batch, Attention_Dim)
        g_proj = torch.tanh(self.gen_proj(gen_x))
        t_proj = torch.tanh(self.trans_proj(trans_x))
        c_proj = torch.tanh(self.clin_proj(clin_x))
        
        # Stack them: (Batch, 3, Attention_Dim)
        stacked = torch.stack([g_proj, t_proj, c_proj], dim=1)
        
        # 2. Calculate Attention Scores
        # We want to learn which modality is most "important" for each patient
        # Dot product with learnable attention vector -> (Batch, 3, 1)
        scores = torch.matmul(stacked, self.attention_vector)
        
        # Squeeze to (Batch, 3) and apply Softmax to get weights summing to 1
        att_weights = F.softmax(scores.squeeze(-1), dim=1)
        
        # 3. Weighted Fusion
        # Multiply weights (Batch, 3, 1) * Stacked Features (Batch, 3, Dim)
        # Sum across dimension 1 to get fused vector (Batch, Dim)
        weighted_features = stacked * att_weights.unsqueeze(-1)
        fused_vector = torch.sum(weighted_features, dim=1)
        
        fused_vector = self.dropout(fused_vector)
        
        return fused_vector, att_weights


class MO_AFN(nn.Module):
    """
    Multi-Omics Attention Fusion Network.
    
    Structure:
    1. Inputs (Genomics Latent, Transcriptomics Latent, Clinical)
    2. Cross-Modal Attention Fusion
    3. MLP Classifier Head with Monte Carlo Dropout
    """
    def __init__(self, genomic_dim=64, transcriptomic_dim=64, clinical_dim=10, 
                 attention_dim=64, hidden_dim=32, dropout_rate=0.3):
        super(MO_AFN, self).__init__()
        
        self.fusion_layer = CrossModalAttention(
            genomic_dim, transcriptomic_dim, clinical_dim, attention_dim
        )
        
        # Classifier Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Monte Carlo Dropout Layer 1
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Monte Carlo Dropout Layer 2
            nn.Linear(hidden_dim // 2, 2) # Output logits for Binary Classification (Resp vs Non-Resp)
        )
        
    def forward(self, gen_x, trans_x, clin_x):
        # 1. Fuse
        fused_embedding, attention_weights = self.fusion_layer(gen_x, trans_x, clin_x)
        
        # 2. Classify
        logits = self.classifier(fused_embedding)
        
        return logits, attention_weights

    def predict_mc_dropout(self, gen_x, trans_x, clin_x, n_samples=50):
        """
        Performs Monte Carlo Dropout inference.
        Runs the model multiple times with dropout active to estimate uncertainty.
        """
        self.train() # Enable Dropout
        outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self(gen_x, trans_x, clin_x)
                probs = F.softmax(logits, dim=1)[:, 1] # Probability of Class 1 (Responder)
                outputs.append(probs)
        
        # Stack predictions (n_samples, batch_size)
        stacked_probs = torch.stack(outputs)
        
        # Mean probability (Prediction)
        mean_prob = stacked_probs.mean(dim=0)
        
        # Standard deviation (Uncertainty)
        uncertainty = stacked_probs.std(dim=0)
        
        return mean_prob, uncertainty
