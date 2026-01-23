import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
import numpy as np

# Adjust path to import parallel_autoencoders and mo_afn
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.mo_afn import MO_AFN

# Dummy Data Generator Strategy (Replicating Phase 1 logic roughly for speed)
# In real scenario, we load the saved Latent Representations from Phase 1
def load_mock_latent_data(n_samples=500):
    # Simulated Latent vectors (output of AEs)
    gen_latent = torch.randn(n_samples, 64)
    trans_latent = torch.randn(n_samples, 64)
    clinical = torch.randn(n_samples, 10)
    
    # Target: 0 (Non-Responder), 1 (Responder)
    labels = torch.randint(0, 2, (n_samples,))
    
    return gen_latent, trans_latent, clinical, labels

# Global Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    """
    Optuna Objective Function.
    Optimizes hyperparameters for the MO-AFN model.
    """
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    attention_dim = trial.suggest_categorical("attention_dim", [32, 64, 128])
    batch_size = 32
    
    # 2. Prepare Data
    gen_l, trans_l, clin, y = load_mock_latent_data()
    
    # Split
    # We use slices for simplicity in PyTorch tensors
    split_idx = int(0.8 * len(y))
    
    train_gen, val_gen = gen_l[:split_idx], gen_l[split_idx:]
    train_trans, val_trans = trans_l[:split_idx], trans_l[split_idx:]
    train_clin, val_clin = clin[:split_idx], clin[split_idx:]
    train_y, val_y = y[:split_idx], y[split_idx:]
    
    train_dataset = TensorDataset(train_gen, train_trans, train_clin, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_gen, val_trans, val_clin, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) # No shuffle for validation
    
    # 3. Model Setup
    model = MO_AFN(
        genomic_dim=64, 
        transcriptomic_dim=64, 
        clinical_dim=10, 
        attention_dim=attention_dim, 
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop (Short Epochs for Tuning)
    for epoch in range(10): 
        model.train()
        for batch in train_loader:
            bg, bt, bc, by = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            logits, _ = model(bg, bt, bc)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                bg, bt, bc, by = [b.to(DEVICE) for b in batch]
                logits, _ = model(bg, bt, bc)
                loss = criterion(logits, by)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(by.cpu().numpy())
        
        current_val_acc = accuracy_score(all_targets, all_preds)
        
        # 5. Pruning (Stop unpromising trials)
        trial.report(current_val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return current_val_acc

if __name__ == "__main__":
    print("Starting Bayesian Optimization with Optuna...")
    
    # Create Study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    # Optimize
    study.optimize(objective, n_trials=20) # 20 trials for demonstration
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    print("\nOptimization Complete.")
