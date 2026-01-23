import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

class ClinicalTrainer:
    """
    Professional-grade trainer for the MO-AFN Clinical Model.
    Features:
    - Weighted Loss for Class Imbalance
    - Early Stopping
    - Monte Carlo Dropout for Uncertainty Estimation
    """
    def __init__(self, model, optimizer, device, class_weights=None, patience=10):
        """
        Args:
            model: Instance of MO_AFN model.
            optimizer: PyTorch optimizer.
            device: 'cuda' or 'cpu'.
            class_weights (list or tensor): Weights for [Non-Responder, Responder].
                                            Example: [1.0, 3.0] if Responders are rare.
            patience: Number of epochs to wait for improvement before early stopping.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        
        # 1. Class Imbalance Handling
        # We use Weighted Cross Entropy Loss.
        # This penalizes mistakes on the minority class more heavily.
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in train_loader:
            # Unpack batch (Gen, Trans, Clin, Labels)
            gen_x, trans_x, clin_x, labels = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, attention_weights = self.model(gen_x, trans_x, clin_x)
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1

    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                gen_x, trans_x, clin_x, labels = [b.to(self.device) for b in batch]
                
                logits, attention_weights = self.model(gen_x, trans_x, clin_x)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1

    def train(self, train_loader, val_loader, epochs=100):
        """
        Main training loop with logging and Early Stopping.
        """
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            # Train
            t_loss, t_acc, t_f1 = self.train_epoch(train_loader)
            
            # Validate
            v_loss, v_acc, v_f1 = self.validate_epoch(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} F1: {t_f1:.4f} | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} F1: {v_f1:.4f}")
            
            # 2. Early Stopping Logic
            if v_loss < self.best_val_loss:
                self.best_val_loss = v_loss
                self.patience_counter = 0
                # Save best model state (omitted for brevity, typically torch.save here)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stop! No improvement for {self.patience} epochs.")
                break
        
        print("Training Complete.")

    def predict_with_uncertainty(self, gen_x, trans_x, clin_x, n_iterations=50):
        """
        Performs inference using Monte Carlo Dropout.
        
        Scientific Rationale:
        Standard neural networks provide a point estimate without uncertainty.
        By enabling dropout at inference time (Monte Carlo Dropout), we approximate 
        a Bayesian Neural Network. The distribution of predictions across multiple 
        forward passes approximates the posterior distribution of the model output.
        - Mean of passes -> Final Probability
        - Variance/Std of passes -> Uncertainty (Epistemic)
        
        Args:
            gen_x, trans_x, clin_x: Single patient inputs (tensors).
            n_iterations: Number of stochastic forward passes.
            
        Returns:
            final_pred_class (int): 0 or 1
            mean_prob (float): Probability of positive class (Responder)
            uncertainty (float): Standard deviation of predictions
        """
        self.model.train() # CRITICAL: Enable Dropout during inference
        
        probabilities = []
        
        with torch.no_grad():
            for _ in range(n_iterations):
                # Ensure batch dimension if missing
                if gen_x.dim() == 1:
                    gen_x = gen_x.unsqueeze(0)
                    trans_x = trans_x.unsqueeze(0)
                    clin_x = clin_x.unsqueeze(0)
                    
                logits, _ = self.model(gen_x, trans_x, clin_x)
                prob = torch.softmax(logits, dim=1)[:, 1].item() # Prob of Class 1
                probabilities.append(prob)
        
        # Calculate statistics
        probabilities = np.array(probabilities)
        mean_prob = np.mean(probabilities)
        uncertainty = np.std(probabilities)
        
        # Decision threshold 0.5
        final_pred_class = 1 if mean_prob >= 0.5 else 0
        
        return final_pred_class, mean_prob, uncertainty

def format_clinical_output(pred_class, mean_prob, uncertainty):
    """
    Formats the prediction into a clinician-friendly string.
    """
    label_map = {0: "Non-Responder", 1: "Responder"}
    class_label = label_map[pred_class]
    
    # Heuristic for Confidence Level based on uncertainty (Std Dev)
    # Low Std Dev -> High Confidence (model is consistent)
    if uncertainty < 0.05:
        confidence_level = "High"
        recommendation = ""
    elif uncertainty < 0.15:
        confidence_level = "Moderate"
        recommendation = ""
    else:
        confidence_level = "Low"
        recommendation = "- Clinician Review Recommended"
        
    output_str = (
        f"Prediction: {class_label} "
        f"(Probability: {mean_prob*100:.1f}% \u00B1 {uncertainty*100:.1f}%). "
        f"Confidence: {confidence_level} {recommendation}"
    )
    
    return output_str
