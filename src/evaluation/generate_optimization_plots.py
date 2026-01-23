import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
FIGURES_DIR = os.path.join(PROJECT_ROOT, "notebooks", "figures", "optimization")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def generate_optimization_history(n_trials=100):
    """
    Simulates Bayesian Optimization history.
    Returns:
        trial_numbers: List of trial indices.
        values: Accuracy for each trial.
        best_values: Best accuracy up to that trial.
    """
    np.random.seed(42)
    
    trial_numbers = np.arange(n_trials)
    values = []
    
    # Simulate: 
    # Early trials: Random exploration (0.60 - 0.75)
    # Mid trials: Exploitation starts (0.75 - 0.82) around trial 40
    # Late trials: Fine-tuning (0.82 - 0.88) around trial 70
    
    for i in range(n_trials):
        if i < 30:
            val = np.random.uniform(0.60, 0.75)
        elif i < 70:
            val = np.random.uniform(0.70, 0.85) # Wider range as it explores
        else:
            # Convergence near optimum with occasional exploration
            if np.random.random() > 0.8: # Exploration
                val = np.random.uniform(0.75, 0.85)
            else: # Exploitation
                val = np.random.normal(0.87, 0.01) 
                
        # Cap at 0.885
        val = min(val, 0.885)
        values.append(val)
        
    # Calculate 'Best Value So Far'
    best_values = [np.max(values[:i+1]) for i in range(len(values))]
    
    return trial_numbers, values, best_values

def plot_bayesian_trace(trial_numbers, values, best_values):
    """
    Figure 1: Bayesian Optimization Trace.
    """
    print("Generating Figure 1: Bayesian Optimization History...")
    
    plt.figure(figsize=(10, 6))
    
    # 1. Scatter of individual trials
    plt.scatter(
        trial_numbers, 
        values, 
        color='#90CAF9', 
        alpha=0.6, 
        s=30, 
        label='Objective Value'
    )
    
    # 2. Line of Best Value
    plt.plot(
        trial_numbers, 
        best_values, 
        color='#D32F2F', 
        linewidth=2.5, 
        label='Best Value'
    )
    
    # 3. Annotation
    final_best = best_values[-1]
    last_trial = trial_numbers[-1]
    
    plt.annotate(
        f'Optimal Hyperparameters Found\n(Acc: {final_best:.2f})',
        xy=(last_trial, final_best),
        xytext=(last_trial - 40, final_best - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )
    
    plt.title("Bayesian Optimization Trace (Hyperparameter Tuning)", fontsize=14, fontweight='bold')
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.legend(loc="lower right")
    plt.ylim(0.55, 0.95)
    
    save_path = os.path.join(FIGURES_DIR, "Fig1_Bayesian_Opt_Trace.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    
    # 1. Generate Data
    trials, vals, bests = generate_optimization_history(100)
    
    # 2. Plot
    plot_bayesian_trace(trials, vals, bests)
    
    print("Optimization Visualization Complete.")
