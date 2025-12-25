import os

import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_preprocessing.sam import SAM
from main import train, validate
from models.PosterV2_7cls import pyramid_trans_expr2


# --- Configuration ---
class Config:
    def __init__(self):
        self.data = '../archive/DATASET'
        self.workers = 4
        self.epochs = 30  # Fewer epochs for tuning (speed)
        self.batch_size = 64
        self.gpu = '0'
        self.checkpoint_path = '../checkpoint/optuna_temp.pth'
        self.best_checkpoint_path = '../checkpoint/optuna_best.pth'


def objective(trial):
    # --- TAILORED SEARCH SPACE ---

    # Known best: 3.5e-5.
    # Search Range: 1.0e-5 to 9.0e-5
    lr = trial.suggest_float("lr", 1e-5, 9e-5, log=True)

    # Known best: 5e-4.
    # Search Range: 1e-4 to 1e-3
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)

    # Known best: 0.05.
    # Search Range: 0.02 to 0.08 (Tightened)
    rho = trial.suggest_float("rho", 0.02, 0.08)

    # Known best: 16. Mamba state size.
    # We test 16 vs 32 to see if "bigger is better"
    d_state = trial.suggest_categorical("d_state", [16, 32])

    print(f"\n=== TRIAL {trial.number} START ===")
    print(f"Params: LR={lr:.2e}, WD={weight_decay:.2e}, RHO={rho:.2f}, STATE={d_state}")

    # 2. Setup Model & Data
    # ---------------------
    args = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Pass the tuned 'd_state' to the model
    model = pyramid_trans_expr2(img_size=224, num_classes=7, d_state=d_state)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    # SAM Optimizer with tuned 'lr', 'rho', 'weight_decay'
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=rho, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data Loading (Simplified)
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

    # Define transforms (Same as main.py)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(traindir, train_tf)
    test_dataset = datasets.ImageFolder(testdir, val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # 3. Training Loop
    # ----------------
    patience = 5  # Stop if no improvement for 5 epochs
    counter = 0  # Counts how many bad epochs in a row
    best_trial_acc = 0.0  # Tracks best accuracy for THIS trial only
    margin = 0.01  # Minimum improvement to reset counter

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc, val_loss, _, _, _, _ = validate(val_loader, model, criterion, args)

        scheduler.step()

        # 1. OPTUNA PRUNING (Competition Check)
        # "Am I doing worse than the other trials?"
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 2. STANDARD EARLY STOPPING (Convergence Check)
        # "Have I stopped learning?"
        if val_acc > best_trial_acc and val_acc - best_trial_acc > margin:
            best_trial_acc = val_acc
            counter = 0  # Reset counter if we improved
        else:
            counter += 1  # Increment if we didn't improve

        if counter >= patience:
            print(f"Trial {trial.number} stopped early at epoch {epoch} (Converged).")
            # We return the best score achieved before stopping
            return best_trial_acc

    return best_trial_acc


def generate_plots(study):
    """Generates graphs for every parameter tuned"""
    print("Generating Analysis Graphs...")
    if not os.path.exists('hpo_plots'):
        os.makedirs('hpo_plots')

    # Convert study to pandas DataFrame
    df = study.trials_dataframe()

    # Clean up column names (remove 'params_' prefix)
    df.columns = [col.replace('params_', '') for col in df.columns]

    # Filter only completed trials
    df = df[df.state == 'COMPLETE']

    # Save raw data for your presentation
    df.to_csv('optuna_results.csv', index=False)
    print("Saved results to 'optuna_results.csv'")

    # List of params we tuned
    params = ['lr', 'weight_decay', 'rho', 'd_state']

    # Set plot style
    sns.set_theme(style="whitegrid")

    for param in params:
        plt.figure(figsize=(10, 6))

        # Scatter plot: Parameter Value vs Accuracy
        if param == 'd_state':
            # Categorical/Discrete plot
            sns.boxplot(x=param, y='value', data=df, palette='viridis')
            plt.title(f'Impact of Mamba State Size ({param}) on Accuracy')
        else:
            # Continuous plot (Log scale for LR/WD)
            sns.scatterplot(x=param, y='value', data=df, s=100, color='blue', alpha=0.7)
            if param in ['lr', 'weight_decay']:
                plt.xscale('log')
            plt.title(f'Impact of {param} on Accuracy')

        plt.ylabel('Validation Accuracy (%)')
        plt.xlabel(param)

        # Save plot
        save_path = f'hpo_plots/{param}_impact.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved graph: {save_path}")

    # Generate Parallel Coordinate Plot (The "All-in-One" visualization)
    try:
        from optuna.visualization import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_image("hpo_plots/all_params_parallel.png")
        print("Saved parallel coordinate plot.")
    except:
        print("Skipping interactive plot (need plotly/kaleido installed). Matplotlib plots are safe.")


if __name__ == "__main__":
    # Create the study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))

    # Enqueue a known good configuration as a baseline
    study.enqueue_trial({
        "lr": 3.5e-5,
        "weight_decay": 5e-4,
        "rho": 0.05,
        "d_state": 16
    })

    print("Starting Optimization (Baseline Enqueued)...")
    study.optimize(objective, n_trials=20)