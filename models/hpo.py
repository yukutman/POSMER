import os
import json
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


class Config:
    def __init__(self):
        self.data = '../archive/DATASET'
        self.workers = 4
        self.epochs = 50  # Fewer epochs for tuning (speed)
        self.batch_size = 64
        self.gpu = '0'
        self.checkpoint_path = '../checkpoint/optuna_temp.pth'
        self.best_checkpoint_path = '../checkpoint/optuna_best.pth'


def objective(trial):
    # search space
    lr = trial.suggest_float("lr", 1e-5, 9e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    rho = trial.suggest_float("rho", 0.02, 0.08)
    d_state = trial.suggest_categorical("d_state", [16, 32])

    print(f"\n=== TRIAL {trial.number} START ===")
    print(f"Params: LR={lr:.2e}, WD={weight_decay:.2e}, RHO={rho:.2f}, STATE={d_state}")

    # Setup Model and Data
    args = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = pyramid_trans_expr2(img_size=224, num_classes=7, d_state=d_state)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, rho=rho, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data Loading
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

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

    # Training Loop with Early Stopping
    patience = 3
    counter = 0
    best_trial_acc = 0.0
    margin = 0.05

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc, val_loss, _, _, _, _ = validate(val_loader, model, criterion, args)

        scheduler.step()

        # Pruning (Competition Check)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early Stopping (Convergence Check)
        if val_acc > best_trial_acc and (val_acc - best_trial_acc) > margin:
            best_trial_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Trial {trial.number} stopped early at epoch {epoch} (Converged).")
            return best_trial_acc

    return best_trial_acc


def save_results(study):
    """
    Saves:
    1. best_params.txt (Easy to read summary)
    2. optuna_results.csv (Raw data for all trials)
    3. hpo_plots/*.png (Visualizations)
    """
    print("\nProcessing Results...")

    # --- 1. Save Best Parameters to TXT ---
    print(f"Best Accuracy Found: {study.best_value:.4f}%")
    print(f"Best Params: {study.best_params}")

    with open('best_params.txt', 'w') as f:
        f.write("========================================\n")
        f.write("       OPTUNA OPTIMIZATION RESULTS      \n")
        f.write("========================================\n\n")
        f.write(f"Best Validation Accuracy: {study.best_value:.4f}%\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  - {key}: {value}\n")
        f.write("\n========================================\n")
    print("-> Saved 'best_params.txt'")

    # --- 2. Save Raw Data to CSV ---
    df = study.trials_dataframe()
    # Clean up column names
    df.columns = [col.replace('params_', '') for col in df.columns]
    # Filter only completed trials
    df_completed = df[df.state == 'COMPLETE']
    df_completed.to_csv('optuna_results.csv', index=False)
    print("-> Saved 'optuna_results.csv'")

    # --- 3. Generate Plots ---
    if not os.path.exists('hpo_plots'):
        os.makedirs('hpo_plots')

    # Set plot style
    sns.set_theme(style="whitegrid")
    params = ['lr', 'weight_decay', 'rho', 'd_state']

    for param in params:
        try:
            plt.figure(figsize=(10, 6))
            if param == 'd_state':
                sns.boxplot(x=param, y='value', data=df_completed, palette='viridis')
            else:
                sns.scatterplot(x=param, y='value', data=df_completed, s=100, color='blue', alpha=0.7)
                if param in ['lr', 'weight_decay']:
                    plt.xscale('log')

            plt.title(f'Impact of {param} on Accuracy')
            plt.ylabel('Validation Accuracy (%)')
            plt.xlabel(param)

            save_path = f'hpo_plots/{param}_impact.png'
            plt.savefig(save_path)
            plt.close()
            print(f"-> Saved graph: {save_path}")
        except Exception as e:
            print(f"Could not plot {param}: {e}")

    # Parallel Coordinate Plot
    try:
        from optuna.visualization import plot_parallel_coordinate
        # Note: writing static images requires 'kaleido' package installed
        # pip install -U kaleido
        fig = plot_parallel_coordinate(study)
        fig.write_image("hpo_plots/all_params_parallel.png")
        print("-> Saved parallel coordinate plot.")
    except Exception:
        pass  # Silently fail if plotly/kaleido not installed.


if __name__ == "__main__":
    # Create the study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))

    # Enqueue baseline
    study.enqueue_trial({
        "lr": 3.5e-5,
        "weight_decay": 5e-4,
        "rho": 0.05,
        "d_state": 16
    })

    print("Starting Optimization...")

    try:
        study.optimize(objective, n_trials=10)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Saving current results...")

    # Call the save function at the end
    save_results(study)

    print("\nDone! Check 'best_params.txt' for your winning config.")