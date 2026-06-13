import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_random_seeds(seed_value=2020):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = True


def default_project_root():
    return Path(__file__).resolve().parents[2]


def make_loader(data_root, train, batch_size, workers, subset=-1, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=str(data_root), train=train, download=False, transform=transform)
    if subset and subset > 0:
        dataset = Subset(dataset, list(range(min(subset, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def last_layer_grad(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            return module.weight.grad.detach().flatten().float().cpu()
    return torch.empty(0)


def train_one_run(model_name, lr, args, loaders, device, run_dir):
    model_cls = VGG_A_BatchNorm if model_name == "bn" else VGG_A
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    step_losses = []
    grad_norms = []
    grad_changes = []
    history = []
    best_acc = 0.0
    prev_grad = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loaders["train"], desc=f"{model_name} lr={lr:g} epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            grad = last_layer_grad(model)
            if grad.numel():
                grad_norms.append(float(torch.linalg.vector_norm(grad)))
                if prev_grad is not None:
                    denom = torch.linalg.vector_norm(prev_grad).item() + 1e-12
                    grad_changes.append(float(torch.linalg.vector_norm(grad - prev_grad).item() / denom))
                else:
                    grad_changes.append(0.0)
                prev_grad = grad

            optimizer.step()

            step_losses.append(loss.item())
            train_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_acc = correct / total
        val_acc, val_loss = get_accuracy(model, loaders["val"], device)
        best_acc = max(best_acc, val_acc)
        history.append({
            "model": model_name,
            "lr": lr,
            "epoch": epoch,
            "train_loss": train_loss / total,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    elapsed = time.time() - start
    ckpt_path = run_dir / f"vgg_{model_name}_lr_{lr:g}.pt"
    torch.save({
        "model": model_name,
        "lr": lr,
        "state_dict": model.state_dict(),
        "best_val_acc": best_acc,
        "params": get_number_of_parameters(model),
        "elapsed_sec": elapsed,
    }, ckpt_path)

    result = {
        "model": model_name,
        "lr": lr,
        "best_val_acc": best_acc,
        "params": get_number_of_parameters(model),
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "grad_changes": grad_changes,
        "history": history,
    }
    np.save(run_dir / f"losses_{model_name}_lr_{lr:g}.npy", np.asarray(step_losses, dtype=np.float32))
    np.save(run_dir / f"grad_norms_{model_name}_lr_{lr:g}.npy", np.asarray(grad_norms, dtype=np.float32))
    np.save(run_dir / f"grad_changes_{model_name}_lr_{lr:g}.npy", np.asarray(grad_changes, dtype=np.float32))
    return result


def write_history_csv(results, path):
    rows = [row for result in results for row in result["history"]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "lr", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)


def compute_band(curves):
    min_len = min(len(c) for c in curves if len(c))
    arr = np.asarray([c[:min_len] for c in curves], dtype=np.float32)
    return arr.min(axis=0), arr.max(axis=0), arr.mean(axis=0)


def plot_loss_landscape(results, figures_dir):
    grouped = {"no_bn": [], "bn": []}
    for result in results:
        grouped[result["model"]].append(result["step_losses"])

    plt.figure(figsize=(9, 5))
    colors = {"no_bn": "#C44E52", "bn": "#4C72B0"}
    labels = {"no_bn": "VGG-A without BN", "bn": "VGG-A with BN"}
    for model_name, curves in grouped.items():
        if not curves:
            continue
        low, high, mean = compute_band(curves)
        x = np.arange(len(mean))
        plt.plot(x, mean, color=colors[model_name], label=labels[model_name])
        plt.fill_between(x, low, high, color=colors[model_name], alpha=0.18)
    plt.xlabel("Optimization step")
    plt.ylabel("Training loss")
    plt.title("Loss landscape band across learning rates")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "vgg_bn_loss_landscape.png", dpi=180)
    plt.close()


def plot_training_curves(results, figures_dir):
    plt.figure(figsize=(10, 4))
    for result in results:
        epochs = [row["epoch"] for row in result["history"]]
        vals = [row["val_acc"] for row in result["history"]]
        plt.plot(epochs, vals, label=f"{result['model']} lr={result['lr']:g}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("VGG-A BN comparison")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "vgg_bn_val_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    for result in results:
        if not result["grad_changes"]:
            continue
        y = np.asarray(result["grad_changes"], dtype=np.float32)
        if len(y) > 50:
            kernel = np.ones(25) / 25
            y = np.convolve(y, kernel, mode="valid")
        plt.plot(y, label=f"{result['model']} lr={result['lr']:g}")
    plt.xlabel("Optimization step")
    plt.ylabel("Relative last-layer gradient change")
    plt.title("Gradient predictiveness proxy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "vgg_bn_gradient_change.png", dpi=180)
    plt.close()


def run(args):
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    figures_dir = output_root / "figures"
    models_dir = output_root / "models"
    tables_dir = output_root / "tables"
    for path in (figures_dir, models_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    set_random_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = {
        "train": make_loader(data_root, True, args.batch_size, args.workers, args.train_subset, True),
        "val": make_loader(data_root, False, args.batch_size, args.workers, args.val_subset, False),
    }

    models = args.models
    results = []
    run_dir = models_dir / "vgg_bn"
    run_dir.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        for lr in args.lrs:
            results.append(train_one_run(model_name, lr, args, loaders, device, run_dir))

    summary = []
    for result in results:
        row = {k: result[k] for k in ("model", "lr", "best_val_acc", "params", "elapsed_sec", "checkpoint")}
        summary.append(row)

    with open(tables_dir / "vgg_bn_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_history_csv(results, tables_dir / "vgg_bn_history.csv")
    plot_loss_landscape(results, figures_dir)
    plot_training_curves(results, figures_dir)

    print(json.dumps({
        "device": str(device),
        "project_root": str(project_root),
        "data_root": str(data_root),
        "summary": summary,
    }, indent=2))


def parse_args():
    project_root = default_project_root()
    parser = argparse.ArgumentParser(description="Train VGG-A with/without BN and plot loss landscape.")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--data-root", default=str(project_root / "data"))
    parser.add_argument("--output-root", default=str(project_root / "outputs"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--val-subset", type=int, default=2000)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 2e-3, 5e-4, 1e-4])
    parser.add_argument("--models", nargs="+", choices=["no_bn", "bn"], default=["no_bn", "bn"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
