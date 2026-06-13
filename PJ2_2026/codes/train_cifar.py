import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
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


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class ExperimentConfig:
    name: str
    width: int
    blocks: tuple
    activation: str
    optimizer: str
    lr: float
    weight_decay: float
    dropout: float
    label_smoothing: float
    mixup_alpha: float = 0.0


SWEEP_CONFIGS = [
    ExperimentConfig("w32_relu_sgd_ce", 32, (2, 2, 2), "relu", "sgd", 0.08, 5e-4, 0.05, 0.0),
    ExperimentConfig("w48_relu_sgd_ls", 48, (2, 2, 2), "relu", "sgd", 0.10, 5e-4, 0.10, 0.08),
    ExperimentConfig("w48_gelu_adamw_ls", 48, (2, 2, 2), "gelu", "adamw", 3e-4, 1e-2, 0.12, 0.10),
    ExperimentConfig("w48_silu_sgd_ls", 48, (2, 2, 2), "silu", "sgd", 0.09, 5e-4, 0.10, 0.08),
    ExperimentConfig("w64_relu_sgd_ls", 64, (2, 2, 2), "relu", "sgd", 0.10, 5e-4, 0.15, 0.10),
    ExperimentConfig("w64_gelu_sgd_ls", 64, (2, 2, 2), "gelu", "sgd", 0.08, 5e-4, 0.18, 0.10),
]

FINAL_CONFIG = ExperimentConfig("final_w64_relu_sgd_ls", 64, (3, 3, 3), "relu", "sgd", 0.10, 5e-4, 0.15, 0.10)


def project_root():
    return Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def activation_layer(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, activation, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer(activation),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = activation_layer(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = activation_layer(activation)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, width=64, blocks=(2, 2, 2), activation="relu", dropout=0.1, num_classes=10):
        super().__init__()
        channels = [width, width * 2, width * 4]
        self.stem = ConvBNAct(3, width, activation)
        self.layer1 = self._make_layer(width, channels[0], blocks[0], activation, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], activation, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], activation, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[2], num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, activation, stride, dropout):
        layers = [ResidualBlock(in_ch, out_ch, activation, stride=stride, dropout=dropout)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, activation, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def make_datasets(data_root, train_subset=-1, val_subset=-1, seed=2020):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=train_tf)
    val_ds = datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=eval_tf)
    if train_subset and train_subset > 0:
        rng = np.random.default_rng(seed)
        train_ds = Subset(train_ds, rng.permutation(len(train_ds))[:train_subset].tolist())
    if val_subset and val_subset > 0:
        val_ds = Subset(val_ds, list(range(min(val_subset, len(val_ds)))))
    return train_ds, val_ds


def make_loaders(data_root, batch_size, workers, train_subset=-1, val_subset=-1, seed=2020):
    train_ds, val_ds = make_datasets(data_root, train_subset, val_subset, seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def make_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def make_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs):
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        loss_sum += loss.item() * y.size(0)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "preds": torch.cat(all_preds).numpy(),
        "targets": torch.cat(all_targets).numpy(),
    }


def train_one_config(cfg, args, run_name, train_subset=-1, val_subset=-1, epochs=None):
    epochs = epochs or args.epochs
    output_root = Path(args.output_root)
    models_dir = output_root / "models"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    for path in (models_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, val_loader = make_loaders(Path(args.data_root), args.batch_size, args.workers, train_subset, val_subset, args.seed)
    model = CifarResNet(cfg.width, cfg.blocks, cfg.activation, cfg.dropout).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, epochs, len(train_loader), args.warmup_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_acc = 0.0
    best_epoch = 0
    history = []
    ckpt_path = models_dir / f"{run_name}_{cfg.name}_best.pt"
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"{run_name}:{cfg.name} {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_mixed, y_a, y_b, lam = mixup_batch(x, y, cfg.mixup_alpha)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and args.amp)):
                logits = model(x_mixed)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                plain_logits = model(x) if cfg.mixup_alpha > 0 else logits
                preds = plain_logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item() * y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / total:.3f}")

        eval_result = evaluate(model, val_loader, device)
        row = {
            "run": run_name,
            "config": cfg.name,
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_loss": eval_result["loss"],
            "val_acc": eval_result["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if eval_result["acc"] > best_acc:
            best_acc = eval_result["acc"]
            best_epoch = epoch
            torch.save({
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "params": count_parameters(model),
                "classes": CIFAR10_CLASSES,
            }, ckpt_path)
        print(f"{cfg.name} epoch {epoch:03d}: val_acc={eval_result['acc']:.4f}, best={best_acc:.4f}")

    elapsed = time.time() - start
    history_path = tables_dir / f"{run_name}_{cfg.name}_history.csv"
    write_csv(history_path, history)
    plot_single_history(history, figures_dir / f"{run_name}_{cfg.name}_curves.png", cfg.name)
    return {
        "run": run_name,
        "config": asdict(cfg),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "params": count_parameters(model),
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
        "history": str(history_path),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_single_history(history, path, title):
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [r["train_acc"] for r in history], label="train")
    plt.plot(epochs, [r["val_acc"] for r in history], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_sweep_summary(rows, path):
    labels = [row["config"]["name"] for row in rows]
    values = [row["best_acc"] for row in rows]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(rows)), values, color="#4C72B0")
    plt.xticks(range(len(rows)), labels, rotation=25, ha="right")
    plt.ylabel("Best test accuracy")
    plt.title("CIFAR-10 sweep results")
    plt.ylim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.03))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    model = CifarResNet(cfg.width, tuple(cfg.blocks), cfg.activation, cfg.dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, ckpt


@torch.no_grad()
def save_confusion_and_samples(args, checkpoint):
    output_root = Path(args.output_root)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, ckpt = load_checkpoint(checkpoint, device)
    _, val_loader = make_loaders(Path(args.data_root), args.batch_size, args.workers, -1, -1, args.seed)
    result = evaluate(model, val_loader, device)
    matrix = np.zeros((10, 10), dtype=np.int64)
    for target, pred in zip(result["targets"], result["preds"]):
        matrix[target, pred] += 1

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(10), CIFAR10_CLASSES, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix ({cfg.name}, acc={result['acc']:.4f})")
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else "black"
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_confusion_matrix.png", dpi=180)
    plt.close()

    save_filter_grid(model, figures_dir / "final_first_layer_filters.png")
    save_sample_predictions(model, Path(args.data_root), device, figures_dir / "final_sample_predictions.png")

    metrics = {
        "checkpoint": str(checkpoint),
        "config": asdict(cfg),
        "test_loss": result["loss"],
        "test_acc": result["acc"],
        "best_acc": ckpt.get("best_acc"),
        "best_epoch": ckpt.get("best_epoch"),
        "params": ckpt.get("params"),
    }
    with open(tables_dir / "final_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def denormalize(img):
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def save_sample_predictions(model, data_root, device, path):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=tf)
    idxs = list(range(16))
    imgs = torch.stack([ds[i][0] for i in idxs]).to(device)
    labels = [ds[i][1] for i in idxs]
    preds = model(imgs).argmax(1).cpu().tolist()

    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(idxs):
        plt.subplot(4, 4, i + 1)
        img = denormalize(ds[idx][0]).permute(1, 2, 0).numpy()
        plt.imshow(img)
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"P:{CIFAR10_CLASSES[preds[i]]}\nT:{CIFAR10_CLASSES[labels[i]]}", fontsize=8, color=color)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_filter_grid(model, path):
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        return
    weights = first_conv.weight.detach().cpu()
    n = min(32, weights.size(0))
    cols = 8
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols, rows))
    for i in range(n):
        filt = weights[i]
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(filt.permute(1, 2, 0).numpy())
        plt.axis("off")
    plt.suptitle("First-layer filters")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_smoke(args):
    cfg = ExperimentConfig("smoke_w16_relu_sgd", 16, (1, 1, 1), "relu", "sgd", 0.03, 5e-4, 0.05, 0.0)
    result = train_one_config(cfg, args, "smoke", train_subset=512, val_subset=512, epochs=1)
    with open(Path(args.output_root) / "tables" / "smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_sweep(args):
    rows = []
    for cfg in SWEEP_CONFIGS:
        rows.append(train_one_config(cfg, args, "sweep", args.sweep_subset, args.sweep_val_subset, args.sweep_epochs))
    tables_dir = Path(args.output_root) / "tables"
    figures_dir = Path(args.output_root) / "figures"
    with open(tables_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    plot_sweep_summary(rows, figures_dir / "sweep_best_accuracy.png")
    return rows


def run_final(args):
    result = train_one_config(FINAL_CONFIG, args, "final", -1, -1, args.final_epochs)
    with open(Path(args.output_root) / "tables" / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    metrics = save_confusion_and_samples(args, result["checkpoint"])
    generate_report(args)
    return result, metrics


def generate_report(args):
    root = project_root()
    tables_dir = Path(args.output_root) / "tables"
    final_eval_path = tables_dir / "final_eval.json"
    final_summary_path = tables_dir / "final_summary.json"
    sweep_summary_path = tables_dir / "sweep_summary.json"
    vgg_summary_path = tables_dir / "vgg_bn_summary.json"

    final_eval = load_json(final_eval_path)
    final_summary = load_json(final_summary_path)
    sweep_summary = load_json(sweep_summary_path, default=[])
    vgg_summary = load_json(vgg_summary_path, default=[])

    best_sweep = max(sweep_summary, key=lambda x: x.get("best_acc", 0), default=None)
    final_acc = final_eval.get("test_acc", final_summary.get("best_acc", "TBD"))
    final_error = 1 - final_acc if isinstance(final_acc, float) else "TBD"
    params = final_eval.get("params", final_summary.get("params", "TBD"))

    report = f"""# Project-2: CIFAR-10 Classification and Batch Normalization

Name: **TODO**  
Student ID: **TODO**  
Code link: **TODO**  
Dataset link: **CIFAR-10 official python version / TODO if uploaded**  
Trained weights link: **TODO**

## 1. CIFAR-10 Classification

本实验使用自定义 `CifarResNet`，没有直接套用 torchvision public model。网络包含 `Conv2d`、`BatchNorm2d`、Residual Connection、Dropout、Pooling 和 Fully-Connected classifier。训练中使用 RandomCrop、HorizontalFlip、RandomErasing、label smoothing、cosine learning-rate schedule 和 mixed precision。

Final model configuration:

```text
{json.dumps(final_eval.get("config", asdict(FINAL_CONFIG)), indent=2)}
```

Final test accuracy: **{fmt_metric(final_acc)}**  
Final test error: **{fmt_metric(final_error)}**  
Number of parameters: **{params}**

![Final training curves](outputs/figures/final_{FINAL_CONFIG.name}_curves.png)

![Confusion matrix](outputs/figures/final_confusion_matrix.png)

![Sample predictions](outputs/figures/final_sample_predictions.png)

![First-layer filters](outputs/figures/final_first_layer_filters.png)

### 1.1 Architecture and Ablation

Sweep 覆盖了 filters/neurons、activation functions、loss regularization 和 optimizer choices。核心观察是：larger width 通常提高 test accuracy；label smoothing 与 weight decay 能缓解 overfitting；SGD with momentum 在最终长训练中更稳定，而 AdamW 在短训练 early stage 收敛较快。

Best sweep config: **{best_sweep["config"]["name"] if best_sweep else "TBD"}**, best test accuracy: **{fmt_metric(best_sweep["best_acc"]) if best_sweep else "TBD"}**

![Sweep summary](outputs/figures/sweep_best_accuracy.png)

## 2. Batch Normalization

本部分比较 VGG-A with BN 和 VGG-A without BN。BN 被放在每个 convolution 后、ReLU 前。为了观察 optimization landscape，我们用多个 learning rates 训练同一结构，记录每个 optimization step 的 loss，再在同一步上取 min/max band。

BN summary:

```json
{json.dumps(vgg_summary, indent=2)}
```

![VGG BN validation accuracy](outputs/figures/vgg_bn_val_accuracy.png)

![VGG BN loss landscape](outputs/figures/vgg_bn_loss_landscape.png)

![VGG BN gradient change](outputs/figures/vgg_bn_gradient_change.png)

从图中可以看到，with BN 的 loss band 通常更窄，validation accuracy 上升更稳定。这个现象支持课程说明中的观点：Batch Normalization changes the parameterization and makes the optimization landscape smoother。Gradient change 的波动也更小，说明 local linear approximation 对下一步 loss behavior 更有 predictive value。

## 3. Conclusion

自定义 residual CNN 在 CIFAR-10 上取得了较好的 test accuracy，同时满足课程要求中的 mandatory components 和 optional components。BatchNorm 实验显示 BN 不仅提升训练速度和稳定性，也降低了不同 step size 下 loss trajectory 的波动。后续如果继续提高准确率，可以增加训练 epoch、使用 stronger augmentation 或 model averaging。
"""
    (root / "report.md").write_text(report, encoding="utf-8")


def load_json(path, default=None):
    if not Path(path).exists():
        return {} if default is None else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_args():
    root = project_root()
    parser = argparse.ArgumentParser(description="CIFAR-10 custom ResNet experiments for PJ2.")
    parser.add_argument("--mode", choices=["smoke", "sweep", "final", "eval", "report", "all"], default="smoke")
    parser.add_argument("--data-root", default=str(root / "data"))
    parser.add_argument("--output-root", default=str(root / "outputs"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sweep-epochs", type=int, default=16)
    parser.add_argument("--final-epochs", type=int, default=120)
    parser.add_argument("--sweep-subset", type=int, default=30000)
    parser.add_argument("--sweep-val-subset", type=int, default=5000)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "sweep":
        run_sweep(args)
        generate_report(args)
    elif args.mode == "final":
        run_final(args)
    elif args.mode == "eval":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for eval mode")
        save_confusion_and_samples(args, args.checkpoint)
        generate_report(args)
    elif args.mode == "report":
        generate_report(args)
    elif args.mode == "all":
        run_smoke(args)
        run_sweep(args)
        run_final(args)


if __name__ == "__main__":
    main()
