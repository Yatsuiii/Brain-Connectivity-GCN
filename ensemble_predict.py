"""
Final ensemble inference script: Fisher Z MLP + PCA-LR blend.

Best stable result on ABIDE I CC200 (seed=42, stratified split):
  - MLP ensemble (h=64, 5 ckpts): test_auc = 0.7235
  - PCA(100)+LR alone:            test_auc = 0.7167  (deterministic, canonical ordering)
  - Blend (alpha=0.02):           test_auc = 0.7329  (val-selected alpha)
  - Baseline fc_mlp:              test_auc = 0.7186
  - Improvement:                  +0.0143

Key design choices for reproducibility:
  - Training data extracted with shuffle=False (canonical _train_paths order)
    so PCA's randomized SVD gives identical results regardless of global random state
  - PCA random_state=42 fixed
  - Alpha selected by val_auc: 0.02

Usage:
    python ensemble_predict.py \
        --ckpt_dir checkpoints/fc_mlp_fisherz \
        --data_dir data

    # tune alpha manually
    python ensemble_predict.py --ckpt_dir checkpoints/fc_mlp_fisherz --alpha 0.02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from brain_gcn.tasks import ClassificationTask
from brain_gcn.utils.data.datamodule import ABIDEDataModule


def build_datamodule(args: argparse.Namespace) -> ABIDEDataModule:
    return ABIDEDataModule(
        data_dir=args.data_dir,
        n_subjects=None,
        window_len=50,
        step=3,
        max_windows=30,
        fc_threshold=0.2,
        use_dynamic_adj=False,
        use_dynamic_adj_sequence=False,
        use_population_adj=False,
        preserve_fc_sign=True,
        use_fc_variance=False,
        use_fisher_z=True,
        use_fc_degree_features=False,
        use_fc_row_features=False,
        n_pca_components=0,
        batch_size=256,
        val_ratio=0.1,
        test_ratio=0.1,
        split_strategy="stratified",
        val_site=None,
        test_site=None,
        num_workers=0,
        overwrite_cache=False,
        force_prepare=False,
    )


def extract_fc_upper(loader) -> tuple[np.ndarray, np.ndarray]:
    """Extract upper-triangle Fisher Z FC features + labels."""
    all_feat, all_lbl = [], []
    for batch in loader:
        adj, lbl = batch[1], batch[2]
        B, N, _ = adj.shape
        idx = torch.triu_indices(N, N, offset=1)
        all_feat.append(adj[:, idx[0], idx[1]].numpy())
        all_lbl.append(lbl.numpy())
    return np.concatenate(all_feat), np.concatenate(all_lbl)


def get_mlp_probs(
    ckpt_dir: Path,
    dm: ABIDEDataModule,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble softmax probabilities from all checkpoints in ckpt_dir."""
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    vprobs, tprobs = [], []
    for ckpt in ckpts:
        task = ClassificationTask.load_from_checkpoint(ckpt, map_location=device, strict=False)
        task.eval().to(device)
        vp, tp = [], []
        with torch.no_grad():
            for b in dm.val_dataloader():
                vp.append(torch.softmax(task(b[0].to(device), b[1].to(device)), -1).cpu())
            for b in dm.test_dataloader():
                tp.append(torch.softmax(task(b[0].to(device), b[1].to(device)), -1).cpu())
        vprobs.append(torch.cat(vp, 0)[:, 1].numpy())
        tprobs.append(torch.cat(tp, 0)[:, 1].numpy())
    return np.stack(vprobs).mean(0), np.stack(tprobs).mean(0)


def fit_pca_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 100,
    C: float = 0.01,
) -> tuple[StandardScaler, PCA, LogisticRegression]:
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_pca, y_train)
    return sc, pca, lr


def predict_pca_lr(sc, pca, lr, X: np.ndarray) -> np.ndarray:
    return lr.predict_proba(pca.transform(sc.transform(X)))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/fc_mlp_fisherz")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha", type=float, default=0.02,
        help="Weight for PCA-LR component (1-alpha for MLP). Val-selected optimum: 0.02.",
    )
    parser.add_argument("--n_pca", type=int, default=100)
    parser.add_argument("--lr_C", type=float, default=0.01)
    parser.add_argument("--accelerator", type=str, default="auto")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    dm = build_datamodule(args)
    dm.prepare_data()
    dm.setup()

    device = "cuda" if args.accelerator != "cpu" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Test set: {len(list(dm._test_paths))} subjects")

    # MLP ensemble
    ckpt_dir = Path(args.ckpt_dir)
    print(f"\nLoading MLP checkpoints from {ckpt_dir} ...")
    mlp_val, mlp_test = get_mlp_probs(ckpt_dir, dm, device=device)

    # PCA-LR — use non-shuffled canonical loader so PCA is reproducible
    print(f"Fitting PCA({args.n_pca})+LR(C={args.lr_C}) on canonical training order ...")
    train_ds = dm._make_dataset(dm._train_paths)
    train_loader_fixed = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
    X_train, y_train = extract_fc_upper(train_loader_fixed)
    X_val,   y_val   = extract_fc_upper(dm.val_dataloader())
    X_test,  y_test  = extract_fc_upper(dm.test_dataloader())

    sc, pca, lr = fit_pca_lr(X_train, y_train, n_components=args.n_pca, C=args.lr_C)
    pca_val  = predict_pca_lr(sc, pca, lr, X_val)
    pca_test = predict_pca_lr(sc, pca, lr, X_test)

    # Blend
    alpha = args.alpha
    blend_val  = alpha * pca_val  + (1 - alpha) * mlp_val
    blend_test = alpha * pca_test + (1 - alpha) * mlp_test

    # Report
    auc = BinaryAUROC()
    lv, lt = torch.tensor(y_val), torch.tensor(y_test)

    mlp_val_auc   = auc(torch.tensor(mlp_val),   lv).item()
    mlp_test_auc  = auc(torch.tensor(mlp_test),  lt).item()
    pca_val_auc   = auc(torch.tensor(pca_val),   lv).item()
    pca_test_auc  = auc(torch.tensor(pca_test),  lt).item()
    blend_val_auc = auc(torch.tensor(blend_val), lv).item()
    blend_test_auc = auc(torch.tensor(blend_test), lt).item()

    print(f"\n{'Component':<35} {'val_AUC':>8}  {'test_AUC':>8}")
    print("-" * 55)
    print(f"{'MLP ensemble (Fisher Z h=64)':<35} {mlp_val_auc:>8.4f}  {mlp_test_auc:>8.4f}")
    print(f"{'PCA({}) + LR (Fisher Z)'.format(args.n_pca):<35} {pca_val_auc:>8.4f}  {pca_test_auc:>8.4f}")
    print(f"{'Blend (alpha={:.2f})'.format(alpha):<35} {blend_val_auc:>8.4f}  {blend_test_auc:>8.4f}")
    print(f"{'Baseline fc_mlp':<35} {'0.7250':>8}  {'0.7186':>8}")
    print(f"\nImprovement over baseline: +{blend_test_auc - 0.7186:.4f} test AUC")


if __name__ == "__main__":
    main()
