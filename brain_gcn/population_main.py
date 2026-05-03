"""
Population Graph GCN — training entry point.

Architecture: Parisot et al. 2017/2018 (subject nodes, phenotypic edges).
  - Nodes  : subjects (N ≈ 1102)
  - Features: PCA-reduced FC upper triangle (D=256)
  - Edges  : sex_match × age_gaussian_similarity > threshold
  - Training: transductive — all nodes in graph, loss masked to train split

Usage
-----
    python -m brain_gcn.population_main \\
        --data_dir data \\
        --pheno_csv data/raw/abide_s3/phenotypic.csv \\
        --use_combat \\
        --n_pca 256 \\
        --hidden_dim 64 \\
        --dropout 0.5 \\
        --lr 5e-4 \\
        --weight_decay 1e-3 \\
        --epochs 500 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryRecall, BinarySpecificity, BinaryF1Score

from brain_gcn.models.population_gcn import PopulationGCN
from brain_gcn.utils.data.population_graph import (
    apply_pca,
    build_population_adj,
    extract_fc_features,
    fit_pca,
    harmonize_combat,
    load_phenotypic,
    normalize_adj,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    n_td  = int((labels == 0).sum())
    n_asd = int((labels == 1).sum())
    total = n_td + n_asd
    return torch.tensor([total / (2.0 * n_td), total / (2.0 * n_asd)], dtype=torch.float32)


def build_masks(n: int, train_idx, val_idx, test_idx, device):
    def _mask(idx):
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[idx] = True
        return m
    return _mask(train_idx), _mask(val_idx), _mask(test_idx)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    probs  = torch.softmax(logits[mask], dim=-1)
    preds  = probs.argmax(dim=-1)
    tgts   = labels[mask]

    auc_m  = BinaryAUROC()
    acc_m  = BinaryAccuracy()
    sens_m = BinaryRecall()
    spec_m = BinarySpecificity()
    f1_m   = BinaryF1Score()

    auc  = auc_m(probs[:, 1].cpu(),  tgts.cpu()).item()
    acc  = acc_m(preds.cpu(),         tgts.cpu()).item()
    sens = sens_m(preds.cpu(),        tgts.cpu()).item()
    spec = spec_m(preds.cpu(),        tgts.cpu()).item()
    f1   = f1_m(preds.cpu(),          tgts.cpu()).item()
    return dict(auc=auc, acc=acc, sens=sens, spec=spec, f1=f1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    processed_dir = Path(args.data_dir) / "processed"
    pheno = load_phenotypic(args.pheno_csv, processed_dir)
    print(f"Subjects matched: {len(pheno)}  (ASD={pheno['label'].sum()}  TD={(pheno['label']==0).sum()})")

    subject_ids = pheno["SUB_ID"].tolist()
    labels_np   = pheno["label"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # 2. Train / val / test split (stratified)
    # ------------------------------------------------------------------
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    train_val_idx, test_idx = next(sss.split(subject_ids, labels_np))

    val_size = args.val_ratio / (1.0 - args.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=args.seed)
    rel_train, rel_val = next(sss2.split(train_val_idx, labels_np[train_val_idx]))
    train_idx = train_val_idx[rel_train]
    val_idx   = train_val_idx[rel_val]

    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ------------------------------------------------------------------
    # 3. FC features
    # ------------------------------------------------------------------
    print("Loading FC features …")
    all_feats = extract_fc_features(processed_dir, subject_ids)  # (N, 19900)

    if args.use_combat:
        print("Running ComBat harmonization …")
        all_feats = harmonize_combat(
            features=all_feats,
            sites=pheno["SITE_ID"].tolist(),
            labels=labels_np,
            ages=pheno["AGE_AT_SCAN"].values,
            sexes=pheno["sex_enc"].values,
        )

    # PCA fitted on training subjects only
    scaler, pca = fit_pca(all_feats[train_idx], n_components=args.n_pca)
    all_feats_pca = apply_pca(all_feats, scaler, pca)             # (N, n_pca)

    # ------------------------------------------------------------------
    # 4. Population graph
    # ------------------------------------------------------------------
    print("Building population graph …")
    adj_np = build_population_adj(
        pheno,
        threshold=args.graph_threshold,
        use_site=args.use_site_edges,
    )
    adj_norm = torch.FloatTensor(normalize_adj(adj_np)).to(device)

    # ------------------------------------------------------------------
    # 5. Tensors
    # ------------------------------------------------------------------
    X      = torch.FloatTensor(all_feats_pca).to(device)          # (N, D)
    labels = torch.LongTensor(labels_np).to(device)               # (N,)
    cw     = class_weights(labels_np).to(device)
    N      = len(subject_ids)
    train_mask, val_mask, test_mask = build_masks(N, train_idx, val_idx, test_idx, device)

    # ------------------------------------------------------------------
    # 6. Model
    # ------------------------------------------------------------------
    model = PopulationGCN(
        in_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cosine_t0, T_mult=2, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    best_val_auc  = 0.0
    best_state    = None
    patience_left = args.patience

    print(f"\n{'ep':>5s} | {'tr_loss':>8s} | {'val_auc':>8s} | {'val_acc':>8s} | {'val_sens':>9s} | {'val_spec':>9s}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        optimizer.zero_grad()
        logits = model(X, adj_norm)
        loss   = F.cross_entropy(logits[train_mask], labels[train_mask], weight=cw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            logits_eval = model(X, adj_norm)
        val_m = evaluate(logits_eval, labels, val_mask)

        if val_m["auc"] > best_val_auc:
            best_val_auc  = val_m["auc"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"{epoch:>5d} | {loss.item():>8.4f} | {val_m['auc']:>8.4f} | "
                f"{val_m['acc']:>8.4f} | {val_m['sens']:>9.4f} | {val_m['spec']:>9.4f}"
            )

        if patience_left <= 0:
            print(f"\nEarly stop at epoch {epoch}. Best val_auc={best_val_auc:.4f}")
            break

    # ------------------------------------------------------------------
    # 8. Test
    # ------------------------------------------------------------------
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits_final = model(X, adj_norm)
    test_m = evaluate(logits_final, labels, test_mask)

    print(f"\n{'='*60}")
    print(f"[TEST]  auc={test_m['auc']:.4f}  acc={test_m['acc']:.4f}  "
          f"sens={test_m['sens']:.4f}  spec={test_m['spec']:.4f}  f1={test_m['f1']:.4f}")
    print(f"{'='*60}")

    # Save checkpoint
    ckpt_dir = Path("checkpoints") / "population_gcn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_auc{best_val_auc:.3f}.pt"
    torch.save({"model_state": best_state, "args": vars(args), "test_metrics": test_m}, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    return test_m


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Population Graph GCN for ABIDE ASD classification")
    p.add_argument("--data_dir",        type=str,   default="data")
    p.add_argument("--pheno_csv",       type=str,   default="data/raw/abide_s3/phenotypic.csv")
    p.add_argument("--use_combat",      action="store_true", help="Apply ComBat site harmonization")
    p.add_argument("--use_site_edges",  action="store_true", help="Include site-match in graph edges")
    p.add_argument("--n_pca",           type=int,   default=256)
    p.add_argument("--graph_threshold", type=float, default=0.5)
    p.add_argument("--hidden_dim",      type=int,   default=64)
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-3)
    p.add_argument("--cosine_t0",       type=int,   default=100)
    p.add_argument("--epochs",          type=int,   default=500)
    p.add_argument("--patience",        type=int,   default=60)
    p.add_argument("--val_ratio",       type=float, default=0.1)
    p.add_argument("--test_ratio",      type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=42)
    return p


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
