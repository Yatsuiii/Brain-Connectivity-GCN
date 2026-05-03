"""
Population graph construction for subject-level GCN (Parisot et al. 2017/2018).

Nodes  = subjects
Edges  = phenotypic similarity (sex match × age Gaussian kernel)
Features = PCA-reduced FC upper triangle, fitted on training subjects only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Phenotypic data
# ---------------------------------------------------------------------------

def load_phenotypic(pheno_csv: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Load ABIDE phenotypic CSV and filter to subjects with processed .npz files."""
    pheno = pd.read_csv(pheno_csv)
    processed_dir = Path(processed_dir)

    available = {int(p.stem) for p in processed_dir.glob("*.npz")}
    pheno = pheno[pheno["SUB_ID"].isin(available)].copy().reset_index(drop=True)

    # DX_GROUP: 1=ASD → label=1, 2=TD → label=0
    pheno["label"] = (pheno["DX_GROUP"] == 1).astype(int)
    # SEX: 1=Male → 0, 2=Female → 1
    pheno["sex_enc"] = (pheno["SEX"] == 2).astype(int)

    return pheno


# ---------------------------------------------------------------------------
# Node features: FC upper triangle → PCA
# ---------------------------------------------------------------------------

def extract_fc_features(processed_dir: str | Path, subject_ids: list[int]) -> np.ndarray:
    """Load upper-triangle FC for each subject. Returns (N, 19900) float32."""
    processed_dir = Path(processed_dir)
    out = []
    for sid in subject_ids:
        data = np.load(processed_dir / f"{sid}.npz", allow_pickle=True)
        fc = data["mean_fc"].astype(np.float32)
        r, c = np.triu_indices(fc.shape[0], k=1)
        out.append(fc[r, c])
    return np.stack(out)


def harmonize_combat(
    features: np.ndarray,
    sites: list[str],
    labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
) -> np.ndarray:
    """ComBat site harmonization on FC upper triangle.

    Preserves biological signal (age, sex, diagnosis) while removing
    scanner-specific batch effects — the dominant noise source in multi-site
    fMRI (ABIDE has 17+ sites with different scanners and protocols).
    """
    from neuroCombat import neuroCombat

    # neuroCombat expects (features, subjects) — transpose
    data_T = features.T  # (19900, N)
    covars = pd.DataFrame({
        "site":  sites,
        "age":   ages,
        "sex":   sexes,
        "dx":    labels,
    })
    result = neuroCombat(
        dat=data_T,
        covars=covars,
        batch_col="site",
        continuous_cols=["age"],
        categorical_cols=["sex", "dx"],
    )
    return result["data"].T.astype(np.float32)   # back to (N, 19900)


def fit_pca(train_feats: np.ndarray, n_components: int = 256) -> tuple[StandardScaler, PCA]:
    """Fit StandardScaler + PCA on training features. Returns fitted objects."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    n_comp = min(n_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(train_scaled)
    var = pca.explained_variance_ratio_.sum()
    print(f"PCA {n_comp} components → {var:.1%} variance explained")
    return scaler, pca


def apply_pca(feats: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return pca.transform(scaler.transform(feats)).astype(np.float32)


# ---------------------------------------------------------------------------
# Population graph
# ---------------------------------------------------------------------------

def build_population_adj(
    subject_df: pd.DataFrame,
    threshold: float = 0.5,
    age_sigma: float | None = None,
    use_site: bool = False,
) -> np.ndarray:
    """Build N×N weighted adjacency from phenotypic similarity.

    Edge weight = sex_match * age_gaussian_sim (* site_match if use_site).
    Edge exists only if weight > threshold.

    Parameters
    ----------
    subject_df  : DataFrame with columns sex_enc, AGE_AT_SCAN, SITE_ID
    threshold   : minimum similarity to keep an edge
    age_sigma   : std dev for Gaussian age kernel (default: std of ages)
    use_site    : include site-match as a multiplier (Parisot original)
                  Disable after ComBat since site effects are removed.
    """
    N = len(subject_df)
    ages  = subject_df["AGE_AT_SCAN"].values.astype(np.float32)
    sexes = subject_df["sex_enc"].values

    if age_sigma is None:
        age_sigma = float(np.std(ages))

    # Age similarity — Gaussian kernel
    diff = ages[:, None] - ages[None, :]
    age_sim = np.exp(-diff**2 / (2 * age_sigma**2))

    # Sex similarity — binary match
    sex_sim = (sexes[:, None] == sexes[None, :]).astype(np.float32)

    W = sex_sim * age_sim

    if use_site:
        sites = np.array(subject_df["SITE_ID"].tolist())   # force plain object array
        site_sim = (sites[:, None] == sites[None, :]).astype(np.float32)
        W = W * site_sim

    adj = np.where(W > threshold, W, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    n_edges = int((adj > 0).sum()) // 2
    density = n_edges / (N * (N - 1) / 2)
    print(f"Population graph: {N} nodes, {n_edges} edges, {density:.1%} density")
    return adj


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization with self-loops: D^{-1/2}(A+I)D^{-1/2}."""
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    return (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]).astype(np.float32)
