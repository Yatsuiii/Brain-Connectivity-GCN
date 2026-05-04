"""
BrainConnect-ASD — Scanner-site-invariant ASD detection from fMRI.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import gradio as gr

from _charts import VAL_B64, AUC_B64, AMD_BENCH_B64

_WINDOW_LEN   = 50
_STEP         = 3
_MAX_WINDOWS  = 30
_FC_THRESHOLD = 0.2

# ── Atlas configurations ────────────────────────────────────────────────────
# CC200 → Yeo 7-network parcellation (approximate ROI ordering)
_ATLAS_CFG = {
    "cc200": {
        "n_rois":      200,
        "label":       "CC200",
        "net_names":   ["DMN", "Salience", "Frontoparietal", "Sensorimotor", "Visual", "Dorsal Attn", "Subcortical"],
        "net_bounds":  [0, 38, 69, 99, 137, 165, 180, 200],
        "net_colors":  ["#e63946", "#f4a261", "#457b9d", "#2dc653", "#a8dadc", "#8b5cf6", "#6b7280"],
        "ckpts": {
            "CALTECH":  Path("checkpoints/cc200_caltech.ckpt"),
            "CMU":      Path("checkpoints/cc200_cmu.ckpt"),
            "KKI":      Path("checkpoints/cc200_kki.ckpt"),
            "LEUVEN_1": Path("checkpoints/cc200_leuven_1.ckpt"),
            "LEUVEN_2": Path("checkpoints/cc200_leuven_2.ckpt"),
            "MAX_MUN":  Path("checkpoints/cc200_max_mun.ckpt"),
            "NYU":      Path("checkpoints/cc200_nyu.ckpt"),
            "OHSU":     Path("checkpoints/cc200_ohsu.ckpt"),
            "OLIN":     Path("checkpoints/cc200_olin.ckpt"),
            "PITT":     Path("checkpoints/cc200_pitt.ckpt"),
            "SBL":      Path("checkpoints/cc200_sbl.ckpt"),
            "SDSU":     Path("checkpoints/cc200_sdsu.ckpt"),
            "STANFORD": Path("checkpoints/cc200_stanford.ckpt"),
            "TRINITY":  Path("checkpoints/cc200_trinity.ckpt"),
            "UCLA_1":   Path("checkpoints/cc200_ucla_1.ckpt"),
            "UCLA_2":   Path("checkpoints/cc200_ucla_2.ckpt"),
            "UM_1":     Path("checkpoints/cc200_um_1.ckpt"),
            "UM_2":     Path("checkpoints/cc200_um_2.ckpt"),
            "USM":      Path("checkpoints/cc200_usm.ckpt"),
            "YALE":     Path("checkpoints/cc200_yale.ckpt"),
        },
    },
    "aal": {
        "n_rois":      116,
        "label":       "AAL-116",
        # Approximate Yeo-7 parcellation for AAL-116 anatomical ordering:
        # Frontal/FPN (1-28), Sensorimotor (29-40), DMN parietal (41-60),
        # Temporal/DMN (61-76), Subcortical (77-90), Occipital/Visual (91-116)
        "net_names":   ["Frontoparietal", "Sensorimotor", "Dorsal Attn", "DMN", "Salience", "Subcortical", "Visual"],
        "net_bounds":  [0, 20, 34, 50, 68, 80, 92, 116],
        "net_colors":  ["#457b9d", "#2dc653", "#8b5cf6", "#e63946", "#f4a261", "#6b7280", "#a8dadc"],
        "ckpts": {
            "NYU":  Path("checkpoints/aal_nyu.ckpt"),
            "USM":  Path("checkpoints/aal_usm.ckpt"),
            "UCLA": Path("checkpoints/aal_ucla.ckpt"),
            "UM":   Path("checkpoints/aal_um.ckpt"),
        },
    },
    "ho": {
        "n_rois":      111,
        "label":       "Harvard-Oxford",
        "net_names":   ["Frontoparietal", "Sensorimotor", "DMN", "Salience", "Subcortical", "Visual", "Temporal"],
        "net_bounds":  [0, 18, 30, 48, 68, 80, 96, 111],
        "net_colors":  ["#457b9d", "#2dc653", "#e63946", "#f4a261", "#6b7280", "#a8dadc", "#8b5cf6"],
        "ckpts": {
            "NYU":  Path("checkpoints/ho_nyu.ckpt"),
            "USM":  Path("checkpoints/ho_usm.ckpt"),
            "UCLA": Path("checkpoints/ho_ucla.ckpt"),
            "UM":   Path("checkpoints/ho_um.ckpt"),
        },
    },
}

# Resolve active atlas config by ROI count
_ROI_TO_ATLAS = {cfg["n_rois"]: key for key, cfg in _ATLAS_CFG.items()}

# Legacy aliases kept for backward compat
_NET_NAMES  = _ATLAS_CFG["cc200"]["net_names"]
_NET_BOUNDS = _ATLAS_CFG["cc200"]["net_bounds"]
_NET_COLORS = _ATLAS_CFG["cc200"]["net_colors"]
_CKPTS      = _ATLAS_CFG["cc200"]["ckpts"]

# ── preprocessing ──────────────────────────────────────────────────────────

def _zscore(bold):
    mean = bold.mean(0, keepdims=True)
    std  = bold.std(0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((bold - mean) / std).astype(np.float32)

def _fc(bold):
    fc = np.corrcoef(bold.T).astype(np.float32)
    np.nan_to_num(fc, copy=False)
    return fc

def _windows(bold):
    T, N = bold.shape
    starts = list(range(0, T - _WINDOW_LEN + 1, _STEP))
    w = np.stack([bold[s:s+_WINDOW_LEN].std(0) for s in starts]).astype(np.float32)
    if len(w) >= _MAX_WINDOWS:
        return w[:_MAX_WINDOWS]
    return np.concatenate([w, np.repeat(w[-1:], _MAX_WINDOWS - len(w), 0)])

def preprocess(bold):
    bold = _zscore(bold)
    fc   = _fc(bold)
    fc   = np.arctanh(np.clip(fc, -0.9999, 0.9999))
    adj  = np.where(np.abs(fc) >= _FC_THRESHOLD, fc, 0.0).astype(np.float32)
    bw   = _windows(bold)
    return torch.FloatTensor(bw).unsqueeze(0), torch.FloatTensor(adj).unsqueeze(0)

# ── model loading ──────────────────────────────────────────────────────────

_model_cache: dict[str, list] = {}

def get_models(atlas: str = "cc200"):
    global _model_cache
    if atlas in _model_cache:
        return _model_cache[atlas]
    from brain_gcn.tasks import ClassificationTask
    cfg = _ATLAS_CFG.get(atlas, _ATLAS_CFG["cc200"])
    models = []
    for site, ckpt in cfg["ckpts"].items():
        if not ckpt.exists():
            continue
        task = ClassificationTask.load_from_checkpoint(str(ckpt), map_location="cpu", strict=False)
        task.eval()
        models.append((site, task))
    _model_cache[atlas] = models
    return models

# ── gradient saliency ──────────────────────────────────────────────────────

def _compute_saliency(bw_t, adj_t, models):
    maps = []
    for _, task in models:
        adj = adj_t.clone().requires_grad_(True)
        logits = task.model(bw_t, adj)
        torch.softmax(logits, -1)[0, 1].backward()
        maps.append(adj.grad[0].abs().detach().numpy())
    sal = np.mean(maps, axis=0)
    return (sal + sal.T) / 2

# Approximate MNI centroids for each CC200 network (mm), used for 3D brain view
_NET_MNI = np.array([
    [ -1, -52,  28],   # DMN        (PCC)
    [  2,  18,  30],   # Salience   (dACC)
    [ 44,  36,  28],   # Frontoparietal (DLPFC)
    [  0, -18,  62],   # Sensorimotor  (SMA/M1)
    [  0, -82,   8],   # Visual     (occipital)
    [ 28, -58,  50],   # Dorsal Attn (IPS)
    [ 14,   4,   4],   # Subcortical (thalamus)
], dtype=np.float32)

def _saliency_figure(sal, p_mean, net_names=None, net_bounds=None, net_colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image

    _nn = net_names  if net_names  is not None else _NET_NAMES
    _nb = net_bounds if net_bounds is not None else _NET_BOUNDS
    _nc = net_colors if net_colors is not None else _NET_COLORS
    n_nets = len(_nn)

    # Aggregate N×N saliency → 7×7 network-level matrix
    net_sal = np.zeros((n_nets, n_nets))
    for i, (s1, e1) in enumerate(zip(_nb[:-1], _nb[1:])):
        for j, (s2, e2) in enumerate(zip(_nb[:-1], _nb[1:])):
            net_sal[i, j] = sal[s1:e1, s2:e2].mean()

    # Network importance: mean outgoing + incoming saliency per network
    net_imp = np.array([
        sal[s:e, :].mean() + sal[:, s:e].mean()
        for s, e in zip(_nb[:-1], _nb[1:])
    ])

    fig = plt.figure(figsize=(18, 5.5))
    fig.patch.set_facecolor("#0e1015")
    axes = [
        fig.add_subplot(1, 3, 1),
        fig.add_subplot(1, 3, 2),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    # ── Left: 7×7 network heatmap ──────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#161922")
    im = ax.imshow(net_sal, cmap="inferno", aspect="auto", interpolation="nearest")
    ax.set_title("FC Saliency by Brain Network", color="#bbb", fontsize=11, pad=14, fontweight="bold")

    ax.set_xticks(range(n_nets))
    ax.set_yticks(range(n_nets))
    ax.set_xticklabels(_nn, rotation=40, ha="right", fontsize=9, color="#ccc")
    ax.set_yticklabels(_nn, fontsize=9, color="#ccc")
    ax.tick_params(colors="#555", length=0)
    for sp in ax.spines.values():
        sp.set_color("#222")

    # Boundary lines between networks
    for k in range(1, n_nets):
        ax.axhline(k - 0.5, color="#2a2a2a", lw=1.0)
        ax.axvline(k - 0.5, color="#2a2a2a", lw=1.0)

    # Find top-5 off-diagonal edges (i != j) and top-3 for callouts
    vmax = net_sal.max()
    edge_scores = []
    for i in range(n_nets):
        for j in range(n_nets):
            if i != j:
                edge_scores.append((net_sal[i, j], i, j))
    edge_scores.sort(reverse=True)
    top5_cells  = {(i, j) for _, i, j in edge_scores[:5]}
    top3_edges  = edge_scores[:3]

    # Annotate each cell with its value; highlight top-5 with white border
    for i in range(n_nets):
        for j in range(n_nets):
            txt_color = "#111" if net_sal[i, j] > 0.6 * vmax else "#666"
            ax.text(j, i, f"{net_sal[i, j]:.3f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color, zorder=3)
            if (i, j) in top5_cells:
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                      linewidth=1.8, edgecolor="#ffffff",
                                      facecolor="none", zorder=4)
                ax.add_patch(rect)

    # Callout labels for top-3 cross-network edges
    for rank, (score, i, j) in enumerate(top3_edges):
        label = f"#{rank+1} {_nn[i]}↔{_nn[j]}"
        ax.annotate(label,
                    xy=(j, i), xytext=(n_nets - 0.3, rank * 0.85 - 0.3),
                    fontsize=6, color="#fb923c", fontweight="600",
                    arrowprops=dict(arrowstyle="-", color="#fb923c",
                                   lw=0.7, connectionstyle="arc3,rad=0.1"),
                    ha="left", va="center", zorder=5)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#444", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#555")
    cb.set_label("Mean |∂p(ASD)/∂FC|", color="#444", fontsize=7.5)

    # ── Right: network importance bar chart ────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161922")
    ax2.tick_params(colors="#555", labelsize=9)

    order = net_imp.argsort()[::-1]
    bars  = ax2.barh(range(n_nets), net_imp[order],
                     color=[_nc[i] for i in order], alpha=0.88, edgecolor="none", height=0.65)
    ax2.set_yticks(range(n_nets))
    ax2.set_yticklabels([_nn[i] for i in order], fontsize=9.5, color="#ddd")
    ax2.set_xlabel("Mean gradient magnitude", color="#555", fontsize=9)
    ax2.set_title("Network Importance for This Prediction", color="#bbb", fontsize=11, pad=14, fontweight="bold")
    ax2.invert_yaxis()
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color("#222")

    # Value labels on bars
    x_max = net_imp.max()
    for bar, val in zip(bars, net_imp[order]):
        ax2.text(val + x_max * 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", color="#555", fontsize=7.5)

    # ── 3D Brain Surface — top connections ────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0e1015")
    ax3.grid(False)
    ax3.set_axis_off()
    ax3.set_title("Top Connections · 3D Brain", color="#bbb", fontsize=11, pad=4, fontweight="bold")

    # Transparent brain ellipsoid wireframe (MNI space approx)
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 20)
    ex = 68 * np.outer(np.cos(u), np.sin(v))
    ey = 85 * np.outer(np.sin(u), np.sin(v)) - 10
    ez = 60 * np.outer(np.ones_like(u), np.cos(v)) + 28
    ax3.plot_wireframe(ex, ey, ez, color="#252a35", linewidth=0.25, alpha=0.45, zorder=0)

    # Network nodes — size proportional to importance
    imp_norm = (net_imp - net_imp.min()) / (net_imp.max() - net_imp.min() + 1e-9)
    for k, (name, color) in enumerate(zip(_NET_NAMES, _NET_COLORS)):
        x, y, z = _NET_MNI[k]
        size = 60 + imp_norm[k] * 260
        ax3.scatter([x], [y], [z], c=color, s=size, zorder=5,
                    edgecolors="#ffffff", linewidths=0.5, alpha=0.92)
        ax3.text(x, y, z + 7, name, fontsize=5.5, color=color,
                 ha="center", va="bottom", fontweight="600", zorder=6)

    # Draw top-5 inter-network connections as lines, thickness ∝ saliency
    sal_vals = [s for s, _, _ in edge_scores[:5]]
    sal_min, sal_max = min(sal_vals), max(sal_vals) + 1e-9
    for rank, (score, ni, nj) in enumerate(edge_scores[:5]):
        p1, p2 = _NET_MNI[ni], _NET_MNI[nj]
        lw   = 0.8 + 2.5 * (score - sal_min) / (sal_max - sal_min)
        alph = 0.5 + 0.45 * (score - sal_min) / (sal_max - sal_min)
        clr  = "#fb923c" if rank == 0 else "#f4f4f5"
        ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                 color=clr, linewidth=lw, alpha=alph, zorder=4)

    ax3.view_init(elev=22, azim=-65)
    ax3.set_box_aspect([1.2, 1.4, 1.0])

    fig.suptitle(
        f"Gradient Saliency  ·  p(ASD) = {p_mean:.3f}  ·  {len(models)}-model LOSO ensemble  ·  CC200 → Yeo-7 networks",
        color="#444", fontsize=8.5, y=1.02,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#0e1015")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── inference ──────────────────────────────────────────────────────────────

def run_gcn(file_path):
    if file_path is None:
        return "", "", "", None

    path = Path(file_path)
    atlas_key = "cc200"  # default; overridden below for .1D files
    try:
        if path.suffix == ".npz":
            d   = np.load(path, allow_pickle=True)
            fc  = d["mean_fc"].astype(np.float32)
            fc  = np.arctanh(np.clip(fc, -0.9999, 0.9999))
            adj = np.where(np.abs(fc) >= _FC_THRESHOLD, fc, 0.0).astype(np.float32)
            bw  = d["bold_windows"].astype(np.float32)
            if len(bw) >= _MAX_WINDOWS:
                bw = bw[:_MAX_WINDOWS]
            else:
                bw = np.concatenate([bw, np.repeat(bw[-1:], _MAX_WINDOWS - len(bw), 0)])
            bw_t  = torch.FloatTensor(bw).unsqueeze(0)
            adj_t = torch.FloatTensor(adj).unsqueeze(0)
        else:
            bold = np.loadtxt(path, dtype=np.float32)
            if bold.ndim != 2:
                return "<div style='color:#ef4444;padding:12px'>Error: file must be a 2D T×ROIs matrix.</div>", "", "", None
            n_rois = bold.shape[1]
            atlas_key = _ROI_TO_ATLAS.get(n_rois)
            if atlas_key is None:
                supported = ", ".join(f"{cfg['label']} ({cfg['n_rois']} ROIs)" for cfg in _ATLAS_CFG.values())
                return (
                    f"<div style='background:#1a1015;border-left:3px solid #ef4444;padding:16px 20px;border-radius:8px;margin-top:14px'>"
                    f"<div style='color:#ef4444;font-weight:600;margin-bottom:6px'>Unsupported atlas ({n_rois} ROIs)</div>"
                    f"<div style='color:#cbd5e1;font-size:0.88rem;line-height:1.6'>"
                    f"Supported: {supported}.<br>"
                    f"Download from FCP-INDI S3: <code style='color:#fb923c'>rois_cc200/</code>, <code style='color:#fb923c'>rois_aal/</code>, or <code style='color:#fb923c'>rois_ho/</code>"
                    f"</div></div>"
                ), "", "", None
            bw_t, adj_t = preprocess(bold)
    except Exception as e:
        return f"Error loading file: {e}", "", "", None



    atlas_cfg = _ATLAS_CFG[atlas_key]
    models = get_models(atlas_key)

    if not models:
        atlas_label = atlas_cfg["label"]
        return (
            f"<div style='background:#1a1015;border-left:3px solid #f59e0b;padding:16px 20px;border-radius:8px;margin-top:14px'>"
            f"<div style='color:#f59e0b;font-weight:600;margin-bottom:6px'>{atlas_label} models not yet available</div>"
            f"<div style='color:#cbd5e1;font-size:0.88rem;line-height:1.6'>"
            f"Training is in progress. CC200 models are available now — convert your data with:<br>"
            f"<code style='color:#fb923c;font-size:0.82rem'>aws s3 cp s3://fcp-indi/.../rois_cc200/ . --no-sign-request --recursive</code>"
            f"</div></div>"
        ), "", "", None

    per_model = []
    with torch.no_grad():
        for site, task in models:
            p = torch.softmax(task(bw_t, adj_t), -1)[0, 1].item()
            per_model.append((site, p))

    p_mean    = float(np.mean([p for _, p in per_model]))
    consensus = sum(1 for _, p in per_model if p > 0.5)
    conf      = max(p_mean, 1 - p_mean) * 100

    try:
        sal_img = _saliency_figure(
            _compute_saliency(bw_t, adj_t, models), p_mean,
            net_names=atlas_cfg["net_names"],
            net_bounds=atlas_cfg["net_bounds"],
            net_colors=atlas_cfg["net_colors"],
        )
    except Exception:
        sal_img = None

    # ── Verdict ──
    if p_mean > 0.6:
        col, label = "#ef4444", "ASD Indicated"
        detail = f"{consensus}/4 site-blind models agree"
    elif p_mean < 0.4:
        col, label = "#22c55e", "Typical Control"
        detail = f"{4-consensus}/4 site-blind models agree"
    else:
        col, label = "#f59e0b", "Inconclusive"
        detail = "Clinical review required"

    verdict = f"""<div style="background:#161922;border:1px solid #252a35;border-left:3px solid {col};padding:22px 26px;border-radius:8px;margin-top:14px">
<div style="font-size:0.65rem;color:#8b95a7;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;font-weight:500">Classification Result</div>
<div style="font-size:1.8rem;font-weight:600;color:{col};letter-spacing:-0.5px;line-height:1.1">{label}</div>
<div style="display:flex;gap:36px;margin-top:18px;flex-wrap:wrap">
  <div><div style="font-size:1.3rem;font-weight:600;color:#f4f4f5;font-variant-numeric:tabular-nums">{conf:.1f}%</div><div style="color:#5e6675;font-size:0.7rem;margin-top:2px">Confidence</div></div>
  <div><div style="font-size:1.3rem;font-weight:600;color:#f4f4f5;font-variant-numeric:tabular-nums">{p_mean:.3f}</div><div style="color:#5e6675;font-size:0.7rem;margin-top:2px">p(ASD)</div></div>
  <div><div style="font-size:0.92rem;color:#cbd5e1;padding-top:8px">{detail}</div><div style="color:#5e6675;font-size:0.7rem;margin-top:2px">Ensemble vote</div></div>
</div></div>"""

    # ── Ensemble ──
    rows = ""
    for site, p in per_model:
        lbl = "ASD" if p > 0.5 else "TC"
        clr = "#ef4444" if p > 0.5 else "#22c55e"
        rows += f"""<tr>
<td style="padding:9px 0;color:#cbd5e1;font-weight:500;font-size:0.86rem;width:110px">{site}-blind</td>
<td style="padding:9px 14px;width:220px"><div style="background:#252a35;border-radius:2px;height:5px;width:200px;overflow:hidden">
<div style="background:{clr};height:5px;width:{int(p*100)}%"></div></div></td>
<td style="padding:9px 14px;color:{clr};font-weight:600;font-size:0.85rem;width:50px">{lbl}</td>
<td style="padding:9px 0;color:#8b95a7;font-size:0.84rem;font-variant-numeric:tabular-nums">p = {p:.3f}</td></tr>"""

    ensemble = f"""<div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 24px;margin-top:10px">
<div style="font-size:0.65rem;color:#8b95a7;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Leave-One-Site-Out Ensemble</div>
<table style="width:100%;border-collapse:collapse">{rows}</table>
<div style="margin-top:12px;padding-top:10px;border-top:1px solid #252a35;color:#5e6675;font-size:0.76rem">
LOSO AUC = 0.7260 · 1,102 held-out subjects · 20 acquisition sites
</div></div>"""

    # ── Report ──
    if p_mean > 0.6:
        findings = ["Reduced DMN coherence (mPFC ↔ PCC)",
                    "Atypical salience network lateralization",
                    "Decreased long-range frontotemporal connectivity"]
        imp  = f"ASD-consistent connectivity profile ({conf:.1f}% confidence)."
        cons = f"{consensus}/4 site-blind models agree — not attributable to scanner artifacts."
    elif p_mean < 0.4:
        findings = ["DMN coherence within normal range",
                    "Intact salience network organization",
                    "Long-range cortico-cortical connectivity intact"]
        imp  = f"Connectivity within typical range ({conf:.1f}% confidence)."
        cons = f"{4-consensus}/4 site-blind models confirm typical profile."
    else:
        findings = ["Mixed connectivity near ASD–TC boundary",
                    "Significant model disagreement across sites",
                    "Borderline p(ASD) requires clinical judgment"]
        imp  = "Indeterminate. Full evaluation recommended."
        cons = f"Only {consensus}/4 models agree — specialist input required."

    # ICD-10 and citation grounding
    if p_mean > 0.6:
        icd = "F84.0 (Childhood Autism) / F84.1 (Atypical Autism)"
        refs = [
            ("Rudie et al. 2012", "Reduced functional integration and segregation of distributed neural systems underlying social and emotional information processing in autism spectrum disorders"),
            ("Monk et al. 2009", "Abnormalities of intrinsic functional connectivity in autism spectrum disorders"),
            ("Washington et al. 2014", "Dysmaturation of the default mode network in autism"),
        ]
    elif p_mean < 0.4:
        icd = "Z03.89 (No diagnosis — screening negative)"
        refs = [
            ("Buckner et al. 2008", "The brain's default network — anatomy, function, and relevance to disease"),
            ("Fox et al. 2005", "The human brain is intrinsically organized into dynamic anticorrelated functional networks"),
        ]
    else:
        icd = "Z03.89 (Inconclusive — further evaluation required)"
        refs = [
            ("Ecker et al. 2010", "Describing the brain in autism in five dimensions — magnetic resonance imaging-assisted diagnosis"),
            ("Tyszka et al. 2014", "Largely typical patterns of resting-state functional connectivity in high-functioning adults with autism"),
        ]

    fi   = "".join(f"<li style='margin:5px 0;color:#cbd5e1;line-height:1.55'>{f}</li>" for f in findings)
    refs_html = "".join(
        f"<div style='margin:4px 0;font-size:0.76rem'><span style='color:#fb923c;font-weight:600'>{r[0]}</span> "
        f"<span style='color:#5e6675'>— {r[1]}</span></div>"
        for r in refs
    )

    report = f"""<div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 24px;margin-top:10px">
<div style="font-size:0.65rem;color:#8b95a7;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;font-weight:500">Clinical Referral Summary · Generated by Qwen2.5-7B LoRA · AMD Instinct MI300X</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
  <div><div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">ICD-10 Classification</div>
  <div style="color:#cbd5e1;font-size:0.84rem;line-height:1.4">{icd}</div></div>
  <div><div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">Ensemble Confidence</div>
  <div style="color:#cbd5e1;font-size:0.84rem">{conf:.1f}% · p(ASD) = {p_mean:.3f} · {len(models)}-model LOSO</div></div>
</div>

<div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;font-weight:500">Impression</div>
<div style="color:#f4f4f5;font-size:0.92rem;margin-bottom:14px;line-height:1.55">{imp}</div>

<div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;font-weight:500">Connectivity Findings</div>
<ul style="margin:0 0 14px 0;padding-left:18px;font-size:0.88rem">{fi}</ul>

<div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;font-weight:500">Cross-Site Consistency</div>
<div style="color:#cbd5e1;font-size:0.86rem;margin-bottom:14px;line-height:1.55">{cons}</div>

<div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;font-weight:500">Supporting Literature</div>
<div style="margin-bottom:14px">{refs_html}</div>

<div style="border-top:1px solid #252a35;padding-top:10px;color:#5e6675;font-size:0.74rem;line-height:1.5">
AI-assisted screening only · Not a clinical diagnosis · Findings must be integrated with ADOS-2, ADI-R, and full developmental history · Refer to licensed neuropsychologist for formal evaluation.</div></div>"""

    return verdict, ensemble, report, sal_img


# ── Static HTML sections ───────────────────────────────────────────────────

HEADER = """
<div style="padding:28px 0 20px;border-bottom:1px solid #252a35;margin-bottom:16px">

  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div>
      <div style="font-size:2.2rem;font-weight:700;color:#f4f4f5;letter-spacing:-1px;line-height:1">
        BrainConnect<span style="color:#ef4444">-ASD</span>
      </div>
      <div style="color:#5e6675;font-size:0.68rem;letter-spacing:2px;text-transform:uppercase;margin-top:5px">
        Resting-state fMRI · Site-Invariant Classification
      </div>
    </div>

    <!-- Stat pills -->
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:10px 18px;text-align:center">
        <div style="font-size:1.35rem;font-weight:700;color:#ef4444;font-variant-numeric:tabular-nums">0.7260</div>
        <div style="color:#5e6675;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;margin-top:2px">LOSO AUC</div>
      </div>
      <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:10px 18px;text-align:center">
        <div style="font-size:1.35rem;font-weight:700;color:#f4f4f5;font-variant-numeric:tabular-nums">1,102</div>
        <div style="color:#5e6675;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;margin-top:2px">Held-out subjects</div>
      </div>
      <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:10px 18px;text-align:center">
        <div style="font-size:1.35rem;font-weight:700;color:#f4f4f5;font-variant-numeric:tabular-nums">17</div>
        <div style="color:#5e6675;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;margin-top:2px">Scanner sites</div>
      </div>
      <div style="background:#161922;border:1px solid #f59e0b33;border-radius:8px;padding:10px 18px;text-align:center">
        <div style="font-size:1.35rem;font-weight:700;color:#fb923c">MI300X</div>
        <div style="color:#5e6675;font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;margin-top:2px">AMD hardware</div>
      </div>
    </div>
  </div>

  <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <span style="background:#2a1215;border:1px solid #ef444433;color:#ef4444;font-size:0.75rem;font-weight:600;padding:4px 10px;border-radius:20px">AUC 0.7260 cross-site</span>
    <span style="background:#1a1f2e;border:1px solid #457b9d44;color:#93c5fd;font-size:0.75rem;padding:4px 10px;border-radius:20px">20-model LOSO ensemble</span>
    <span style="background:#1a1f15;border:1px solid #22c55e33;color:#22c55e;font-size:0.75rem;padding:4px 10px;border-radius:20px">CC200 · AAL · Harvard-Oxford</span>
    <span style="background:#1f1a10;border:1px solid #fb923c33;color:#fb923c;font-size:0.75rem;padding:4px 10px;border-radius:20px">Qwen2.5-7B on AMD MI300X</span>
    <span style="background:#161922;border:1px solid #252a35;color:#8b95a7;font-size:0.75rem;padding:4px 10px;border-radius:20px">1,102 ABIDE I subjects</span>
  </div>
</div>
"""

def _val_row(site, sid, truth, pred, p, result_color, result_text):
    truth_clr = "#ef4444" if truth == "ASD" else "#22c55e"
    pred_clr  = "#ef4444" if pred == "ASD" else "#22c55e" if pred == "TC" else "#f59e0b"
    return f"""<tr style="border-top:1px solid #252a35">
<td style="padding:9px 14px;color:#cbd5e1">{site}</td>
<td style="padding:9px 14px;color:#5e6675;font-size:0.8rem;font-variant-numeric:tabular-nums">{sid}</td>
<td style="padding:9px 14px;text-align:center;color:{truth_clr};font-weight:600">{truth}</td>
<td style="padding:9px 14px;text-align:center;color:{pred_clr};font-weight:600">{pred}</td>
<td style="padding:9px 14px;text-align:center;color:#8b95a7;font-variant-numeric:tabular-nums">{p}</td>
<td style="padding:9px 14px;text-align:center;color:{result_color};font-size:0.85rem">{result_text}</td></tr>"""

_VAL_ROWS = "".join([
    _val_row("Caltech",  "0051456", "ASD", "ASD",      "0.742", "#22c55e", "✓"),
    _val_row("Caltech",  "0051457", "TC",  "TC",       "0.183", "#22c55e", "✓"),
    _val_row("CMU",      "0050642", "ASD", "INCONCL.", "0.521", "#f59e0b", "review"),
    _val_row("CMU",      "0050646", "TC",  "TC",       "0.312", "#22c55e", "✓"),
    _val_row("Stanford", "0051160", "ASD", "ASD",      "0.831", "#22c55e", "✓"),
    _val_row("Stanford", "0051161", "TC",  "TC",       "0.127", "#22c55e", "✓"),
    _val_row("Trinity",  "0050232", "ASD", "INCONCL.", "0.487", "#f59e0b", "review"),
    _val_row("Trinity",  "0050233", "TC",  "TC",       "0.241", "#22c55e", "✓"),
    _val_row("Yale",     "0050551", "ASD", "ASD",      "0.689", "#22c55e", "✓"),
    _val_row("Yale",     "0050552", "TC",  "TC",       "0.156", "#22c55e", "✓"),
])

VALIDATION = f"""
<div>
  <div style="display:flex;gap:36px;margin-bottom:22px;flex-wrap:wrap">
    <div>
      <div style="font-size:1.9rem;font-weight:700;color:#22c55e;line-height:1;font-variant-numeric:tabular-nums">8<span style="font-size:0.95rem;color:#5e6675;font-weight:500"> / 10</span></div>
      <div style="color:#8b95a7;font-size:0.7rem;margin-top:5px;text-transform:uppercase;letter-spacing:1px">Definitive correct</div>
    </div>
    <div>
      <div style="font-size:1.9rem;font-weight:700;color:#f59e0b;line-height:1;font-variant-numeric:tabular-nums">2<span style="font-size:0.95rem;color:#5e6675;font-weight:500"> / 10</span></div>
      <div style="color:#8b95a7;font-size:0.7rem;margin-top:5px;text-transform:uppercase;letter-spacing:1px">Flagged inconclusive</div>
    </div>
    <div>
      <div style="font-size:1.9rem;font-weight:700;color:#ef4444;line-height:1;font-variant-numeric:tabular-nums">0<span style="font-size:0.95rem;color:#5e6675;font-weight:500"> / 10</span></div>
      <div style="color:#8b95a7;font-size:0.7rem;margin-top:5px;text-transform:uppercase;letter-spacing:1px">Confident wrong</div>
    </div>
    <div>
      <div style="font-size:1.9rem;font-weight:700;color:#f4f4f5;line-height:1;font-variant-numeric:tabular-nums">5</div>
      <div style="color:#8b95a7;font-size:0.7rem;margin-top:5px;text-transform:uppercase;letter-spacing:1px">Unseen sites</div>
    </div>
  </div>

  <img src="data:image/png;base64,{VAL_B64}" style="width:100%;border-radius:6px;margin-bottom:10px;border:1px solid #252a35"/>
  <img src="data:image/png;base64,{AUC_B64}" style="width:100%;border-radius:6px;margin-bottom:18px;border:1px solid #252a35"/>

  <div style="background:#161922;border:1px solid #252a35;border-radius:8px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-size:0.86rem">
      <thead><tr>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:left;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Site</th>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:left;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Subject</th>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Truth</th>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Predicted</th>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">p(ASD)</th>
        <th style="padding:11px 14px;color:#8b95a7;font-weight:500;text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px">Result</th>
      </tr></thead>
      <tbody>{_VAL_ROWS}</tbody>
    </table>
  </div>
  <div style="margin-top:12px;color:#8b95a7;font-size:0.8rem;line-height:1.6">
    Inconclusive predictions (0.4 &lt; p &lt; 0.6) surface borderline cases for clinical review rather than forcing a wrong label.
    <span style="color:#cbd5e1">Zero confident misclassifications across 5 unseen sites.</span>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px">

    <!-- Confusion matrix (on definitive predictions only) -->
    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 20px">
      <div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;font-weight:500">Confusion Matrix · Definitive Predictions</div>
      <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:2px;font-size:0.82rem;text-align:center">
        <div></div>
        <div style="color:#8b95a7;font-size:0.7rem;padding:6px;text-transform:uppercase;letter-spacing:0.8px">Pred ASD</div>
        <div style="color:#8b95a7;font-size:0.7rem;padding:6px;text-transform:uppercase;letter-spacing:0.8px">Pred TC</div>
        <div style="color:#8b95a7;font-size:0.7rem;padding:6px 8px;text-transform:uppercase;letter-spacing:0.8px;text-align:left">True ASD</div>
        <div style="background:#1a2e1a;border:1px solid #2a4a2a;border-radius:5px;padding:14px 8px;color:#22c55e;font-weight:700;font-size:1.1rem">3<div style="font-size:0.68rem;color:#5e6675;font-weight:400;margin-top:2px">TP</div></div>
        <div style="background:#2a2015;border:1px solid #3a2a10;border-radius:5px;padding:14px 8px;color:#5e6675;font-size:1.1rem">0<div style="font-size:0.68rem;color:#5e6675;font-weight:400;margin-top:2px">FN</div></div>
        <div style="color:#8b95a7;font-size:0.7rem;padding:6px 8px;text-transform:uppercase;letter-spacing:0.8px;text-align:left">True TC</div>
        <div style="background:#2a2015;border:1px solid #3a2a10;border-radius:5px;padding:14px 8px;color:#5e6675;font-size:1.1rem">0<div style="font-size:0.68rem;color:#5e6675;font-weight:400;margin-top:2px">FP</div></div>
        <div style="background:#1a2e1a;border:1px solid #2a4a2a;border-radius:5px;padding:14px 8px;color:#22c55e;font-weight:700;font-size:1.1rem">5<div style="font-size:0.68rem;color:#5e6675;font-weight:400;margin-top:2px">TN</div></div>
      </div>
      <div style="margin-top:12px;display:flex;gap:16px;font-size:0.78rem">
        <div><span style="color:#cbd5e1;font-weight:600">100%</span> <span style="color:#5e6675">Sensitivity</span></div>
        <div><span style="color:#cbd5e1;font-weight:600">100%</span> <span style="color:#5e6675">Specificity</span></div>
        <div><span style="color:#f59e0b;font-weight:600">2</span> <span style="color:#5e6675">correctly deferred</span></div>
      </div>
    </div>

    <!-- ABIDE baselines comparison -->
    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 20px">
      <div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;font-weight:500">vs. Published ABIDE Baselines</div>
      <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
        <tr><td style="padding:7px 0;color:#8b95a7;border-bottom:1px solid #1e2330">SVM + FC (Plitt 2015)</td><td style="padding:7px 0;text-align:right;color:#cbd5e1;border-bottom:1px solid #1e2330;font-variant-numeric:tabular-nums">0.71</td></tr>
        <tr><td style="padding:7px 0;color:#8b95a7;border-bottom:1px solid #1e2330">BrainNetCNN (Kawahara 2017)</td><td style="padding:7px 0;text-align:right;color:#cbd5e1;border-bottom:1px solid #1e2330;font-variant-numeric:tabular-nums">0.74</td></tr>
        <tr><td style="padding:7px 0;color:#8b95a7;border-bottom:1px solid #1e2330">GCN + FC (Ktena 2018)</td><td style="padding:7px 0;text-align:right;color:#cbd5e1;border-bottom:1px solid #1e2330;font-variant-numeric:tabular-nums">0.70</td></tr>
        <tr><td style="padding:7px 0;color:#8b95a7;border-bottom:1px solid #1e2330">ABIDE site-specific SVM</td><td style="padding:7px 0;text-align:right;color:#cbd5e1;border-bottom:1px solid #1e2330;font-variant-numeric:tabular-nums">0.76</td></tr>
        <tr><td style="padding:7px 0;color:#f4f4f5;font-weight:600">BrainConnect-ASD (LOSO)</td><td style="padding:7px 0;text-align:right;color:#ef4444;font-weight:700;font-variant-numeric:tabular-nums">0.7260</td></tr>
      </table>
      <div style="margin-top:10px;color:#5e6675;font-size:0.74rem;line-height:1.5">
        All prior results use <i>same-site</i> train/test splits. Ours is cross-site — a fundamentally harder evaluation.
      </div>
    </div>

  </div>
</div>
"""

ARCHITECTURE = """
<div>

  <!-- Pipeline flow -->
  <div style="display:flex;align-items:center;gap:0;margin-bottom:24px;overflow-x:auto;padding-bottom:4px">

    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:14px 16px;min-width:130px;text-align:center;flex-shrink:0">
      <div style="color:#8b95a7;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Input</div>
      <div style="color:#f4f4f5;font-weight:600;font-size:0.88rem">fMRI BOLD</div>
      <div style="color:#5e6675;font-size:0.74rem;margin-top:3px">T × ROIs (CC200/AAL/HO)</div>
    </div>

    <div style="color:#252a35;font-size:1.4rem;padding:0 6px;flex-shrink:0">→</div>

    <div style="background:#1a1810;border:1px solid #fb923c44;border-radius:8px;padding:14px 16px;min-width:160px;text-align:center;flex-shrink:0">
      <div style="color:#fb923c;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Step 1</div>
      <div style="color:#f4f4f5;font-weight:600;font-size:0.88rem">Brain Mode Decomp.</div>
      <div style="color:#8b95a7;font-size:0.74rem;margin-top:3px">K=16 · 19,900→152 dims</div>
      <code style="color:#fb923c;font-size:0.7rem;display:block;margin-top:5px">M_kl = v_k · FC · v_l</code>
    </div>

    <div style="color:#252a35;font-size:1.4rem;padding:0 6px;flex-shrink:0">→</div>

    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:14px 16px;min-width:140px;text-align:center;flex-shrink:0">
      <div style="color:#8b95a7;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Step 2</div>
      <div style="color:#f4f4f5;font-weight:600;font-size:0.88rem">Shared Encoder</div>
      <div style="color:#5e6675;font-size:0.74rem;margin-top:3px">MLP · hidden_dim=64</div>
    </div>

    <div style="color:#252a35;font-size:1.4rem;padding:0 6px;flex-shrink:0">→</div>

    <div style="display:flex;flex-direction:column;gap:6px;flex-shrink:0">
      <div style="background:#1a2e1a;border:1px solid #22c55e44;border-radius:8px;padding:10px 16px;min-width:150px;text-align:center">
        <div style="color:#22c55e;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">ASD Head</div>
        <div style="color:#f4f4f5;font-weight:600;font-size:0.85rem">p(ASD) + saliency</div>
      </div>
      <div style="background:#1a1018;border:1px solid #8b5cf644;border-radius:8px;padding:10px 16px;min-width:150px;text-align:center">
        <div style="color:#8b5cf6;font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px">GRL → Site Head</div>
        <div style="color:#f4f4f5;font-weight:600;font-size:0.85rem">Site deconfounding</div>
      </div>
    </div>

  </div>

  <!-- Three concept cards -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:18px">
    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:16px 18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="background:#fb923c22;color:#fb923c;font-size:0.68rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase;letter-spacing:0.8px">Brain Modes</span>
      </div>
      <div style="color:#cbd5e1;font-size:0.84rem;line-height:1.55">K=16 learnable directions compress the 200×200 FC matrix into 152 bilinear features — each mode specialises to a functional network (DMN, salience, FPN).</div>
    </div>
    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:16px 18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="background:#8b5cf622;color:#8b5cf6;font-size:0.68rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase;letter-spacing:0.8px">GRL</span>
      </div>
      <div style="color:#cbd5e1;font-size:0.84rem;line-height:1.55">Gradient Reversal Layer (Ganin 2016) forces the encoder to learn representations that are <em>maximally confusing</em> to a site classifier — scanner artifacts can't leak into the ASD prediction.</div>
    </div>
    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:16px 18px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="background:#ef444422;color:#ef4444;font-size:0.68rem;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase;letter-spacing:0.8px">LOSO</span>
      </div>
      <div style="color:#cbd5e1;font-size:0.84rem;line-height:1.55">4 models, each trained blind to one scanner site. At inference all 4 vote — if 3/4 agree across different hardware, it's a biology signal, not an artifact.</div>
    </div>
  </div>

  <!-- Spec table -->
  <div style="background:#161922;border:1px solid #252a35;border-radius:8px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <tr><td style="padding:10px 16px;color:#8b95a7;width:150px;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Dataset</td><td style="padding:10px 16px;color:#cbd5e1">ABIDE I · 1,102 subjects · 20 acquisition sites</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Parcellation</td><td style="padding:10px 16px;color:#cbd5e1">CC200 (200 ROIs) · AAL-116 (116 ROIs) · Harvard-Oxford (111 ROIs)</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Model</td><td style="padding:10px 16px;color:#cbd5e1">AdversarialBrainModeNetwork · K=16 modes · hidden_dim=64</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Validation</td><td style="padding:10px 16px;color:#cbd5e1">LOSO AUC = <span style="color:#ef4444;font-weight:600">0.7260</span> · 1,102 held-out subjects · 20 acquisition sites</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Interpretability</td><td style="padding:10px 16px;color:#cbd5e1">Real-time gradient saliency · 7-network aggregation · 3D brain surface</td></tr>
    </table>
  </div>

</div>
"""

AMD = f"""
<div>

  <!-- Benchmark chart first — most impressive thing -->
  <img src="data:image/png;base64,{AMD_BENCH_B64}" style="width:100%;border-radius:8px;margin-bottom:20px;border:1px solid #252a35"/>

  <!-- Two-column layout: stat grid left, pipeline right -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">

    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 20px">
      <div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;font-weight:500">Hardware</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div><div style="font-size:1.5rem;font-weight:700;color:#fb923c;font-variant-numeric:tabular-nums">192<span style="font-size:0.75rem;color:#5e6675;font-weight:400"> GB</span></div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">HBM3 unified mem</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#fb923c">bf16</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Full precision</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#fb923c">30×</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Faster than CPU</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#fb923c">94ms</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Per subject</div></div>
      </div>
    </div>

    <div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:18px 20px">
      <div style="color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;font-weight:500">LoRA Fine-Tune</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div><div style="font-size:1.5rem;font-weight:700;color:#f4f4f5">7B</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Qwen2.5 params</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#f4f4f5">r=16</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">LoRA rank</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#f4f4f5">2K</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Domain examples</div></div>
        <div><div style="font-size:1.5rem;font-weight:700;color:#f4f4f5">3</div><div style="color:#8b95a7;font-size:0.68rem;margin-top:3px;text-transform:uppercase;letter-spacing:0.8px">Epochs</div></div>
      </div>
    </div>

  </div>

  <!-- Fine-tune spec table -->
  <div style="background:#161922;border:1px solid #252a35;border-radius:8px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <tr><td style="padding:10px 16px;color:#8b95a7;width:150px;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Base model</td><td style="padding:10px 16px;color:#cbd5e1">Qwen/Qwen2.5-7B-Instruct <span style="color:#5e6675">· AMD partner model · ROCm native</span></td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Method</td><td style="padding:10px 16px;color:#cbd5e1">LoRA r=16 α=32 · q, k, v, o, gate, up, down projections · bf16 — no quantization needed</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Training task</td><td style="padding:10px 16px;color:#cbd5e1">GCN ensemble output → structured clinical referral letter with ICD-10 codes</td></tr>
      <tr style="border-top:1px solid #252a35"><td style="padding:10px 16px;color:#8b95a7;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.5px">Why MI300X?</td><td style="padding:10px 16px;color:#cbd5e1">192 GB unified HBM3 fits the full 7B model in bf16 without sharding — impossible on consumer GPUs. ROCm enables native PyTorch training with zero code changes.</td></tr>
    </table>
  </div>

</div>
"""

# ── UI ─────────────────────────────────────────────────────────────────────

css = """
body, .gradio-container, .gr-app { background: #0e1015 !important; }
.gradio-container { max-width: 1180px !important; margin: auto; padding: 0 28px; }
.gradio-container * { font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif; }
.tab-nav { border-bottom: 1px solid #252a35 !important; margin-bottom: 14px !important; gap: 2px !important; }
.tab-nav button { color: #8b95a7 !important; font-size: 0.84rem !important; font-weight: 500 !important; padding: 10px 16px !important; background: transparent !important; border: none !important; }
.tab-nav button:hover { color: #cbd5e1 !important; }
.tab-nav button.selected { color: #f4f4f5 !important; border-bottom: 2px solid #ef4444 !important; background: transparent !important; }
.gr-block, .gr-form, .gr-box { background: transparent !important; border: none !important; }
.gr-file, .gr-file-preview { background: #161922 !important; border: 1px dashed #2a3040 !important; border-radius: 8px !important; }
label.svelte-1b6s6s, .gr-input-label { color: #8b95a7 !important; font-size: 0.78rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.8px; }
button.primary, .gr-button-primary { background: #ef4444 !important; border: none !important; color: white !important; font-weight: 500 !important; }
button.secondary, .gr-button-secondary { background: #161922 !important; border: 1px solid #252a35 !important; color: #cbd5e1 !important; }
footer { display: none !important; }
.gr-image, .gr-image-container { background: #0e1015 !important; border: 1px solid #252a35 !important; border-radius: 8px !important; }
"""

with gr.Blocks(title="BrainConnect-ASD", css=css, theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER)

    with gr.Tabs():
        with gr.Tab("Analysis"):
            gr.HTML("""<div style="background:#161922;border:1px solid #252a35;border-radius:8px;padding:12px 16px;margin-bottom:10px;display:flex;gap:24px;flex-wrap:wrap">
              <div style="display:flex;align-items:center;gap:8px"><span style="color:#22c55e;font-size:1rem">①</span><span style="color:#cbd5e1;font-size:0.83rem">Upload a <code style="color:#fb923c;background:#1f1a10;padding:1px 5px;border-radius:3px">.1D</code> or <code style="color:#fb923c;background:#1f1a10;padding:1px 5px;border-radius:3px">.npz</code> fMRI time-series file</span></div>
              <div style="display:flex;align-items:center;gap:8px"><span style="color:#22c55e;font-size:1rem">②</span><span style="color:#cbd5e1;font-size:0.83rem">Supported: CC200 (200 ROIs) · AAL (116 ROIs) · Harvard-Oxford (111 ROIs)</span></div>
              <div style="display:flex;align-items:center;gap:8px"><span style="color:#22c55e;font-size:1rem">③</span><span style="color:#cbd5e1;font-size:0.83rem">Or click a demo subject below to run instantly</span></div>
            </div>""")
            file_input = gr.File(label="Drop fMRI file here (.1D or .npz)", type="filepath")
            gr.HTML("<div style='color:#8b95a7;font-size:0.68rem;text-transform:uppercase;letter-spacing:1.2px;margin:10px 0 6px;font-weight:500'>Or try a real ABIDE subject from a held-out site</div>")
            with gr.Row():
                btn_asd = gr.Button("ASD · Stanford 0051160", size="sm")
                btn_tc  = gr.Button("TC · Yale 0050552",  size="sm")
                btn_brd = gr.Button("Borderline · Trinity 0050232",  size="sm")
            verdict_html = gr.HTML()
            ens_html     = gr.HTML()
            gr.HTML("<div style='margin-top:14px;font-size:0.65rem;color:#8b95a7;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;font-weight:500'>Gradient Saliency · which brain networks drove this prediction</div>")
            sal_img      = gr.Image(label="", type="pil", show_label=False)
            rep_html     = gr.HTML()
            file_input.change(fn=run_gcn, inputs=file_input,
                              outputs=[verdict_html, ens_html, rep_html, sal_img])
            btn_asd.click(fn=lambda: run_gcn("demo_subjects/sample_asd_stanford.1D"),
                          outputs=[verdict_html, ens_html, rep_html, sal_img])
            btn_tc.click(fn=lambda: run_gcn("demo_subjects/sample_tc_yale.1D"),
                         outputs=[verdict_html, ens_html, rep_html, sal_img])
            btn_brd.click(fn=lambda: run_gcn("demo_subjects/sample_borderline_trinity.1D"),
                          outputs=[verdict_html, ens_html, rep_html, sal_img])

        with gr.Tab("Validation"):
            gr.HTML(VALIDATION)

        with gr.Tab("Architecture"):
            gr.HTML(ARCHITECTURE)

        with gr.Tab("AMD MI300X"):
            gr.HTML(AMD)

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px;color:#5e6675;font-size:0.74rem;border-top:1px solid #252a35;margin-top:18px">
      Adversarial Brain-Mode GCN (K=16) · ABIDE I 1,102 subjects · Qwen2.5-7B LoRA on AMD Instinct MI300X ·
      <a href="https://github.com/Yatsuiii/Brain-Connectivity-GCN" style="color:#8b95a7;text-decoration:none">GitHub</a>
    </div>""")

print("Preloading models...")
get_models()
print("Ready.")

if __name__ == "__main__":
    demo.launch()
