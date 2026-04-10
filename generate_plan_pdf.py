"""Generate Brain-Connectivity-GCN project plan as a PDF."""
from fpdf import FPDF

FONT_DIR = (
    "/home/Yatsuiii/Brain-Connectivity-GCN/venv/lib/python3.14"
    "/site-packages/matplotlib/mpl-data/fonts/ttf"
)
MONO_DIR = FONT_DIR


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("dv",  "",  f"{FONT_DIR}/DejaVuSans.ttf")
        self.add_font("dv",  "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
        self.add_font("dv",  "I", f"{FONT_DIR}/DejaVuSans-Oblique.ttf")
        self.add_font("dv",  "BI",f"{FONT_DIR}/DejaVuSans-BoldOblique.ttf")
        self.add_font("mono","",  f"{MONO_DIR}/DejaVuSansMono.ttf")
        self.add_font("mono","B", f"{MONO_DIR}/DejaVuSansMono-Bold.ttf")

    def header(self):
        self.set_font("dv", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Brain-Connectivity-GCN \u2014 Project Plan", align="R")
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("dv", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_font("dv", "B", 13)
        self.set_text_color(25, 75, 160)
        self.ln(5)
        self.cell(0, 8, text)
        self.ln(2)
        self.set_draw_color(25, 75, 160)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
        self.set_text_color(30, 30, 30)

    def sub_title(self, text):
        self.set_font("dv", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 7, text)
        self.ln(5)

    def body(self, text):
        self.set_font("dv", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, items):
        self.set_font("dv", "", 10)
        self.set_text_color(50, 50, 50)
        w = self.epw - 5
        for item in items:
            self.set_x(self.l_margin + 5)
            self.multi_cell(w, 6, f"\u2022  {item}")
        self.ln(2)

    def kv_table(self, rows):
        for key, val in rows:
            self.set_font("dv", "B", 9)
            self.set_fill_color(235, 242, 255)
            self.cell(58, 7, key, border=1, fill=True)
            self.set_font("dv", "", 9)
            self.set_fill_color(255, 255, 255)
            self.cell(0, 7, val, border=1)
            self.ln()
        self.ln(3)

    def col_table(self, headers, rows):
        self.set_font("dv", "B", 9)
        self.set_fill_color(210, 225, 255)
        for h, w in headers:
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        for row in rows:
            alt = row[-1]
            self.set_fill_color(245, 249, 245) if alt else self.set_fill_color(255, 255, 255)
            self.set_font("dv", "", 9)
            for val, (_, w) in zip(row[:-1], headers):
                self.cell(w, 6, str(val), border=1, fill=alt)
            self.ln()
        self.ln(3)

    def code_block(self, lines):
        self.set_font("mono", "", 8)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(30, 30, 30)
        for line in lines:
            self.cell(0, 5, line, fill=True)
            self.ln()
        self.set_text_color(50, 50, 50)
        self.ln(3)


# ── Build PDF ──────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title block
pdf.set_font("dv", "B", 20)
pdf.set_text_color(20, 60, 140)
pdf.ln(4)
pdf.cell(0, 12, "Brain-Connectivity-GCN", align="C")
pdf.ln(10)
pdf.set_font("dv", "", 12)
pdf.set_text_color(70, 70, 70)
pdf.multi_cell(
    0, 7,
    "Dynamic Functional Connectivity + Graph Neural Network\n"
    "for Autism Spectrum Disorder Classification",
    align="C",
)
pdf.ln(3)
pdf.set_font("dv", "I", 10)
pdf.cell(0, 6, "Brain-specific graph-temporal classifier  |  April 2026", align="C")
pdf.ln(10)

# 1. Motivation
pdf.section_title("1.  Motivation & Goal")
pdf.body(
    "This project builds a brain-connectivity classifier that identifies "
    "Autism Spectrum Disorder (ASD) vs. Typically Developing (TD) controls from resting-state "
    "fMRI data.\n\n"
    "The key insight: temporal encoders capture how signals evolve over time, while graph "
    "layers capture ROI connectivity structure. In the brain, this maps naturally to dynamic functional "
    "connectivity \u2014 the way correlations between brain regions change across a scan. "
    "The task is not forecasting; we classify subjects based on their "
    "connectivity dynamics."
)

# 2. Dataset
pdf.section_title("2.  Dataset")
pdf.kv_table([
    ("Dataset",       "ABIDE I  (Autism Brain Imaging Data Exchange)"),
    ("Subjects",      "~884 usable  (ASD: ~403 / TD: ~481)"),
    ("Modality",      "Resting-state fMRI  (BOLD signal)"),
    ("Preprocessing", "cpac pipeline, band-pass filtering ON, GSR ON  (PCP standard)"),
    ("Atlas",         "CC200 \u2014 Craddock 2012, 200 ROIs"),
    ("Download",      "nilearn.datasets.fetch_abide_pcp  (~80 MB total)"),
    ("Split",         "Stratified 80 / 10 / 10  (train / val / test by subject label)"),
])
pdf.body(
    "CC200 was chosen because pre-extracted ROI time series are available directly "
    "(no NIfTI processing required). The pipeline is atlas-agnostic and can be swapped "
    "for Schaefer-100/200 with func_preproc + NiftiLabelsMasker."
)

# 3. Data Pipeline
pdf.section_title("3.  Data Pipeline")
pdf.sub_title("3.1  Per-subject processing steps")
pdf.bullet([
    "Load CC200 BOLD time series  \u2192  shape (T, 200)",
    "Z-score each ROI independently  (removes site-level amplitude differences)",
    "Full-scan Pearson FC  \u2192  (200, 200)  [mean_fc]",
    "Sliding window  (50 TR window, 5 TR step)  \u2192  W brain-state snapshots",
    "Per window: mean BOLD (200,) + Pearson FC (200, 200)",
    "Save to .npz cache: bold, mean_fc, bold_windows, fc_windows, label, site",
])
pdf.sub_title("3.2  Batch tensor shapes")
pdf.kv_table([
    ("bold_windows",  "(B, W, 200)  \u2014 node-feature sequence  \u2192  GCN encoder input"),
    ("adj",           "(B, 200, 200)  \u2014 adjacency matrix (see options below)"),
    ("label",         "(B,)  \u2014 0 = TD, 1 = ASD"),
])
pdf.sub_title("3.3  Adjacency options")
pdf.bullet([
    "Population adj  (default, recommended): mean FC across training subjects, |r| >= 0.2",
    "Per-subject static: subject's own mean_fc",
    "Per-subject dynamic: mean of fc_windows  (sliding-window FCs)",
])

# 4. Architecture
pdf.section_title("4.  Model Architecture  \u2014  BrainGCN Classifier")
pdf.body("Subject-level binary classifier for dynamic brain connectivity:")
pdf.bullet([
    "Input: bold_windows (B, W, N)  +  adj (B, N, N)",
    "BrainGCNCell  (per brain-state snapshot w):",
    "    GCN step:  A_hat @ [x_w, h]   where A_hat = D^{-1/2}(A+I)D^{-1/2}",
    "    GRU step:  update/reset gates use graph-convolved features",
    "After W snapshots: hidden state  (B, N, hidden_dim)",
    "Graph readout: global mean-pool across ROIs  \u2192  (B, hidden_dim)",
    "MLP head: Linear \u2192 BN \u2192 ReLU \u2192 Dropout(0.5) \u2192 Linear(2)  \u2192  logits",
    "Loss: cross-entropy  |  Optimizer: Adam  |  Metrics: Accuracy, AUC-ROC, F1",
])

pdf.add_page()

# 5. Differences table
pdf.section_title("5.  Key Design Choices")
pdf.col_table(
    [("Aspect", 45), ("Old reference direction", 72), ("BrainGCN (This project)", 73)],
    [
        ("Domain",        "External graph sequence", "Resting-state fMRI",       False),
        ("Nodes",         "Road sensors",          "Brain ROIs (200, CC200)",     True),
        ("Node features", "Speed at time t",       "Mean BOLD in window w",       False),
        ("Adjacency",     "Fixed road graph",      "FC matrix (data-driven)",     True),
        ("Task",          "Per-node regression",   "Subject classification",      False),
        ("Output",        "Future speed values",   "ASD / TD label",              True),
        ("Loss",          "MSE",                   "Cross-entropy",               False),
        ("Graph readout", "None (per-node)",        "Global mean-pool",           True),
        ("Dataset",       "Legacy sensor datasets", "ABIDE I  (~884 subjects)",  False),
    ]
)

# 6. Project structure
pdf.section_title("6.  Project Structure")
pdf.code_block([
    "Brain-Connectivity-GCN/",
    "  brain_gcn/",
    "    utils/",
    "      graph_conv.py                  # Batched adjacency normalisation",
    "      data/",
    "        download.py                  # ABIDE I download via nilearn",
    "        functional_connectivity.py   # FC, sliding-window, thresholding",
    "        preprocess.py                # z-score + .npz cache per subject",
    "        dataset.py                   # ABIDEDataset (PyTorch Dataset)",
    "        datamodule.py                # ABIDEDataModule (Lightning DataModule)",
    "    models/                          # [next]  BrainGCNCell, BrainGCNClassifier",
    "    tasks/                           # [next]  ClassificationTask",
    "    main.py                          # [next]  training entry point",
    "  data/",
    "    raw/                             # nilearn cache (.1D files)",
    "    processed/                       # per-subject .npz files",
    "  references/",
    "    graph_temporal_reference/        # archived background reference only",
    "  Brain_Connectivity_GCN_Plan.pdf    # this document",
])

# 7. Roadmap
pdf.section_title("7.  Implementation Roadmap")
pdf.col_table(
    [("#", 8), ("Component", 90), ("Status", 25), ("Notes", 67)],
    [
        ("1", "Data pipeline (download, FC, dataset, datamodule)", "DONE",  "Tested with 20 subjects",        False),
        ("2", "Graph-temporal classifier",                         "DONE",  "GCN projection + GRU encoder",     True),
        ("3", "Attention / mean ROI readout",                      "DONE",  "subject-level graph pooling",      False),
        ("4", "ClassificationTask  (Lightning Module)",            "DONE",  "CE loss, AUC, F1",                True),
        ("5", "main.py + argparse training entry point",           "DONE",  "cached-data training CLI",         False),
        ("6", "Ablation baselines  (GCN-only, GRU-only)",          "TODO",  "isolate spatial vs temporal",     True),
        ("7", "Site-stratified cross-validation",                  "TODO",  "ABIDE benchmark comparison",      False),
        ("8", "Attention readout + ROI importance maps",           "TODO",  "interpretability",                True),
    ]
)

# 8. Hyperparameters
pdf.section_title("8.  Default Hyperparameters")
pdf.kv_table([
    ("Atlas / ROIs",      "CC200  \u2014  200 ROIs"),
    ("Window length",     "50 TRs  (~100 s at TR = 2 s)"),
    ("Window step",       "5 TRs"),
    ("Max windows (W)",   "30  (truncate for uniform batch shape)"),
    ("FC threshold",      "|r| >= 0.2  (sparsify adjacency)"),
    ("Hidden dim",        "64"),
    ("MLP dropout",       "0.5"),
    ("Batch size",        "32"),
    ("Learning rate",     "1e-3  (Adam)"),
    ("Weight decay",      "1e-4"),
    ("Max epochs",        "200  with early stopping on val AUC-ROC"),
])

# 9. References
pdf.section_title("9.  Key References")
pdf.bullet([
    "Di Martino et al. (2014). The Autism Brain Imaging Data Exchange. Mol. Psychiatry.",
    "Craddock et al. (2012). A whole brain fMRI atlas via spectral clustering. HBM.",
    "Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.",
    "Li et al. (2021). BrainGNN: Interpretable Brain Graph Neural Network. Med. Image Anal.",
    "Hutchison et al. (2013). Dynamic functional connectivity: Promise, issues, interpretations. NeuroImage.",
])

out = "/home/Yatsuiii/Brain-Connectivity-GCN/Brain_Connectivity_GCN_Plan.pdf"
pdf.output(out)
print(f"PDF saved to {out}")
