#!/usr/bin/env python3
"""
Create synthetic ABIDE-like data for quick testing.
Generates minimal .npz files compatible with the training pipeline.
"""
import sys
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def create_synthetic_data(n_subjects=50, n_rois=200, window_len=50, num_windows=20):
    """Create synthetic dataset."""
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating synthetic ABIDE dataset...")
    print(f"  Subjects: {n_subjects}")
    print(f"  ROIs: {n_rois}")
    print(f"  Windows per subject: {num_windows}")
    
    subjects_created = 0
    for subj_id in range(1, n_subjects + 1):
        # Label: roughly 50-50 split
        label = subj_id % 2
        
        # Random BOLD signal: (num_windows, n_rois)
        # Use realistic scale (0-500 arbitrary intensity units)
        bold_windows = np.random.randn(num_windows, n_rois).astype(np.float32)
        bold_windows = (bold_windows * 50 + 300)  # Shift to 300±50
        bold_windows = np.clip(bold_windows, 0, 1000)
        
        # Get mean FC for this subject
        # Correlation between random signals gives values in [-1, 1]
        mean_bold = bold_windows.mean(axis=0)  # (n_rois,)
        
        # Simple FC: correlation-like matrix
        mean_fc = np.random.randn(n_rois, n_rois).astype(np.float32)
        mean_fc = (mean_fc + mean_fc.T) / 2  # Make symmetric
        mean_fc = np.clip(mean_fc / (np.std(mean_fc) + 1e-8), -1, 1)  # Normalize to [-1,1]
        
        # Per-window FC matrices
        fc_windows = np.zeros((num_windows, n_rois, n_rois), dtype=np.float32)
        for w in range(num_windows):
            # Each window gets slightly different FC
            fc_w = np.random.randn(n_rois, n_rois).astype(np.float32)
            fc_w = (fc_w + fc_w.T) / 2
            fc_w = np.clip(fc_w / (np.std(fc_w) + 1e-8), -1, 1)
            fc_windows[w] = fc_w
        
        # Save
        output_file = processed_dir / f"sub-{subj_id:04d}_label-{label}.npz"
        np.savez(
            output_file,
            bold_windows=bold_windows,
            mean_fc=mean_fc,
            fc_windows=fc_windows,
            label=label,
            subject_id=f"sub-{subj_id:04d}",
            site="synthetic",
        )
        subjects_created += 1
        
        if subjects_created % 10 == 0:
            print(f"  Created {subjects_created}/{n_subjects}...")
    
    print(f"✅ Created {subjects_created} synthetic subjects in {processed_dir}")
    print(f"   Ready for training!")

if __name__ == "__main__":
    create_synthetic_data(n_subjects=60)
