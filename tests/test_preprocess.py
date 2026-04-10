"""
Unit tests for preprocessing utilities.

Tests:
  - Framewise displacement (FD) computation (Power 2012)
  - Motion scrubbing: correct threshold and output shape
  - Z-score normalization
"""

import numpy as np
import pytest

from brain_gcn.utils.data.preprocess import (
    compute_fd,
    load_motion_params,
    scrub_bold,
)


class TestFramewiseDisplacement:
    """Test Power 2012 FD computation."""

    def test_fd_shape(self):
        """FD output shape should be (T-1,) for input (T, 6)."""
        motion_params = np.random.randn(100, 6).astype(np.float32)
        fd = compute_fd(motion_params)
        assert fd.shape == (99,)

    def test_fd_zero_motion(self):
        """Zero motion should yield zero FD."""
        motion_params = np.zeros((50, 6), dtype=np.float32)
        fd = compute_fd(motion_params)
        assert np.allclose(fd, 0.0)

    def test_fd_translation_only(self):
        """Translation without rotation should be counted 1:1."""
        motion_params = np.zeros((10, 6), dtype=np.float32)
        motion_params[1:, :3] = 1.0  # 1mm translation in all directions from step 1 onward
        fd = compute_fd(motion_params)
        # diff[0] = [1,1,1,0,0,0], sum = 3 mm (motion from 0 to step 1)
        # diff[1:] = [0,0,0,0,0,0], sum = 0 mm (no change after step 1)
        assert np.isclose(fd[0], 3.0, atol=0.01), f"Expected fd[0]≈3.0, got {fd[0]}"
        assert np.allclose(fd[1:], 0.0, atol=0.01), f"Expected fd[1:]≈0.0, got {fd[1:]}"

    def test_fd_rotation_conversion(self):
        """Rotation in radians should be converted to mm using 50mm sphere radius."""
        motion_params = np.zeros((10, 6), dtype=np.float32)
        rot_rad = 0.1  # 0.1 radians
        motion_params[1, 3] = rot_rad  # rotation around x at step 1 only
        fd = compute_fd(motion_params)
        # diff[0] = step[1] - step[0] = [0,0,0,0.1,0,0], rotation: 0.1*50=5mm, fd[0]=5mm
        # diff[1] = step[2] - step[1] = [0,0,0,-0.1,0,0], rotation: 0.1*50=5mm, fd[1]=5mm
        # diff[2:] = [0,0,0,0,0,0], fd[2:]=0mm
        assert np.isclose(fd[0], 5.0, atol=0.01), f"Expected fd[0]≈5.0, got {fd[0]}"
        assert np.isclose(fd[1], 5.0, atol=0.01), f"Expected fd[1]≈5.0, got {fd[1]}"
        assert np.allclose(fd[2:], 0.0, atol=0.01), f"Expected fd[2:]≈0.0, got {fd[2:]}"


class TestMotionScrubbing:
    """Test motion scrubbing with FD threshold."""

    def test_scrub_removes_high_motion_trs(self):
        """TRs with FD > threshold should be removed."""
        bold = np.random.randn(100, 50).astype(np.float32)
        fd = np.zeros(99)
        fd[10:20] = 1.0  # TRs 11-20 have high FD
        cleaned = scrub_bold(bold, fd, fd_threshold=0.5, min_clean_trs=50)
        assert cleaned is not None
        assert cleaned.shape[0] < 100  # Some TRs removed
        assert cleaned.shape[1] == 50  # ROI dimension preserved

    def test_scrub_insufficient_clean_trs(self):
        """Return None if cleaned TRs < min_clean_trs."""
        bold = np.random.randn(50, 50).astype(np.float32)
        fd = np.ones(49) * 1.0  # All TRs high motion
        cleaned = scrub_bold(bold, fd, fd_threshold=0.5, min_clean_trs=40)
        assert cleaned is None

    def test_scrub_preserves_clean_trs(self):
        """TRs with FD < threshold should be preserved."""
        bold = np.random.randn(100, 50).astype(np.float32)
        fd = np.ones(99) * 0.1  # All low FD
        cleaned = scrub_bold(bold, fd, fd_threshold=0.5, min_clean_trs=50)
        assert cleaned is not None
        assert cleaned.shape[0] == 100  # All TRs kept

    def test_scrub_tr0_always_kept(self):
        """TR 0 should always be kept (no preceding frame to compare)."""
        bold = np.arange(100 * 50, dtype=np.float32).reshape(100, 50)
        fd = np.ones(99) * 1.0  # All high FD
        cleaned = scrub_bold(bold, fd, fd_threshold=0.5, min_clean_trs=1)
        assert cleaned is not None
        # Only TR 0 should remain
        assert cleaned.shape[0] == 1
        # TR 0 should be first row (indices 0-49 from arange)
        assert np.allclose(cleaned[0], bold[0])


class TestLoadMotionParams:
    """Test motion parameter loading from subject dict."""

    def test_load_missing_confounds(self):
        """Missing confounds should return None."""
        subject = {"subject_id": "test", "bold": np.random.randn(100, 50)}
        params = load_motion_params(subject)
        assert params is None

    def test_load_valid_confounds(self):
        """Valid 6-col confounds should be returned."""
        confounds = np.random.randn(100, 6).astype(np.float32)
        subject = {"subject_id": "test", "confounds": confounds}
        params = load_motion_params(subject)
        assert params is not None
        assert params.shape == (100, 6)

    def test_load_insufficient_columns(self):
        """Confounds with < 6 columns should return None."""
        confounds = np.random.randn(100, 3)
        subject = {"subject_id": "test", "confounds": confounds}
        params = load_motion_params(subject)
        assert params is None
