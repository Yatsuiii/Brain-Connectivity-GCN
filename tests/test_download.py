"""
Unit tests for data download & loading.

Tests:
  - Label mapping assertion catches invalid DX_GROUP values
  - Subject extraction and validation
  - Phenotypic filtering
"""

import numpy as np
import pandas as pd
import pytest

from brain_gcn.utils.data.download import extract_subjects, get_label


class TestLabelMapping:
    """Test ASD/TD label mapping and validation."""

    def test_get_label_asd(self):
        """DX_GROUP=1 should map to label=1 (ASD)."""
        row = pd.Series({"DX_GROUP": 1})
        assert get_label(row) == 1

    def test_get_label_tc(self):
        """DX_GROUP=2 should map to label=0 (TC)."""
        row = pd.Series({"DX_GROUP": 2})
        assert get_label(row) == 0

    def test_extract_subjects_validates_labels(self):
        """extract_subjects should assert valid DX_GROUP values (1 or 2)."""
        # Create a mock dataset with invalid label
        phenotypic = pd.DataFrame({
            "SUB_ID": ["sub_001", "sub_002"],
            "DX_GROUP": [1, 3],  # 3 is invalid
            "SITE_ID": ["site_a", "site_b"],
        })

        # Mock Bunch-like object
        class MockDataset:
            def __init__(self):
                self.phenotypic = phenotypic
                self.rois_cc200 = [
                    np.random.randn(150, 200).astype(np.float32),
                    np.random.randn(150, 200).astype(np.float32),
                ]

        dataset = MockDataset()
        with pytest.raises(AssertionError, match="Unexpected DX_GROUP"):
            extract_subjects(dataset)

    def test_extract_subjects_valid_labels(self):
        """extract_subjects should succeed with valid DX_GROUP (1 or 2)."""
        phenotypic = pd.DataFrame({
            "SUB_ID": ["sub_001", "sub_002"],
            "DX_GROUP": [1, 2],  # Valid: ASD and TC
            "SITE_ID": ["site_a", "site_b"],
        })

        class MockDataset:
            def __init__(self):
                self.phenotypic = phenotypic
                self.rois_cc200 = [
                    np.random.randn(150, 200).astype(np.float32),
                    np.random.randn(150, 200).astype(np.float32),
                ]

        dataset = MockDataset()
        subjects = extract_subjects(dataset, min_timepoints=100)
        assert len(subjects) == 2
        assert subjects[0]["label"] == 1  # ASD
        assert subjects[1]["label"] == 0  # TC

    def test_extract_subjects_min_timepoints(self):
        """Subjects with insufficient TRs should be excluded."""
        phenotypic = pd.DataFrame({
            "SUB_ID": ["sub_001", "sub_002"],
            "DX_GROUP": [1, 2],
            "SITE_ID": ["site_a", "site_b"],
        })

        class MockDataset:
            def __init__(self):
                self.phenotypic = phenotypic
                self.rois_cc200 = [
                    np.random.randn(50, 200).astype(np.float32),   # Too short
                    np.random.randn(150, 200).astype(np.float32),  # Valid
                ]

        dataset = MockDataset()
        subjects = extract_subjects(dataset, min_timepoints=100)
        assert len(subjects) == 1
        assert subjects[0]["subject_id"] == "sub_002"

    def test_extract_subjects_nan_rejection(self):
        """Subjects with NaN/Inf values should be excluded."""
        phenotypic = pd.DataFrame({
            "SUB_ID": ["sub_001", "sub_002"],
            "DX_GROUP": [1, 2],
            "SITE_ID": ["site_a", "site_b"],
        })

        bold_nan = np.random.randn(150, 200).astype(np.float32)
        bold_nan[10, 5] = np.nan

        class MockDataset:
            def __init__(self):
                self.phenotypic = phenotypic
                self.rois_cc200 = [
                    bold_nan,  # Invalid: has NaN
                    np.random.randn(150, 200).astype(np.float32),  # Valid
                ]

        dataset = MockDataset()
        subjects = extract_subjects(dataset, min_timepoints=100)
        assert len(subjects) == 1
        assert subjects[0]["subject_id"] == "sub_002"
