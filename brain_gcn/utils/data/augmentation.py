"""BOLD augmentation, connectivity, and signal-processing helpers.

All functions accept time-by-ROI arrays and preserve ``float32`` inputs where
practical. Random augmentations accept an optional NumPy generator so experiments
can remain reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score

from .preprocess import compute_fd, scrub_bold


class BoldAugmentation:
    """Shape-preserving augmentations for BOLD time series."""

    @staticmethod
    def gaussian_noise(bold, std=0.005, rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        return (x + rng.normal(0.0, std, x.shape)).astype(x.dtype, copy=False)

    @staticmethod
    def temporal_jitter(bold, jitter_std=0.5, rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        out = np.empty_like(x)
        shifts = np.rint(rng.normal(0.0, jitter_std, x.shape[1])).astype(int)
        for roi, shift in enumerate(shifts):
            out[:, roi] = np.roll(x[:, roi], shift)
        return out

    @staticmethod
    def roi_dropout(bold, dropout_rate=0.1, rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        out = x.copy()
        out[:, rng.random(x.shape[1]) < dropout_rate] = 0
        return out

    @staticmethod
    def frequency_dropout(bold, dropout_rate=0.05, rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        spectrum = np.fft.rfft(x, axis=0)
        mask = rng.random(spectrum.shape) >= dropout_rate
        mask[0] = True
        return np.fft.irfft(spectrum * mask, n=x.shape[0], axis=0).astype(x.dtype)

    @staticmethod
    def time_warping(bold, max_warp=0.05, rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        t = np.linspace(0.0, 1.0, x.shape[0])
        exponent = rng.uniform(1.0 - max_warp, 1.0 + max_warp)
        warped_t = np.clip(t**exponent, 0.0, 1.0)
        out = np.empty_like(x)
        for roi in range(x.shape[1]):
            out[:, roi] = np.interp(t, warped_t, x[:, roi])
        return out

    @staticmethod
    def amplitude_scaling(bold, scale_range=(0.9, 1.1), rng=None):
        rng = rng or np.random.default_rng()
        x = np.asarray(bold)
        scales = rng.uniform(*scale_range, size=(1, x.shape[1]))
        return (x * scales).astype(x.dtype, copy=False)


class AugmentationPipeline:
    """Sequentially apply named :class:`BoldAugmentation` operations."""

    def __init__(self, augmentations: Sequence[tuple[str, dict]]):
        self.augmentations = list(augmentations)

    def apply(self, bold, rng=None):
        rng = rng or np.random.default_rng()
        out = np.asarray(bold).copy()
        for name, kwargs in self.augmentations:
            operation = getattr(BoldAugmentation, name, None)
            if operation is None:
                raise ValueError(f"Unknown BOLD augmentation: {name}")
            out = operation(out, rng=rng, **kwargs)
        return out

    @classmethod
    def light(cls):
        return cls([("gaussian_noise", {"std": 0.005})])

    @classmethod
    def moderate(cls):
        return cls([
            ("gaussian_noise", {"std": 0.01}),
            ("temporal_jitter", {"jitter_std": 0.5}),
        ])

    @classmethod
    def aggressive(cls):
        return cls([
            ("gaussian_noise", {"std": 0.02}),
            ("temporal_jitter", {"jitter_std": 1.0}),
            ("roi_dropout", {"dropout_rate": 0.1}),
            ("amplitude_scaling", {"scale_range": (0.8, 1.2)}),
        ])


class FunctionalConnectivityMeasures:
    """Functional-connectivity estimators returning ROI-by-ROI matrices."""

    @staticmethod
    def pearson_correlation(bold):
        fc = np.corrcoef(np.asarray(bold), rowvar=False)
        return np.nan_to_num(fc).astype(np.float32)

    @staticmethod
    def partial_correlation(bold, ridge=1e-5):
        covariance = np.cov(np.asarray(bold), rowvar=False)
        precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
        scale = np.sqrt(np.clip(np.diag(precision), ridge, None))
        partial = -precision / np.outer(scale, scale)
        np.fill_diagonal(partial, 1.0)
        return np.nan_to_num(partial).astype(np.float32)

    @staticmethod
    def mutual_information(bold, bins=10):
        x = np.asarray(bold)
        n_rois = x.shape[1]
        discretized = np.empty_like(x, dtype=np.int32)
        for roi in range(n_rois):
            edges = np.histogram_bin_edges(x[:, roi], bins=bins)
            discretized[:, roi] = np.digitize(x[:, roi], edges[1:-1])
        result = np.eye(n_rois, dtype=np.float32)
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                value = mutual_info_score(discretized[:, i], discretized[:, j])
                result[i, j] = result[j, i] = value
        return result

    @staticmethod
    def coherence(bold, freq_range=(0.01, 0.1), fs=0.5):
        x = np.asarray(bold)
        n_rois = x.shape[1]
        result = np.eye(n_rois, dtype=np.float32)
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                frequencies, values = signal.coherence(x[:, i], x[:, j], fs=fs)
                band = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
                value = float(values[band].mean()) if band.any() else 0.0
                result[i, j] = result[j, i] = value
        return result


class SignalPreprocessing:
    """Common research preprocessing operations."""

    @staticmethod
    def bandpass_filter(bold, freq_range=(0.01, 0.1), fs=0.5, order=4):
        x = np.asarray(bold)
        sos = signal.butter(order, freq_range, btype="bandpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, x, axis=0).astype(x.dtype)

    @staticmethod
    def motion_scrubbing(bold, motion, threshold=0.5, min_clean_trs=1):
        fd = compute_fd(motion)
        return scrub_bold(bold, fd, threshold, min_clean_trs)

    @staticmethod
    def ica_denoise(bold, n_components=20, random_state=42):
        x = np.asarray(bold)
        n_components = min(n_components, *x.shape)
        ica = FastICA(n_components=n_components, random_state=random_state, whiten="unit-variance")
        sources = ica.fit_transform(x)
        return ica.inverse_transform(sources).astype(x.dtype)


class MultiSiteNormalization:
    """Simple site-wise location/scale harmonization for exploratory use."""

    @staticmethod
    def harmonization(bold_list, sites, epsilon=1e-6):
        if len(bold_list) != len(sites):
            raise ValueError("bold_list and sites must have the same length")
        arrays = [np.asarray(x) for x in bold_list]
        global_data = np.concatenate(arrays, axis=0)
        global_mean = global_data.mean(axis=0, keepdims=True)
        global_std = global_data.std(axis=0, keepdims=True)
        by_site = {}
        for site_name, array in zip(sites, arrays):
            by_site.setdefault(site_name, []).append(array)
        stats = {
            site_name: (
                np.concatenate(group, axis=0).mean(axis=0, keepdims=True),
                np.concatenate(group, axis=0).std(axis=0, keepdims=True),
            )
            for site_name, group in by_site.items()
        }
        return [
            (((array - stats[site_name][0]) / (stats[site_name][1] + epsilon))
             * (global_std + epsilon) + global_mean).astype(array.dtype)
            for array, site_name in zip(arrays, sites)
        ]
