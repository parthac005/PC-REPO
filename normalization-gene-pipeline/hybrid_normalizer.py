"""
Hybrid Normalization Pipeline for Bulk RNA-seq Differential Expression Analysis
================================================================================
Stage 1: Statistical normalization  (TMM / DESeq2-style size factors + RUVSeq SVs)
Stage 2: Deep learning refinement   (Variational Autoencoder for latent correction)
Stage 3: DE testing                 (DESeq2-style NB GLM with LRT / Wald test)

Requirements:
    pip install torch numpy pandas scipy scikit-learn matplotlib seaborn anndata
    pip install pydeseq2          # DESeq2 Python port
    pip install rpy2              # optional — only needed for R-based TMM/RUVSeq
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import nbinom
from scipy.special import digamma
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Statistical Normalization
# ---------------------------------------------------------------------------

class StatisticalNormalizer:
    """
    Stage 1: Classical statistical normalization for bulk RNA-seq.

    Steps applied in order:
      1. TMM size-factor estimation (Robinson & Oshlack 2010)
      2. GC-content / length bias correction (CQN-style offset, optional)
      3. Surrogate variable estimation for unwanted variation (RUVg-style SVD)

    Parameters
    ----------
    method : str
        Primary size-factor method — 'tmm' or 'deseq2'.
    n_svs : int
        Number of surrogate variables to estimate (0 = skip RUV step).
    gc_correction : bool
        Whether to apply a loess-based GC / length offset.
    """

    def __init__(self, method: str = "tmm", n_svs: int = 2, gc_correction: bool = False):
        self.method = method
        self.n_svs = n_svs
        self.gc_correction = gc_correction

        self.size_factors_: Optional[np.ndarray] = None
        self.sv_matrix_: Optional[np.ndarray] = None
        self.norm_counts_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        counts: np.ndarray,
        gene_info: Optional[pd.DataFrame] = None,
        control_genes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit the normalizer and return normalized log-counts.

        Parameters
        ----------
        counts : array (n_samples, n_genes)   raw integer counts
        gene_info : DataFrame with columns 'gc_content' and 'length' (optional)
        control_genes : bool index of housekeeping / spike-in genes for RUV

        Returns
        -------
        norm_log : array (n_samples, n_genes)  log1p normalized counts
        """
        log.info("Stage 1 — statistical normalization  [method=%s, n_svs=%d]", self.method, self.n_svs)

        # 1. Size-factor normalization
        if self.method == "tmm":
            sf = self._tmm_size_factors(counts)
        else:
            sf = self._deseq2_size_factors(counts)
        self.size_factors_ = sf
        norm = counts / sf[:, None]   # (n_samples, n_genes)
        log.info("  Size factors  min=%.3f  max=%.3f  median=%.3f", sf.min(), sf.max(), np.median(sf))

        # 2. Optional GC / length correction
        if self.gc_correction and gene_info is not None:
            norm = self._gc_length_correction(norm, gene_info)
            log.info("  GC / length correction applied")

        # 3. Surrogate variable estimation
        if self.n_svs > 0:
            sv, norm = self._ruv_correction(norm, control_genes)
            self.sv_matrix_ = sv
            log.info("  Estimated %d surrogate variable(s)", self.n_svs)

        self.norm_counts_ = norm
        norm_log = np.log1p(norm)
        log.info("  Stage 1 complete — output shape %s", norm_log.shape)
        return norm_log

    # ------------------------------------------------------------------
    # TMM size-factor estimation
    # ------------------------------------------------------------------

    def _tmm_size_factors(self, counts: np.ndarray) -> np.ndarray:
        """
        Trimmed mean of M-values (TMM).
        Reference sample = sample closest to the 75th-percentile library size.
        """
        lib_sizes = counts.sum(axis=1)
        ref_idx = np.argmin(np.abs(lib_sizes - np.percentile(lib_sizes, 75)))

        sf = np.ones(counts.shape[0])
        ref = counts[ref_idx]

        for i in range(counts.shape[0]):
            sf[i] = self._tmm_factor(counts[i], ref)

        # Normalize so geometric mean = 1
        sf /= np.exp(np.mean(np.log(sf + 1e-8)))
        return sf

    def _tmm_factor(
        self,
        sample: np.ndarray,
        ref: np.ndarray,
        log_ratio_trim: float = 0.30,
        sum_trim: float = 0.05,
        min_count: int = 5,
    ) -> float:
        mask = (sample >= min_count) & (ref >= min_count)
        if mask.sum() < 10:
            return 1.0

        s, r = sample[mask].astype(float), ref[mask].astype(float)
        ns, nr = s.sum(), r.sum()

        M = np.log2(s / ns) - np.log2(r / nr)                 # log-ratio
        A = 0.5 * (np.log2(s / ns) + np.log2(r / nr))         # average log-intensity
        v = (ns - s) / (ns * s) + (nr - r) / (nr * r)         # asymptotic variance

        # Remove Inf / NaN
        ok = np.isfinite(M) & np.isfinite(A) & np.isfinite(v)
        M, A, v = M[ok], A[ok], v[ok]

        if len(M) == 0:
            return 1.0

        # Trim on M and A
        lm, um = np.quantile(M, [log_ratio_trim, 1 - log_ratio_trim])
        la, ua = np.quantile(A, [sum_trim, 1 - sum_trim])
        keep = (M >= lm) & (M <= um) & (A >= la) & (A <= ua)

        if keep.sum() == 0:
            return 1.0

        w = 1.0 / v[keep]
        return float(2 ** (np.sum(w * M[keep]) / np.sum(w)))

    # ------------------------------------------------------------------
    # DESeq2-style size factors (median-of-ratios)
    # ------------------------------------------------------------------

    def _deseq2_size_factors(self, counts: np.ndarray) -> np.ndarray:
        log_counts = np.log(counts + 0.5)
        log_geo_means = log_counts.mean(axis=0)
        ratios = log_counts - log_geo_means[None, :]
        sf = np.exp(np.median(ratios, axis=1))
        sf /= np.exp(np.mean(np.log(sf)))
        return sf

    # ------------------------------------------------------------------
    # GC-content / length correction (loess offset, CQN-inspired)
    # ------------------------------------------------------------------

    def _gc_length_correction(
        self, norm: np.ndarray, gene_info: pd.DataFrame
    ) -> np.ndarray:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        gc = gene_info["gc_content"].values
        ln = np.log(gene_info["length"].values + 1)
        X = np.column_stack([gc, gc ** 2, ln])
        corrected = norm.copy()

        for i in range(norm.shape[0]):
            y = np.log1p(norm[i])
            reg = LinearRegression().fit(X, y)
            bias = reg.predict(X) - reg.predict(X).mean()
            corrected[i] = np.expm1(y - bias)

        return corrected

    # ------------------------------------------------------------------
    # RUV-style surrogate variable estimation
    # ------------------------------------------------------------------

    def _ruv_correction(
        self,
        norm: np.ndarray,
        control_genes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SVD-based unwanted variation removal (RUVg-style)."""
        log_norm = np.log1p(norm)

        if control_genes is not None and control_genes.sum() > 0:
            ctl = log_norm[:, control_genes]
        else:
            # Fallback: use the least-variable genes as pseudo-controls
            gene_cv = log_norm.std(axis=0) / (log_norm.mean(axis=0) + 1e-8)
            n_ctl = max(50, norm.shape[1] // 10)
            ctl_idx = np.argsort(gene_cv)[:n_ctl]
            ctl = log_norm[:, ctl_idx]

        # Center and SVD
        ctl_c = ctl - ctl.mean(axis=0)
        U, S, Vt = np.linalg.svd(ctl_c, full_matrices=False)
        sv = U[:, : self.n_svs]   # (n_samples, n_svs)

        # Regress SVs out of every gene
        coef = np.linalg.lstsq(sv, log_norm, rcond=None)[0]  # (n_svs, n_genes)
        corrected_log = log_norm - sv @ coef
        corrected = np.expm1(corrected_log)
        return sv, corrected
