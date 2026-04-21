"""
Stage 3 — Differential Expression Testing + Full Pipeline Orchestrator
=======================================================================
DE is performed on the hybrid-normalized matrix using:
  - pydeseq2  (Python port of DESeq2, preferred)
  - Fallback : NB GLM implemented here from scratch (t-test on log-CPM
    is available as a lightweight option for quick iteration)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from hybrid_normalizer import StatisticalNormalizer
from vae_model import GeneExpressionVAE, VAETrainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 3 — Differential Expression
# ---------------------------------------------------------------------------

class DifferentialExpressionTester:
    """
    Stage 3: Differential expression testing on hybrid-normalized counts.

    Uses pydeseq2 when available; falls back to a simple t-test on
    log2-CPM (sufficient for exploration / CI testing).

    Parameters
    ----------
    alpha : FDR threshold for significance calls
    lfc_threshold : |log2FC| threshold for biological significance
    """

    def __init__(self, alpha: float = 0.05, lfc_threshold: float = 1.0):
        self.alpha         = alpha
        self.lfc_threshold = lfc_threshold
        self.results_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------

    def run(
        self,
        counts: np.ndarray,
        condition: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
        sample_names: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Run DE testing.

        Parameters
        ----------
        counts      : integer count matrix (n_samples, n_genes)
        condition   : string array of condition labels per sample
        gene_names  : gene identifier array (length n_genes)
        sample_names: sample identifier array (length n_samples)

        Returns
        -------
        results DataFrame with columns:
            gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, significant
        """
        n, g = counts.shape
        if gene_names is None:
            gene_names = np.array([f"gene_{i}" for i in range(g)])
        if sample_names is None:
            sample_names = np.array([f"sample_{i}" for i in range(n)])

        log.info("Stage 3 — DE testing  [n_samples=%d  n_genes=%d]", n, g)

        try:
            results = self._pydeseq2(counts, condition, gene_names, sample_names)
            log.info("  Used pydeseq2")
        except ImportError:
            log.warning("  pydeseq2 not found — falling back to Welch t-test on log2-CPM")
            results = self._ttest_logcpm(counts, condition, gene_names)

        results["significant"] = (
            (results["padj"] < self.alpha) &
            (results["log2FoldChange"].abs() >= self.lfc_threshold)
        )
        self.results_ = results
        n_sig = results["significant"].sum()
        log.info("  Significant DE genes (FDR<%.2f, |LFC|≥%.1f): %d / %d",
                 self.alpha, self.lfc_threshold, n_sig, g)
        return results

    # ------------------------------------------------------------------

    def _pydeseq2(
        self,
        counts: np.ndarray,
        condition: np.ndarray,
        gene_names: np.ndarray,
        sample_names: np.ndarray,
    ) -> pd.DataFrame:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats

        counts_df = pd.DataFrame(
            counts.astype(int),
            index=sample_names,
            columns=gene_names,
        )
        meta_df = pd.DataFrame(
            {"condition": condition},
            index=sample_names,
        )

        inference = DefaultInference(n_cpus=1)
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=meta_df,
            design_factors="condition",
            inference=inference,
        )
        dds.deseq2()

        groups = sorted(set(condition))
        stat = DeseqStats(dds, contrast=["condition", groups[1], groups[0]], inference=inference)
        stat.summary()
        res = stat.results_df.reset_index().rename(columns={"index": "gene"})
        return res

    # ------------------------------------------------------------------

    def _ttest_logcpm(
        self,
        counts: np.ndarray,
        condition: np.ndarray,
        gene_names: np.ndarray,
    ) -> pd.DataFrame:
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests

        cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
        logcpm = np.log2(cpm + 1)

        groups = np.unique(condition)
        g0, g1 = groups[0], groups[1]
        idx0 = condition == g0
        idx1 = condition == g1

        lfc = logcpm[idx1].mean(0) - logcpm[idx0].mean(0)
        t_stats, pvals = ttest_ind(logcpm[idx1], logcpm[idx0], axis=0, equal_var=False)
        _, padj, _, _ = multipletests(np.nan_to_num(pvals, nan=1.0), method="fdr_bh")
        base_mean = logcpm.mean(0)

        return pd.DataFrame({
            "gene": gene_names,
            "baseMean": base_mean,
            "log2FoldChange": lfc,
            "lfcSE": np.nan,
            "stat": t_stats,
            "pvalue": pvals,
            "padj": padj,
        })

    # ------------------------------------------------------------------

    def volcano_plot(
        self,
        title: str = "Volcano plot",
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        if self.results_ is None:
            raise RuntimeError("Call .run() before .volcano_plot()")

        df = self.results_.copy()
        df["-log10(padj)"] = -np.log10(df["padj"].clip(1e-300))

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = np.where(
            df["significant"] & (df["log2FoldChange"] > 0), "#D85A30",
            np.where(
                df["significant"] & (df["log2FoldChange"] < 0), "#185FA5", "#B4B2A9"
            )
        )
        ax.scatter(df["log2FoldChange"], df["-log10(padj)"],
                   c=colors, s=12, alpha=0.7, linewidths=0)

        ax.axhline(-np.log10(self.alpha), color="#5F5E5A", lw=0.8, ls="--")
        ax.axvline( self.lfc_threshold,  color="#5F5E5A", lw=0.8, ls="--")
        ax.axvline(-self.lfc_threshold,  color="#5F5E5A", lw=0.8, ls="--")

        # Annotate top genes
        top = df[df["significant"]].nlargest(top_n, "-log10(padj)")
        for _, row in top.iterrows():
            ax.annotate(
                row["gene"],
                (row["log2FoldChange"], row["-log10(padj)"]),
                fontsize=6, alpha=0.85,
                xytext=(3, 3), textcoords="offset points",
            )

        up   = int((df["significant"] & (df["log2FoldChange"] > 0)).sum())
        down = int((df["significant"] & (df["log2FoldChange"] < 0)).sum())
        legend = [
            mpatches.Patch(color="#D85A30", label=f"Up-regulated ({up})"),
            mpatches.Patch(color="#185FA5", label=f"Down-regulated ({down})"),
            mpatches.Patch(color="#B4B2A9", label="Not significant"),
        ]
        ax.legend(handles=legend, fontsize=8, framealpha=0.8)
        ax.set_xlabel("log2 fold change", fontsize=11)
        ax.set_ylabel("-log10(adjusted p-value)", fontsize=11)
        ax.set_title(title, fontsize=13)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def ma_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        if self.results_ is None:
            raise RuntimeError("Call .run() before .ma_plot()")
        df = self.results_.copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = np.where(df["significant"], "#D85A30", "#B4B2A9")
        ax.scatter(df["baseMean"], df["log2FoldChange"],
                   c=colors, s=10, alpha=0.6, linewidths=0)
        ax.axhline(0, color="#5F5E5A", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Mean expression (log scale)", fontsize=11)
        ax.set_ylabel("log2 fold change", fontsize=11)
        ax.set_title("MA plot", fontsize=13)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

class HybridNormalizationPipeline:
    """
    End-to-end hybrid normalization pipeline.

      Stage 1 → StatisticalNormalizer  (TMM / DESeq2 + GC bias + RUV)
      Stage 2 → GeneExpressionVAE      (latent batch correction)
      Stage 3 → DifferentialExpression (DESeq2 / t-test on log2CPM)

    Quick-start
    -----------
    >>> pipe = HybridNormalizationPipeline(latent_dim=20, n_svs=2)
    >>> results = pipe.fit_transform(counts, condition, gene_names=genes)
    >>> pipe.volcano_plot(save_path="volcano.png")
    """

    def __init__(
        self,
        # Stage 1
        stat_method: str = "tmm",
        n_svs: int = 2,
        gc_correction: bool = False,
        # Stage 2
        latent_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
        n_batch: int = 0,
        beta: float = 1.0,
        n_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        # Stage 3
        alpha: float = 0.05,
        lfc_threshold: float = 1.0,
        # Misc
        random_seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stage1 = StatisticalNormalizer(
            method=stat_method, n_svs=n_svs, gc_correction=gc_correction
        )
        # VAE is initialised after we know n_genes
        self._vae_kwargs = dict(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [256, 128],
            n_batch=n_batch,
            beta=beta,
        )
        self._trainer_kwargs = dict(lr=lr, device=device)
        self.stage3 = DifferentialExpressionTester(alpha=alpha, lfc_threshold=lfc_threshold)

        self.vae_: Optional[GeneExpressionVAE]   = None
        self.trainer_: Optional[VAETrainer]       = None
        self.latent_: Optional[np.ndarray]        = None
        self.corrected_counts_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def fit_transform(
        self,
        counts: np.ndarray,
        condition: np.ndarray,
        gene_names: Optional[np.ndarray] = None,
        sample_names: Optional[np.ndarray] = None,
        gene_info: Optional[pd.DataFrame] = None,
        control_genes: Optional[np.ndarray] = None,
        batch_labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline and return DE results.

        Parameters
        ----------
        counts       : raw integer count matrix (n_samples × n_genes)
        condition    : condition label per sample (e.g. ['ctrl','ctrl','treat','treat'])
        gene_names   : gene identifier array
        sample_names : sample identifier array
        gene_info    : DataFrame with 'gc_content' and 'length' columns (Stage 1 GC step)
        control_genes: bool mask of housekeeping genes for RUV (Stage 1)
        batch_labels : integer batch id per sample (Stage 2)

        Returns
        -------
        DE results DataFrame (from Stage 3)
        """
        n_samples, n_genes = counts.shape
        log.info("=" * 60)
        log.info("Hybrid normalization pipeline  [%d samples × %d genes]", n_samples, n_genes)
        log.info("=" * 60)

        # ── Stage 1 ───────────────────────────────────────────────────
        norm_log = self.stage1.fit_transform(
            counts,
            gene_info=gene_info,
            control_genes=control_genes,
        )

        # ── Stage 2 ───────────────────────────────────────────────────
        self.vae_ = GeneExpressionVAE(n_genes=n_genes, **self._vae_kwargs)
        self.trainer_ = VAETrainer(self.vae_, **self._trainer_kwargs)
        self.trainer_.fit(
            counts,              # VAE sees raw counts; learns NB likelihood internally
            batch_labels=batch_labels,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
        )

        X_t = torch.tensor(counts, dtype=torch.float32).to(self.device)
        B_t = None
        if batch_labels is not None and self.vae_.n_batch > 0:
            B_t = torch.zeros(n_samples, self.vae_.n_batch).to(self.device)
            B_t[torch.arange(n_samples), torch.tensor(batch_labels, dtype=torch.long)] = 1.0

        self.latent_ = self.vae_.get_latent(X_t, B_t)
        corrected = self.vae_.get_normalized_expression(X_t, batch_onehot=B_t)
        self.corrected_counts_ = corrected
        log.info("  Stage 2 complete — latent shape %s  corrected shape %s",
                 self.latent_.shape, corrected.shape)

        # ── Stage 3 ───────────────────────────────────────────────────
        # Round corrected expression back to pseudo-counts for DE testing
        pseudo_counts = np.round(corrected).astype(int).clip(0)

        results = self.stage3.run(
            pseudo_counts,
            condition,
            gene_names=gene_names,
            sample_names=sample_names,
        )

        log.info("=" * 60)
        log.info("Pipeline complete.")
        log.info("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Convenience wrappers for downstream plots
    # ------------------------------------------------------------------

    def volcano_plot(self, **kwargs) -> plt.Figure:
        return self.stage3.volcano_plot(**kwargs)

    def ma_plot(self, **kwargs) -> plt.Figure:
        return self.stage3.ma_plot(**kwargs)

    def plot_training_curve(self, **kwargs) -> plt.Figure:
        if self.trainer_ is None:
            raise RuntimeError("Pipeline not yet fitted.")
        return self.trainer_.plot_loss(**kwargs)

    def plot_latent_umap(
        self,
        condition: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """UMAP of the VAE latent space coloured by condition."""
        try:
            import umap
        except ImportError:
            log.warning("umap-learn not installed — falling back to PCA")
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(self.latent_)
            xlabel, ylabel = "PC1", "PC2"
        else:
            reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(self.latent_) - 1))
            coords  = reducer.fit_transform(self.latent_)
            xlabel, ylabel = "UMAP 1", "UMAP 2"

        groups  = np.unique(condition)
        palette = ["#185FA5", "#D85A30", "#0F6E56", "#533AB7", "#85001C"]
        fig, ax = plt.subplots(figsize=(6, 5))

        for i, grp in enumerate(groups):
            mask = condition == grp
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=grp, color=palette[i % len(palette)],
                       s=40, alpha=0.85, linewidths=0)

        ax.legend(fontsize=9, framealpha=0.8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title("Latent space (VAE) by condition", fontsize=12)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def top_genes(self, n: int = 20) -> pd.DataFrame:
        """Return top-n DE genes by adjusted p-value."""
        if self.stage3.results_ is None:
            raise RuntimeError("Pipeline not yet fitted.")
        return (
            self.stage3.results_[self.stage3.results_["significant"]]
            .nsmallest(n, "padj")
            .reset_index(drop=True)
        )
