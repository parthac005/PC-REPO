"""
demo.py — End-to-end demo with synthetic RNA-seq data
======================================================
Generates a simulated bulk RNA-seq dataset (2 conditions, 2 batches),
runs the full hybrid normalization pipeline, and saves key figures.

Run:
    python demo.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import logging
from pathlib import Path

from pipeline import HybridNormalizationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Simulate RNA-seq data
# ---------------------------------------------------------------------------

def simulate_rnaseq(
    n_samples: int = 60,
    n_genes: int = 2000,
    n_de_genes: int = 200,
    lfc: float = 2.0,
    n_batches: int = 2,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic negative-binomial counts with:
      - 2 balanced conditions (ctrl / treat)
      - batch effect (mean shift proportional to library size)
      - n_de_genes genes truly differentially expressed
    """
    rng = np.random.default_rng(seed)

    condition  = np.array(["ctrl"] * (n_samples // 2) + ["treat"] * (n_samples // 2))
    batch      = np.tile(np.arange(n_batches), n_samples // n_batches + 1)[:n_samples]
    lib_sizes  = rng.integers(5_000_000, 20_000_000, size=n_samples)

    # Base mean per gene (log-normal)
    base_mean  = rng.lognormal(mean=3.0, sigma=1.5, size=n_genes)
    dispersion = rng.uniform(0.05, 0.5, size=n_genes)

    # DE gene indices
    de_up   = rng.choice(n_genes, size=n_de_genes // 2, replace=False)
    de_down = rng.choice(list(set(range(n_genes)) - set(de_up)),
                         size=n_de_genes // 2, replace=False)

    counts = np.zeros((n_samples, n_genes), dtype=int)

    for i in range(n_samples):
        mu = base_mean.copy() * lib_sizes[i] / 1e7  # scale to library

        # Apply condition effect
        if condition[i] == "treat":
            mu[de_up]   *= (2 ** lfc)
            mu[de_down] /= (2 ** lfc)

        # Apply batch effect (multiplicative)
        batch_factor = 1.5 if batch[i] == 0 else 0.9
        mu *= batch_factor

        # Sample NB counts
        p    = dispersion / (mu + dispersion)
        r    = 1.0 / dispersion
        counts[i] = np.random.negative_binomial(r, p)

    de_mask = np.zeros(n_genes, dtype=bool)
    de_mask[de_up] = True
    de_mask[de_down] = True

    gene_names   = np.array([f"GENE_{i:05d}" for i in range(n_genes)])
    sample_names = np.array([f"S{i:03d}_{condition[i]}_B{batch[i]}" for i in range(n_samples)])

    return dict(
        counts=counts,
        condition=condition,
        batch=batch,
        gene_names=gene_names,
        sample_names=sample_names,
        de_mask=de_mask,
        de_up=de_up,
        de_down=de_down,
    )


# ---------------------------------------------------------------------------
# 2. Run pipeline
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    log.info("Simulating RNA-seq dataset …")
    data = simulate_rnaseq(n_samples=60, n_genes=2000, n_de_genes=200, lfc=2.0)

    pipe = HybridNormalizationPipeline(
        # Stage 1
        stat_method="tmm",
        n_svs=2,
        gc_correction=False,
        # Stage 2
        latent_dim=20,
        hidden_dims=[256, 128],
        n_batch=2,        # 2 batches
        beta=1.0,
        n_epochs=100,     # increase to 300+ for real data
        batch_size=16,    # small because n_samples=60
        lr=1e-3,
        # Stage 3
        alpha=0.05,
        lfc_threshold=1.0,
        random_seed=42,
    )

    results = pipe.fit_transform(
        counts       = data["counts"],
        condition    = data["condition"],
        gene_names   = data["gene_names"],
        sample_names = data["sample_names"],
        batch_labels = data["batch"],
    )

    # ── Save results ──────────────────────────────────────────────────
    results_path = out_dir / "de_results.csv"
    results.to_csv(results_path, index=False)
    log.info("DE results saved → %s", results_path)

    pipe.volcano_plot(
        title="Hybrid pipeline — Volcano plot",
        save_path=str(out_dir / "volcano.png"),
    )
    pipe.ma_plot(save_path=str(out_dir / "ma_plot.png"))
    pipe.plot_training_curve(save_path=str(out_dir / "training_curve.png"))
    pipe.plot_latent_umap(
        condition=data["condition"],
        save_path=str(out_dir / "latent_umap.png"),
    )

    # ── Performance metrics ───────────────────────────────────────────
    true_de = set(data["gene_names"][data["de_mask"]])
    predicted_de = set(results[results["significant"]]["gene"].values)

    tp = len(true_de & predicted_de)
    fp = len(predicted_de - true_de)
    fn = len(true_de - predicted_de)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    log.info("──────────────────────────────────")
    log.info("Performance  (simulated ground truth)")
    log.info("  True DE genes       : %d", len(true_de))
    log.info("  Predicted DE genes  : %d", len(predicted_de))
    log.info("  True positives      : %d", tp)
    log.info("  Precision           : %.3f", precision)
    log.info("  Recall              : %.3f", recall)
    log.info("  F1 score            : %.3f", f1)
    log.info("──────────────────────────────────")

    log.info("Top 10 DE genes:")
    print(pipe.top_genes(10).to_string(index=False))

    log.info("All outputs saved in  %s/", out_dir)


if __name__ == "__main__":
    main()
