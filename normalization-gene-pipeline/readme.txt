# Hybrid Normalization Pipeline for Bulk RNA-seq DE Analysis

A three-stage hybrid normalization framework combining classical statistical
methods with a Variational Autoencoder for improved differential expression
analysis of bulk RNA-seq data.

## Architecture

```
Raw Counts
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 1 — Statistical Normalization              │
│  TMM / DESeq2 size factors                       │
│  → GC-content / length correction (optional)     │
│  → RUVg surrogate variable estimation            │
└─────────────────────────────────────────────────┘
    │  normalized log-counts
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 2 — VAE Deep Learning Refinement           │
│  Encoder  : FC → BN → ReLU → μ, log σ²          │
│  Decoder  : FC → BN → ReLU → NB(rate, r)        │
│  Loss     : ELBO = NB-LL − β·KL                 │
│  Output   : batch-corrected expression matrix    │
└─────────────────────────────────────────────────┘
    │  corrected counts
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 3 — Differential Expression Testing        │
│  pydeseq2 (preferred) or Welch t-test fallback   │
│  FDR correction (Benjamini–Hochberg)             │
│  Output   : ranked gene list + plots             │
└─────────────────────────────────────────────────┘
```

## File Structure

```
hybrid_normalization/
├── hybrid_normalizer.py   # Stage 1 — StatisticalNormalizer
├── vae_model.py           # Stage 2 — GeneExpressionVAE + VAETrainer
├── pipeline.py            # Stage 3 + HybridNormalizationPipeline
├── demo.py                # End-to-end demo with simulated data
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Optional (recommended for Stage 3)
```bash
pip install pydeseq2          # Python port of DESeq2
pip install umap-learn        # UMAP visualisation of latent space
```

## Quick Start

```python
from pipeline import HybridNormalizationPipeline

pipe = HybridNormalizationPipeline(
    stat_method  = "tmm",    # 'tmm' or 'deseq2'
    n_svs        = 2,        # surrogate variables for batch
    latent_dim   = 20,       # VAE latent dimension
    n_batch      = 2,        # number of technical batches
    n_epochs     = 300,
    alpha        = 0.05,
    lfc_threshold= 1.0,
)

results = pipe.fit_transform(
    counts       = counts_matrix,       # np.ndarray (n_samples × n_genes)
    condition    = condition_array,     # e.g. ['ctrl','ctrl','treat','treat']
    gene_names   = gene_name_array,
    sample_names = sample_name_array,
    batch_labels = batch_label_array,   # integer batch ids, 0-based
)

# Outputs
pipe.volcano_plot(save_path="volcano.png")
pipe.ma_plot(save_path="ma_plot.png")
pipe.plot_latent_umap(condition=condition_array, save_path="umap.png")
pipe.plot_training_curve(save_path="training.png")
print(pipe.top_genes(20))
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stat_method` | `"tmm"` | Size-factor method: `"tmm"` or `"deseq2"` |
| `n_svs` | `2` | Number of surrogate variables (RUVg step) |
| `gc_correction` | `False` | Enable GC/length bias correction |
| `latent_dim` | `20` | VAE latent space dimension |
| `hidden_dims` | `[256,128]` | Encoder/decoder hidden layer widths |
| `n_batch` | `0` | Number of batch categories (0 = no batch covariate) |
| `beta` | `1.0` | KL weight in ELBO (β-VAE; increase to strengthen disentanglement) |
| `n_epochs` | `200` | Maximum VAE training epochs |
| `batch_size` | `128` | Mini-batch size for VAE |
| `lr` | `1e-3` | Adam learning rate |
| `alpha` | `0.05` | FDR threshold for DE significance |
| `lfc_threshold` | `1.0` | |log2FC| threshold for biological significance |

## Guidance for Real Data

- **n_svs**: Start with 2–5. Check PCA before and after Stage 1.
- **latent_dim**: 10–30 for bulk RNA-seq. Larger if you have >200 samples.
- **hidden_dims**: `[256, 128]` works for most datasets. For large datasets (>1000 samples) try `[512, 256, 128]`.
- **n_batch**: Set to the actual number of sequencing batches. Use `0` if batches are unknown.
- **beta**: Values >1 encourage more disentangled latent representations.
- **n_epochs**: 200–500 for real data. Early stopping will trigger before if validation loss plateaus.
- Always run DE testing with and without Stage 2 and compare volcano plots to validate the DL step is helping.
