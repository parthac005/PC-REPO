"""
Stage 2 — Variational Autoencoder for Gene Expression Refinement
=================================================================
Architecture:
  Encoder  : FC → BN → ReLU (×2 layers) → μ, log σ²
  Decoder  : FC → BN → ReLU (×2 layers) → softmax (proportions) × library_size
  Loss     : ELBO = reconstruction NB-LL + β * KL divergence
  Extras   : Batch covariate injection at encoder and decoder input
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Tuple
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _fc_block(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: List[int],
        n_batch: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_batch = n_batch
        input_dim = n_genes + n_batch

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(_fc_block(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers)

        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_batch > 0 and batch_onehot is not None:
            x = torch.cat([x, batch_onehot], dim=-1)
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dims: List[int],
        n_batch: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_batch  = n_batch
        self.n_genes  = n_genes
        input_dim     = latent_dim + n_batch

        layers = []
        prev = input_dim
        for h in reversed(hidden_dims):
            layers.append(_fc_block(prev, h, dropout))
            prev = h
        self.net = nn.Sequential(*layers)

        # Output: gene proportions (softmax) + per-gene dispersion (softplus)
        self.fc_px_scale = nn.Linear(prev, n_genes)   # log proportions
        self.fc_px_r     = nn.Linear(prev, n_genes)   # log dispersion

    def forward(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_batch > 0 and batch_onehot is not None:
            z = torch.cat([z, batch_onehot], dim=-1)
        h = self.net(z)
        px_scale = F.softmax(self.fc_px_scale(h), dim=-1)   # proportions
        px_r     = torch.exp(self.fc_px_r(h))                # dispersion > 0
        # Expected counts = library_size × proportion
        px_rate  = library.unsqueeze(-1) * px_scale
        return px_rate, px_r


# ---------------------------------------------------------------------------
# Negative-Binomial log-likelihood
# ---------------------------------------------------------------------------

def _nb_loglik(
    x: torch.Tensor,
    mu: torch.Tensor,
    r: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Log-likelihood of NB(r, p) parametrized by mean μ and dispersion r.
    ℓ = Γ(x+r)/Γ(r)/x! + x·log(μ/(μ+r)) + r·log(r/(r+μ))
    """
    log_theta_mu_eps = torch.log(r + mu + eps)
    res = (
        torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1)
        + r * (torch.log(r + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
    )
    return res


# ---------------------------------------------------------------------------
# Full VAE model
# ---------------------------------------------------------------------------

class GeneExpressionVAE(nn.Module):
    """
    Variational Autoencoder for bulk RNA-seq normalization / batch correction.

    Parameters
    ----------
    n_genes      : number of input genes
    latent_dim   : dimension of the latent space  (default 20)
    hidden_dims  : list of hidden layer widths     (default [256, 128])
    n_batch      : number of batch categories (0 = no batch covariate)
    beta         : KL weight in ELBO (beta-VAE; 1.0 = standard VAE)
    dropout      : dropout rate inside FC blocks
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
        n_batch: int = 0,
        beta: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.n_genes    = n_genes
        self.latent_dim = latent_dim
        self.n_batch    = n_batch
        self.beta       = beta

        self.encoder = Encoder(n_genes, latent_dim, hidden_dims, n_batch, dropout)
        self.decoder = Decoder(n_genes, latent_dim, hidden_dims, n_batch, dropout)

    # ------------------------------------------------------------------

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at inference

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        library: torch.Tensor,
        batch_onehot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Log-transform input before encoding (standard practice)
        x_log = torch.log1p(x)

        mu, logvar = self.encoder(x_log, batch_onehot)
        z = self.reparameterize(mu, logvar)
        px_rate, px_r = self.decoder(z, library, batch_onehot)

        return dict(mu=mu, logvar=logvar, z=z, px_rate=px_rate, px_r=px_r)

    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction: mean NB log-likelihood across all cells and genes
        reconst = _nb_loglik(x, out["px_rate"], out["px_r"]).mean()

        # KL divergence: -½ Σ(1 + log σ² - μ² - σ²)
        kl = -0.5 * (1 + out["logvar"] - out["mu"] ** 2 - out["logvar"].exp()).sum(-1).mean()

        elbo = reconst - self.beta * kl
        return -elbo, reconst, kl   # minimize negative ELBO

    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(
        self,
        x: torch.Tensor,
        batch_onehot: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Return the posterior mean (μ) as the latent embedding."""
        self.eval()
        x_log = torch.log1p(x)
        mu, _ = self.encoder(x_log, batch_onehot)
        return mu.cpu().numpy()

    @torch.no_grad()
    def get_normalized_expression(
        self,
        x: torch.Tensor,
        library: Optional[torch.Tensor] = None,
        batch_onehot: Optional[torch.Tensor] = None,
        normalize_library: bool = True,
    ) -> np.ndarray:
        """
        Return decoder-reconstructed expression, optionally scaled to a
        fixed library size (10 000) for comparability across samples.
        """
        self.eval()
        if library is None:
            library = x.sum(dim=-1)
        out = self.forward(x, library, batch_onehot)
        if normalize_library:
            # Use fixed library = 1e4 to decouple from sequencing depth
            fixed_lib = torch.full((x.shape[0],), 1e4, device=x.device)
            _, px_r = self.decoder(out["z"], fixed_lib, batch_onehot)
            px_scale = F.softmax(
                self.decoder.fc_px_scale(
                    self.decoder.net(
                        torch.cat([out["z"], batch_onehot], -1) if self.n_batch > 0 and batch_onehot is not None
                        else out["z"]
                    )
                ), dim=-1
            )
            expr = (fixed_lib.unsqueeze(-1) * px_scale).cpu().numpy()
        else:
            expr = out["px_rate"].cpu().numpy()
        return expr


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VAETrainer:
    """
    Trains GeneExpressionVAE with early stopping and learning-rate scheduling.

    Parameters
    ----------
    model        : GeneExpressionVAE instance
    lr           : initial learning rate
    weight_decay : L2 regularisation
    patience     : early-stopping patience (epochs)
    """

    def __init__(
        self,
        model: GeneExpressionVAE,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model    = model
        self.device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience // 2, factor=0.5, verbose=False
        )
        self.patience  = patience
        self.history   = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------

    def fit(
        self,
        counts: np.ndarray,
        batch_labels: Optional[np.ndarray] = None,
        n_epochs: int = 200,
        batch_size: int = 128,
        val_split: float = 0.1,
    ) -> "VAETrainer":
        """
        Train the VAE.

        Parameters
        ----------
        counts       : raw integer count array (n_samples, n_genes)
        batch_labels : integer batch label per sample (0-based); None = ignore
        n_epochs     : maximum training epochs
        batch_size   : mini-batch size
        val_split    : fraction of data held out for validation / early stopping
        """
        X = torch.tensor(counts, dtype=torch.float32)
        lib = X.sum(dim=-1)

        if batch_labels is not None and self.model.n_batch > 0:
            B = torch.zeros(len(counts), self.model.n_batch)
            B[torch.arange(len(counts)), torch.tensor(batch_labels, dtype=torch.long)] = 1.0
        else:
            B = None

        # Train / val split
        n_val = max(1, int(len(X) * val_split))
        idx   = torch.randperm(len(X))
        val_idx, trn_idx = idx[:n_val], idx[n_val:]

        def _loader(idx_):
            ds = TensorDataset(
                X[idx_], lib[idx_],
                *([] if B is None else [B[idx_]])
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        trn_loader = _loader(trn_idx)
        val_loader = _loader(val_idx)

        best_val, patience_ctr, best_state = float("inf"), 0, None

        log.info("Stage 2 — VAE training  [device=%s  epochs=%d  batch=%d  genes=%d  latent=%d]",
                 self.device, n_epochs, batch_size, self.model.n_genes, self.model.latent_dim)

        for epoch in range(1, n_epochs + 1):
            trn_loss = self._epoch(trn_loader, train=True)
            val_loss = self._epoch(val_loader, train=False)

            self.history["train_loss"].append(trn_loss)
            self.history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            if val_loss < best_val - 1e-4:
                best_val    = val_loss
                patience_ctr = 0
                best_state  = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1

            if epoch % 20 == 0 or epoch == 1:
                log.info("  epoch %3d/%d  train=%.4f  val=%.4f  patience=%d/%d",
                         epoch, n_epochs, trn_loss, val_loss, patience_ctr, self.patience)

            if patience_ctr >= self.patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

        if best_state:
            self.model.load_state_dict(best_state)
            log.info("  Restored best model (val_loss=%.4f)", best_val)

        return self

    # ------------------------------------------------------------------

    def _epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        ctx = torch.enable_grad if train else torch.no_grad

        with ctx():
            for batch in loader:
                x   = batch[0].to(self.device)
                lib = batch[1].to(self.device)
                bon = batch[2].to(self.device) if len(batch) == 3 else None

                out  = self.model(x, lib, bon)
                loss, _, _ = self.model.loss(x, out)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total += loss.item() * len(x)

        return total / len(loader.dataset)

    # ------------------------------------------------------------------

    def plot_loss(self, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(self.history["train_loss"], label="train", lw=1.5)
        ax.plot(self.history["val_loss"],   label="val",   lw=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ELBO loss")
        ax.set_title("VAE training curve")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
