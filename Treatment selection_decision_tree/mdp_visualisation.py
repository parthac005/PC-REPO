"""
mdp_visualisation.py
====================
All plots for the Healthcare Treatment MDP.

Functions
---------
plot_transition_heatmap()   — T[s, a, s'] heatmap for each action
plot_reward_matrix()        — heatmap of R[s, a]
plot_value_function()       — bar chart of V(s) for each solver
plot_policy()               — heatmap of recommended actions
plot_convergence()          — delta-per-iteration curves
plot_patient_trajectory()   — single patient journey
plot_cohort_outcomes()      — final-state distribution across cohorts
plot_policy_comparison()    — grouped bar comparing policies on KPIs
plot_q_learning_curve()     — smoothed episode return over training
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import Dict, List, Optional

from mdp_environment import (
    TreatmentMDP, STATE_NAMES, ACTION_NAMES, N_STATES, N_ACTIONS
)

# ── Colour palettes ───────────────────────────────────────────────────────────
PALETTE_ACTIONS = ["#B4B2A9","#1D9E75","#185FA5","#534AB7","#D85A30"]
PALETTE_STATES  = ["#0F6E56","#1D9E75","#97C459","#EF9F27","#D85A30","#A32D2D","#444441"]
PALETTE_METHODS = ["#185FA5","#0F6E56","#D85A30","#534AB7"]

ACTION_SHORT = ["Watch", "Lifestyle", "Mono", "Combo", "Intensive"]


# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Optional[str]):
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Transition probability heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_heatmap(
    mdp: TreatmentMDP,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """One heatmap per action: rows=current state, cols=next state."""
    fig, axes = plt.subplots(1, N_ACTIONS, figsize=(20, 4), sharey=True)
    short_labels = [s[:5] for s in STATE_NAMES]

    for a, ax in enumerate(axes):
        data = mdp.T[:, a, :]
        im   = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(N_STATES)); ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_STATES)); ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(ACTION_SHORT[a], fontsize=10, color=PALETTE_ACTIONS[a])
        ax.set_xlabel("Next state", fontsize=8)
        if a == 0: ax.set_ylabel("Current state", fontsize=8)

        for i in range(N_STATES):
            for j in range(N_STATES):
                v = data[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")

    fig.suptitle("Transition Probabilities  T[s, a, s']  — one subplot per action",
                 fontsize=12, y=1.02)
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Probability")
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reward matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_matrix(
    mdp: TreatmentMDP,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mdp.R, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(N_ACTIONS)); ax.set_xticklabels(ACTION_SHORT, fontsize=9)
    ax.set_yticks(range(N_STATES)); ax.set_yticklabels(STATE_NAMES, fontsize=9)
    ax.set_xlabel("Action", fontsize=10); ax.set_ylabel("State", fontsize=10)
    ax.set_title("Expected Immediate Reward  R[state, action]", fontsize=12)

    for i in range(N_STATES):
        for j in range(N_ACTIONS):
            ax.text(j, i, f"{mdp.R[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(mdp.R[i,j]) > 6 else "black")

    plt.colorbar(im, ax=ax, label="Reward")
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Value function comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_function(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    x   = np.arange(N_STATES)
    w   = 0.25
    n   = len(results)
    off = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    for i, (name, res) in enumerate(results.items()):
        bars = ax.bar(x + off[i], res["V"], w,
                      label=name, color=PALETTE_METHODS[i % len(PALETTE_METHODS)],
                      alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("State Value  V(s)", fontsize=10)
    ax.set_title("Value Function  V(s)  across Solvers", fontsize=12)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.yaxis.grid(True, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Optimal policy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_policy(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4), sharey=True)
    if n_methods == 1: axes = [axes]

    cmap = matplotlib.colors.ListedColormap(PALETTE_ACTIONS)
    bounds = np.arange(-0.5, N_ACTIONS + 0.5)
    norm   = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    for ax, (name, res) in zip(axes, results.items()):
        pol   = res["policy"].reshape(-1, 1)
        im    = ax.imshow(pol, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(range(N_STATES)); ax.set_yticklabels(STATE_NAMES, fontsize=9)
        ax.set_xticks([]); ax.set_title(name, fontsize=10)
        for s in range(N_STATES):
            ax.text(0, s, ACTION_SHORT[res["policy"][s]],
                    ha="center", va="center", fontsize=8, color="white", fontweight="500")

    patches = [mpatches.Patch(color=PALETTE_ACTIONS[a], label=ACTION_NAMES[a])
               for a in range(N_ACTIONS)]
    fig.legend(handles=patches, loc="lower center", ncol=N_ACTIONS,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Optimal Treatment Policy  π(s)", fontsize=12)
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Value Iteration + Policy Iteration delta
    for i, name in enumerate(["Value Iteration", "Policy Iteration"]):
        if name in results:
            h = results[name]["history"]
            axes[0].plot(h, label=name, color=PALETTE_METHODS[i], lw=1.6)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Max |ΔV|  (log scale)")
    axes[0].set_title("DP Solver Convergence"); axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, lw=0.4, alpha=0.5); axes[0].set_axisbelow(True)

    # Right: Q-Learning episode returns (smoothed)
    if "Q-Learning" in results:
        h      = np.array(results["Q-Learning"]["history"])
        window = max(1, len(h) // 200)
        kernel = np.ones(window) / window
        smooth = np.convolve(h, kernel, mode="valid")
        x      = np.arange(len(smooth))
        axes[1].plot(x, smooth, color=PALETTE_METHODS[2], lw=1.2, label="Smoothed return")
        axes[1].fill_between(x,
            np.convolve(h, kernel, "valid") - np.convolve(np.abs(h - h.mean()), kernel, "valid") * 0.5,
            np.convolve(h, kernel, "valid") + np.convolve(np.abs(h - h.mean()), kernel, "valid") * 0.5,
            color=PALETTE_METHODS[2], alpha=0.15)
        axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Episode Return")
        axes[1].set_title("Q-Learning Training Curve"); axes[1].legend(fontsize=9)
        axes[1].yaxis.grid(True, lw=0.4, alpha=0.5); axes[1].set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Single patient trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_patient_trajectory(
    trajectory: pd.DataFrame,
    title: str = "Patient Treatment Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    steps  = trajectory["step"]
    states = trajectory["state"]

    # ── Health state over time ──────────────────────────────────────
    ax0 = axes[0]
    for s in range(7):
        mask = states == s
        if mask.any():
            ax0.scatter(steps[mask], states[mask], color=PALETTE_STATES[s],
                        s=80, zorder=4, label=STATE_NAMES[s])
    ax0.plot(steps, states, color="#B4B2A9", lw=1.0, zorder=2, ls="--")
    ax0.set_yticks(range(7)); ax0.set_yticklabels(STATE_NAMES, fontsize=8)
    ax0.invert_yaxis()
    ax0.set_ylabel("Health State", fontsize=10)
    ax0.set_title(title, fontsize=12)
    ax0.yaxis.grid(True, lw=0.4, alpha=0.4); ax0.set_axisbelow(True)
    ax0.legend(fontsize=7, loc="upper right", ncol=4)

    # ── Action taken ─────────────────────────────────────────────────
    ax1 = axes[1]
    actions_int = trajectory["action"].replace(-1, np.nan)
    for a in range(N_ACTIONS):
        mask = trajectory["action"] == a
        if mask.any():
            ax1.scatter(steps[mask], [a]*mask.sum(), color=PALETTE_ACTIONS[a],
                        s=60, zorder=3, label=ACTION_SHORT[a])
    ax1.set_yticks(range(N_ACTIONS))
    ax1.set_yticklabels(ACTION_SHORT, fontsize=8)
    ax1.set_ylabel("Action", fontsize=10)
    ax1.yaxis.grid(True, lw=0.4, alpha=0.4); ax1.set_axisbelow(True)
    ax1.legend(fontsize=7, loc="upper right", ncol=5)

    # ── Cumulative reward ─────────────────────────────────────────────
    ax2 = axes[2]
    ax2.plot(steps, trajectory["cumulative_reward"],
             color="#185FA5", lw=1.8)
    ax2.fill_between(steps, 0, trajectory["cumulative_reward"],
                     color="#185FA5", alpha=0.12)
    ax2.axhline(0, color="#888", lw=0.8, ls="--")
    ax2.set_xlabel("Visit (month)", fontsize=10)
    ax2.set_ylabel("Cumulative Reward", fontsize=10)
    ax2.yaxis.grid(True, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cohort outcome distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_cohort_outcomes(
    cohort_results: Dict,
    policy_name: str = "Optimal",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fs    = cohort_results["final_states"]
    rets  = cohort_results["total_returns"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: final state bar chart
    ax = axes[0]
    counts = [(STATE_NAMES[s], int((fs == s).sum())) for s in range(7)]
    names, vals = zip(*counts)
    bars = ax.bar(names, vals, color=PALETTE_STATES, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Final Health State"); ax.set_ylabel("Number of Patients")
    ax.set_title(f"Final State Distribution — {policy_name} Policy")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.grid(True, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    # Right: return distribution
    ax2 = axes[1]
    ax2.hist(rets, bins=30, color="#185FA5", alpha=0.75, edgecolor="white", lw=0.4)
    ax2.axvline(rets.mean(), color="#D85A30", lw=1.8, label=f"Mean={rets.mean():.1f}")
    ax2.axvline(np.median(rets), color="#0F6E56", lw=1.8, ls="--", label=f"Median={np.median(rets):.1f}")
    ax2.set_xlabel("Total Discounted Return"); ax2.set_ylabel("Frequency")
    ax2.set_title(f"Return Distribution — {policy_name} Policy")
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Policy comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_policy_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    metrics = ["Mean Return", "% Improved(0-2)", "% Remission", "% Terminal"]
    n       = len(comparison_df)
    x       = np.arange(len(metrics))
    w       = 0.8 / n
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (policy_name, row) in enumerate(comparison_df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=policy_name, color=PALETTE_METHODS[i % len(PALETTE_METHODS)],
                      alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score / Percentage"); ax.set_title("Policy Comparison across KPIs")
    ax.yaxis.grid(True, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Q-value heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_q_values(
    Q: np.ndarray,
    title: str = "Q-Values  Q(state, action)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Q, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(N_ACTIONS)); ax.set_xticklabels(ACTION_SHORT, fontsize=9)
    ax.set_yticks(range(N_STATES));  ax.set_yticklabels(STATE_NAMES, fontsize=9)
    ax.set_xlabel("Action"); ax.set_ylabel("State")
    ax.set_title(title, fontsize=12)

    for i in range(N_STATES):
        for j in range(N_ACTIONS):
            best = Q[i].argmax() == j
            ax.text(j, i, f"{Q[i,j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(Q[i,j]) > abs(Q[i]).max()*0.6 else "black",
                    fontweight="bold" if best else "normal")

    plt.colorbar(im, ax=ax, label="Q-value")
    fig.tight_layout()
    return _save(fig, save_path)
