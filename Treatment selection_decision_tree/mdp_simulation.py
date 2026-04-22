"""
mdp_simulation.py
=================
Patient trajectory simulation and policy evaluation utilities.

Functions
---------
simulate_patient()     — run one patient under a given policy
simulate_cohort()      — simulate N patients, collect statistics
evaluate_policy()      — compute expected return via Monte Carlo
compare_policies()     — side-by-side comparison of multiple policies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from mdp_environment import (
    TreatmentMDP, STATE_NAMES, ACTION_NAMES,
    N_STATES, TERMINAL_STATE
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-patient simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_patient(
    mdp          : TreatmentMDP,
    policy       : np.ndarray,
    initial_state: int,
    n_steps      : int = 24,
    seed         : Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate one patient's treatment trajectory under a policy.

    Parameters
    ----------
    mdp           : TreatmentMDP
    policy        : array (S,) mapping state → action
    initial_state : starting health state (0–5)
    n_steps       : number of monthly visits to simulate
    seed          : random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        step, state, state_name, action, action_name, reward, cumulative_reward
    """
    if seed is not None:
        mdp._rng = np.random.default_rng(seed)

    state = mdp.reset(initial_state)
    rows  = []
    cum_r = 0.0

    for t in range(n_steps):
        action              = int(policy[state])
        next_state, r, done = mdp.step(action)
        cum_r              += r

        rows.append({
            "step"              : t + 1,
            "state"             : state,
            "state_name"        : STATE_NAMES[state],
            "action"            : action,
            "action_name"       : ACTION_NAMES[action],
            "reward"            : round(r, 3),
            "cumulative_reward" : round(cum_r, 3),
        })
        state = next_state
        if done:
            rows.append({
                "step"              : t + 2,
                "state"             : TERMINAL_STATE,
                "state_name"        : STATE_NAMES[TERMINAL_STATE],
                "action"            : -1,
                "action_name"       : "—",
                "reward"            : 0.0,
                "cumulative_reward" : round(cum_r, 3),
            })
            break

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Cohort simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_cohort(
    mdp          : TreatmentMDP,
    policy       : np.ndarray,
    n_patients   : int = 500,
    n_steps      : int = 24,
    initial_dist : Optional[np.ndarray] = None,
    seed         : int = 0,
) -> Dict:
    """
    Simulate a cohort of patients under a policy.

    Parameters
    ----------
    initial_dist : probability over starting states (default: uniform over 1–5)

    Returns
    -------
    dict with keys:
        trajectories : list of DataFrames
        summary      : aggregate statistics DataFrame
        final_states : ndarray of final states
        total_returns: ndarray of per-patient total returns
    """
    rng = np.random.default_rng(seed)

    if initial_dist is None:
        # Uniform over Mild, Moderate, Severe (states 2–4) — realistic clinic
        initial_dist = np.array([0.0, 0.10, 0.35, 0.35, 0.20, 0.0, 0.0])
    initial_dist /= initial_dist.sum()

    initial_states = rng.choice(N_STATES, size=n_patients, p=initial_dist)

    trajectories   = []
    total_returns  = []
    final_states   = []

    for i in range(n_patients):
        traj = simulate_patient(
            mdp, policy,
            initial_state=int(initial_states[i]),
            n_steps=n_steps,
            seed=int(rng.integers(1_000_000)),
        )
        trajectories.append(traj)
        total_returns.append(traj["cumulative_reward"].iloc[-1])
        final_states.append(traj["state"].iloc[-1])

    total_returns = np.array(total_returns)
    final_states  = np.array(final_states)

    # Build summary by final state
    state_counts = {STATE_NAMES[s]: int((final_states == s).sum())
                    for s in range(N_STATES)}

    summary = pd.DataFrame({
        "Final State"        : list(state_counts.keys()),
        "Count"              : list(state_counts.values()),
        "Percentage"         : [round(v / n_patients * 100, 1)
                                 for v in state_counts.values()],
    })

    return {
        "trajectories"  : trajectories,
        "summary"       : summary,
        "final_states"  : final_states,
        "total_returns" : total_returns,
        "mean_return"   : float(total_returns.mean()),
        "std_return"    : float(total_returns.std()),
        "n_patients"    : n_patients,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy evaluation (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    mdp        : TreatmentMDP,
    policy     : np.ndarray,
    n_episodes : int = 2_000,
    n_steps    : int = 36,
    seed       : int = 0,
) -> Dict:
    """
    Monte Carlo policy evaluation: estimate V^π(s) for each starting state.

    Returns
    -------
    dict:
        V_mc     : array (S,)  — MC estimated value per state
        returns  : dict s → list of episode returns
        ci_lower : array (S,)  — 95 % CI lower bound
        ci_upper : array (S,)  — 95 % CI upper bound
    """
    rng         = np.random.default_rng(seed)
    returns_per : Dict[int, List[float]] = {s: [] for s in range(N_STATES - 1)}

    for _ in range(n_episodes):
        start = int(rng.integers(0, N_STATES - 1))
        traj  = simulate_patient(
            mdp, policy, initial_state=start,
            n_steps=n_steps, seed=int(rng.integers(1_000_000)),
        )
        g = sum(
            row["reward"] * (mdp.gamma ** (row["step"] - 1))
            for _, row in traj.iterrows()
        )
        returns_per[start].append(g)

    V_mc     = np.zeros(N_STATES)
    ci_lower = np.zeros(N_STATES)
    ci_upper = np.zeros(N_STATES)

    for s in range(N_STATES - 1):
        r = np.array(returns_per[s])
        if len(r) > 1:
            V_mc[s]     = r.mean()
            se          = r.std() / np.sqrt(len(r))
            ci_lower[s] = V_mc[s] - 1.96 * se
            ci_upper[s] = V_mc[s] + 1.96 * se

    return {
        "V_mc"    : V_mc,
        "returns" : returns_per,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_policies(
    mdp        : TreatmentMDP,
    policies   : Dict[str, np.ndarray],
    n_patients : int = 300,
    n_steps    : int = 24,
    seed       : int = 0,
) -> pd.DataFrame:
    """
    Compare multiple policies on cohort-level outcomes.

    Parameters
    ----------
    policies : dict mapping policy name → policy array

    Returns
    -------
    DataFrame: one row per (policy, metric)
    """
    rows = []
    for name, pol in policies.items():
        cohort = simulate_cohort(mdp, pol, n_patients=n_patients,
                                  n_steps=n_steps, seed=seed)
        fs     = cohort["final_states"]
        rows.append({
            "Policy"           : name,
            "Mean Return"      : round(cohort["mean_return"], 3),
            "Std Return"       : round(cohort["std_return"], 3),
            "% Remission"      : round((fs == 0).mean() * 100, 1),
            "% Controlled"     : round((fs == 1).mean() * 100, 1),
            "% Improved(0-2)"  : round((fs <= 2).mean() * 100, 1),
            "% Terminal"       : round((fs == 6).mean() * 100, 1),
        })

    return pd.DataFrame(rows).set_index("Policy")


# ─────────────────────────────────────────────────────────────────────────────
# Named baseline policies for benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def make_baseline_policies() -> Dict[str, np.ndarray]:
    """
    Return a set of clinically-motivated hand-crafted policies.

    Conservative   — always Lifestyle Modification
    Standard       — Monotherapy for all states
    Aggressive     — always Combination Therapy
    Step-up        — escalate treatment with severity (clinical guideline)
    """
    import numpy as np

    conservative = np.full(N_STATES, 1)    # Lifestyle everywhere
    standard     = np.full(N_STATES, 2)    # Monotherapy everywhere
    aggressive   = np.full(N_STATES, 3)    # Combination everywhere

    # Step-up: Remission→Watch, Controlled→Lifestyle, Mild→Mono,
    #          Moderate→Combo, Severe→Intensive, Critical→Intensive
    step_up = np.array([0, 1, 2, 3, 4, 4, 0])

    return {
        "Conservative (Lifestyle)"  : conservative,
        "Standard (Monotherapy)"    : standard,
        "Aggressive (Combination)"  : aggressive,
        "Step-Up (Guideline)"       : step_up,
    }
