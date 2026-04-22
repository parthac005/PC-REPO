"""
mdp_solvers.py
==============
Three algorithms for solving the Healthcare Treatment MDP:

  1. Value Iteration        — exact DP solution (model-based)
  2. Policy Iteration       — exact DP solution (faster convergence)
  3. Q-Learning             — model-free RL (learns from simulated episodes)

All solvers return a consistent result dict with keys:
  policy  : ndarray (S,)   — optimal action per state
  V       : ndarray (S,)   — state value function
  Q       : ndarray (S, A) — state-action value function
  history : list           — convergence diagnostics
"""

import numpy as np
from typing import Dict, Any, Optional
from mdp_environment import TreatmentMDP, N_STATES, N_ACTIONS, TERMINAL_STATE


# ─────────────────────────────────────────────────────────────────────────────
# 1. Value Iteration
# ─────────────────────────────────────────────────────────────────────────────

def value_iteration(
    mdp: TreatmentMDP,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> Dict[str, Any]:
    """
    Classic Value Iteration (Bellman optimality operator).

    V_{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} T(s,a,s') V_k(s') ]

    Converges when max|V_{k+1} - V_k| < theta.

    Parameters
    ----------
    mdp      : TreatmentMDP
    theta    : convergence threshold
    max_iter : safety cap on iterations

    Returns
    -------
    dict with keys: policy, V, Q, history (delta per iteration)
    """
    T, R, gamma = mdp.T, mdp.R, mdp.gamma
    V       = np.zeros(N_STATES)
    history = []

    for iteration in range(1, max_iter + 1):
        # Q(s,a) = R(s,a) + γ Σ_{s'} T(s,a,s') V(s')
        Q = R + gamma * np.einsum("san,n->sa", T, V)

        V_new  = Q.max(axis=1)
        delta  = float(np.max(np.abs(V_new - V)))
        history.append(delta)
        V = V_new

        if delta < theta:
            break

    policy = Q.argmax(axis=1)
    policy[TERMINAL_STATE] = 0  # no meaningful action in terminal

    return {
        "policy"     : policy,
        "V"          : V,
        "Q"          : Q,
        "history"    : history,
        "iterations" : iteration,
        "converged"  : delta < theta,
        "method"     : "Value Iteration",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Policy Iteration
# ─────────────────────────────────────────────────────────────────────────────

def policy_iteration(
    mdp: TreatmentMDP,
    eval_theta: float = 1e-8,
    max_iter: int = 500,
) -> Dict[str, Any]:
    """
    Policy Iteration: alternates between policy evaluation and improvement.

    Policy Evaluation  — solve V^π via iterative Bellman expectation
    Policy Improvement — greedy update π(s) = argmax_a Q^π(s,a)

    Typically converges in far fewer outer iterations than Value Iteration.
    """
    T, R, gamma = mdp.T, mdp.R, mdp.gamma

    # Initialise with the safest policy (Monotherapy everywhere)
    policy  = np.full(N_STATES, 2, dtype=int)
    history = []

    for outer_iter in range(max_iter):

        # ── Policy Evaluation ────────────────────────────────────────
        V = np.zeros(N_STATES)
        for _ in range(10_000):
            # V(s) = R(s, π(s)) + γ Σ_{s'} T(s,π(s),s') V(s')
            V_new  = R[np.arange(N_STATES), policy] + gamma * np.einsum(
                "sn,n->s",
                T[np.arange(N_STATES), policy, :],
                V,
            )
            if np.max(np.abs(V_new - V)) < eval_theta:
                break
            V = V_new

        # ── Policy Improvement ───────────────────────────────────────
        Q          = R + gamma * np.einsum("san,n->sa", T, V)
        new_policy = Q.argmax(axis=1)
        new_policy[TERMINAL_STATE] = 0

        delta = int((new_policy != policy).sum())
        history.append(delta)
        policy = new_policy

        if delta == 0:   # policy stable → convergence
            break

    return {
        "policy"     : policy,
        "V"          : V,
        "Q"          : Q,
        "history"    : history,
        "iterations" : outer_iter + 1,
        "converged"  : delta == 0,
        "method"     : "Policy Iteration",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Q-Learning  (model-free)
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedy:
    """ε-greedy exploration with linear decay."""

    def __init__(
        self,
        n_actions  : int,
        eps_start  : float = 1.0,
        eps_end    : float = 0.05,
        decay_steps: int   = 50_000,
        seed       : int   = 0,
    ):
        self.n     = n_actions
        self.eps   = eps_start
        self.e_end = eps_end
        self.decay = (eps_start - eps_end) / decay_steps
        self._rng  = np.random.default_rng(seed)
        self._step = 0

    def __call__(self, Q_row: np.ndarray) -> int:
        self._step += 1
        self.eps    = max(self.e_end, self.eps - self.decay)
        if self._rng.random() < self.eps:
            return int(self._rng.integers(self.n))
        return int(np.argmax(Q_row))


def q_learning(
    mdp          : TreatmentMDP,
    n_episodes   : int   = 50_000,
    max_steps    : int   = 50,
    lr           : float = 0.1,
    lr_decay     : float = 0.9999,
    eps_start    : float = 1.0,
    eps_end      : float = 0.05,
    seed         : int   = 42,
) -> Dict[str, Any]:
    """
    Tabular Q-Learning with ε-greedy exploration and learning-rate decay.

    Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

    Parameters
    ----------
    mdp         : TreatmentMDP
    n_episodes  : total training episodes
    max_steps   : maximum steps per episode (horizon)
    lr          : initial learning rate α
    lr_decay    : multiplicative LR decay per episode
    eps_start   : initial exploration ε
    eps_end     : minimum exploration ε
    seed        : random seed

    Returns
    -------
    dict with keys: policy, V, Q, history (episode returns), method
    """
    Q       = np.zeros((N_STATES, N_ACTIONS))
    explore = EpsilonGreedy(N_ACTIONS, eps_start, eps_end,
                            decay_steps=n_episodes // 2, seed=seed)
    history = []
    alpha   = lr

    for episode in range(n_episodes):
        # Start from a random non-terminal state
        state = mdp.reset()
        ep_return = 0.0
        discount  = 1.0

        for step in range(max_steps):
            action                = explore(Q[state])
            next_state, r, done   = mdp.step(action)

            # Bellman update
            best_next             = np.max(Q[next_state])
            td_target             = r + mdp.gamma * best_next * (1 - int(done))
            Q[state, action]     += alpha * (td_target - Q[state, action])

            ep_return += discount * r
            discount  *= mdp.gamma
            state      = next_state

            if done:
                break

        history.append(ep_return)
        alpha = max(1e-4, alpha * lr_decay)

    policy = Q.argmax(axis=1)
    policy[TERMINAL_STATE] = 0
    V = Q.max(axis=1)

    return {
        "policy"     : policy,
        "V"          : V,
        "Q"          : Q,
        "history"    : history,
        "iterations" : n_episodes,
        "converged"  : True,   # RL doesn't have a hard convergence criterion
        "method"     : "Q-Learning",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all three and compare
# ─────────────────────────────────────────────────────────────────────────────

def solve_all(mdp: TreatmentMDP, verbose: bool = True) -> Dict[str, Dict]:
    """Run all three solvers and return results keyed by method name."""
    results = {}

    for name, fn, kwargs in [
        ("Value Iteration", value_iteration, {}),
        ("Policy Iteration", policy_iteration, {}),
        ("Q-Learning",       q_learning,       {"n_episodes": 60_000}),
    ]:
        res = fn(mdp, **kwargs)
        results[name] = res
        if verbose:
            print(f"[{name:20s}]  iters={res['iterations']:6d}  "
                  f"converged={res['converged']}  "
                  f"V(Moderate)={res['V'][3]:.3f}")

    return results
