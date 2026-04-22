"""
mdp_environment.py
==================
Defines the Healthcare Treatment Selection MDP.

Clinical context
----------------
A chronic-disease patient (e.g. Type-2 Diabetes / Hypertension / Heart Failure)
progresses through health states over discrete time steps (e.g. monthly visits).
At each visit the clinician chooses a treatment action.
The MDP captures:
  • State space   — patient health status (HbA1c / blood pressure / severity)
  • Action space  — available treatments (lifestyle, monotherapy, combo, intensive)
  • Transition    — probabilistic next-state given (state, action)
  • Reward        — clinical outcomes minus side-effect / cost penalties
  • Discount      — γ = 0.95  (future quality-of-life still valued)

States  (7 levels)
------------------
  0  Remission          — target met, no symptoms
  1  Controlled         — near-target, mild symptoms
  2  Mild               — slightly above target
  3  Moderate           — moderately above target
  4  Severe             — well above target, risk present
  5  Critical           — organ risk, hospitalisation likely
  6  Terminal / Adverse — end-state (absorbing)

Actions (5)
-----------
  0  Watchful waiting   — monitor only
  1  Lifestyle          — diet, exercise, education
  2  Monotherapy        — single first-line drug
  3  Combination        — two drugs / drug + lifestyle
  4  Intensive          — triple therapy / insulin / specialist referral

Reward shaping
--------------
  +10  reaching Remission
  + 5  reaching Controlled
  + 2  staying in same state (stability bonus)
  - 1  mild worsening (state + 1)
  - 5  significant worsening (state + 2)
  -10  reaching Critical
  -20  reaching Terminal
  - c  treatment cost penalty (action-dependent)
  - s  side-effect penalty (action-dependent, state-dependent)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_NAMES = [
    "Remission",
    "Controlled",
    "Mild",
    "Moderate",
    "Severe",
    "Critical",
    "Terminal",
]

ACTION_NAMES = [
    "Watchful Waiting",
    "Lifestyle Modification",
    "Monotherapy",
    "Combination Therapy",
    "Intensive Therapy",
]

N_STATES  = len(STATE_NAMES)   # 7
N_ACTIONS = len(ACTION_NAMES)  # 5

# Action cost penalties (higher action = more expensive / burdensome)
ACTION_COST = np.array([0.0, 0.5, 1.5, 2.5, 4.0])

# Side-effect multiplier: (action, state) → penalty
# More intensive treatment has higher side-effect risk in severe states
SIDE_EFFECT_BASE = np.array([0.0, 0.2, 0.5, 1.0, 2.0])  # per action

# State-severity multiplier for side effects
STATE_SIDE_MULT = np.array([0.5, 0.5, 0.7, 1.0, 1.3, 1.5, 0.0])

# Reward for arriving in each state
STATE_REWARD = np.array([10.0, 5.0, 0.0, -2.0, -5.0, -10.0, -20.0])

# Terminal state index
TERMINAL_STATE = 6

GAMMA = 0.95  # discount factor


# ─────────────────────────────────────────────────────────────────────────────
# Transition probability builder
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix() -> np.ndarray:
    """
    Returns T[s, a, s'] = P(next_state=s' | state=s, action=a).

    Design principles
    -----------------
    • Terminal state is absorbing: T[6, :, 6] = 1.0
    • More intensive actions shift probability mass toward better states.
    • Treatment efficacy follows a diminishing-returns curve (action 4
      is not infinitely better than action 3).
    • Spontaneous worsening is always possible (biological noise).
    """
    T = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # Treatment efficacy: improvement probability per action
    # Indexed as [action] → probability of moving UP (improving) one state
    action_improve_p = np.array([0.02, 0.10, 0.25, 0.40, 0.50])
    action_big_improve_p = np.array([0.00, 0.02, 0.05, 0.12, 0.20])

    # Spontaneous worsening probability (independent of action)
    worsen_p      = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.0])
    big_worsen_p  = np.array([0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.0])

    # Terminal state — absorbing
    T[TERMINAL_STATE, :, TERMINAL_STATE] = 1.0

    for s in range(N_STATES - 1):  # skip Terminal
        for a in range(N_ACTIONS):
            probs = np.zeros(N_STATES)

            p_big_imp = action_big_improve_p[a]
            p_imp     = action_improve_p[a]
            p_big_wor = big_worsen_p[s]
            p_wor     = worsen_p[s]

            # Big improvement (2 states up)
            if s >= 2:
                probs[s - 2] += p_big_imp
            elif s == 1:
                probs[0] += p_big_imp
            else:
                probs[0] += p_big_imp   # already at 0, stays

            # Small improvement (1 state up)
            if s >= 1:
                probs[s - 1] += p_imp
            else:
                probs[0] += p_imp

            # Spontaneous worsening (1 state down)
            effective_wor = max(0, p_wor - p_imp * 0.3)  # treatment dampens worsening
            if s < TERMINAL_STATE - 1:
                probs[s + 1] += effective_wor
            else:
                probs[TERMINAL_STATE] += effective_wor

            # Big worsening (2 states down)
            effective_big_wor = max(0, p_big_wor - p_big_imp * 0.3)
            if s < TERMINAL_STATE - 2:
                probs[s + 2] += effective_big_wor
            else:
                probs[TERMINAL_STATE] += effective_big_wor

            # Stay in same state (remainder)
            p_stay = 1.0 - probs.sum()
            probs[s] += max(0, p_stay)

            # Normalise (guard against floating-point drift)
            probs = np.clip(probs, 0, None)
            probs /= probs.sum()
            T[s, a, :] = probs

    return T


def compute_reward_matrix(T: np.ndarray) -> np.ndarray:
    """
    R[s, a] = expected immediate reward for taking action a in state s.

    R(s, a) = Σ_{s'} T(s,a,s') * STATE_REWARD[s']
              - ACTION_COST[a]
              - SIDE_EFFECT_BASE[a] * STATE_SIDE_MULT[s]
    """
    # Expected state reward
    expected_state_reward = np.einsum("san,n->sa", T, STATE_REWARD)

    # Action cost and side-effects
    cost    = ACTION_COST[np.newaxis, :]                           # (1, A)
    side_fx = (
        SIDE_EFFECT_BASE[np.newaxis, :] *
        STATE_SIDE_MULT[:, np.newaxis]
    )                                                               # (S, A)

    R = expected_state_reward - cost - side_fx
    return R


# ─────────────────────────────────────────────────────────────────────────────
# MDP Environment class
# ─────────────────────────────────────────────────────────────────────────────

class TreatmentMDP:
    """
    Healthcare Treatment Selection MDP.

    Parameters
    ----------
    gamma : float
        Discount factor (default 0.95).
    seed  : int
        Random seed for reproducible simulation.

    Attributes
    ----------
    T : ndarray (S, A, S)  transition probabilities
    R : ndarray (S, A)     expected immediate rewards
    """

    def __init__(self, gamma: float = GAMMA, seed: int = 42):
        self.gamma        = gamma
        self.n_states     = N_STATES
        self.n_actions    = N_ACTIONS
        self.state_names  = STATE_NAMES
        self.action_names = ACTION_NAMES

        self.T = build_transition_matrix()
        self.R = compute_reward_matrix(self.T)

        self._rng          = np.random.default_rng(seed)
        self._current_state: Optional[int] = None

    # ── Environment interface ────────────────────────────────────────

    def reset(self, initial_state: Optional[int] = None) -> int:
        """Reset to a given state (default: random non-terminal state)."""
        if initial_state is not None:
            self._current_state = initial_state
        else:
            self._current_state = int(self._rng.integers(1, N_STATES - 1))
        return self._current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take one treatment step.

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool  (True if Terminal reached)
        """
        s = self._current_state
        probs = self.T[s, action, :]
        next_state = int(self._rng.choice(N_STATES, p=probs))

        reward = self.R[s, action]
        done   = (next_state == TERMINAL_STATE)
        self._current_state = next_state
        return next_state, reward, done

    # ── Utilities ───────────────────────────────────────────────────

    def transition_probabilities(self, state: int, action: int) -> Dict[str, float]:
        """Return a readable dict of next-state probabilities."""
        return {
            STATE_NAMES[s_]: round(float(self.T[state, action, s_]), 4)
            for s_ in range(N_STATES)
            if self.T[state, action, s_] > 0.0
        }

    def describe(self) -> str:
        lines = [
            "Healthcare Treatment Selection MDP",
            "=" * 40,
            f"States  ({N_STATES}): {', '.join(STATE_NAMES)}",
            f"Actions ({N_ACTIONS}): {', '.join(ACTION_NAMES)}",
            f"Discount γ = {self.gamma}",
            "",
            "Reward matrix R[state, action] (rows=states, cols=actions):",
        ]
        header = "        " + "  ".join(f"{a[:8]:>8}" for a in ACTION_NAMES)
        lines.append(header)
        for s in range(N_STATES):
            row = f"{STATE_NAMES[s][:8]:>8}: " + "  ".join(
                f"{self.R[s, a]:>8.2f}" for a in range(N_ACTIONS)
            )
            lines.append(row)
        return "\n".join(lines)
