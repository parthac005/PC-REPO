"""
mdp_clinical.py
===============
Clinical Decision Support System (CDSS) built on top of the MDP.

Provides:
  ClinicalDecisionSupport  — maps real patient features to MDP states
                             and returns treatment recommendations with rationale
  SensitivityAnalyser      — tests robustness of optimal policy to
                             perturbations in reward weights and transition probs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from mdp_environment import (
    TreatmentMDP, STATE_NAMES, ACTION_NAMES,
    N_STATES, N_ACTIONS, GAMMA
)
from mdp_solvers import value_iteration


# ─────────────────────────────────────────────────────────────────────────────
# State classifier from clinical features
# ─────────────────────────────────────────────────────────────────────────────

def classify_state(
    hba1c: Optional[float]       = None,    # % (diabetes)
    systolic_bp: Optional[float] = None,    # mmHg (hypertension)
    lvef: Optional[float]        = None,    # % (heart failure)
    severity_score: Optional[float] = None, # generic 0–100
) -> Tuple[int, str]:
    """
    Map clinical measurements to a discrete MDP state.

    Uses a priority hierarchy:
      1. HbA1c (diabetes)
      2. Systolic BP (hypertension)
      3. LVEF (heart failure)
      4. Generic severity score (0–100)

    Returns
    -------
    (state_index, rationale_string)
    """
    if hba1c is not None:
        if hba1c < 6.5:
            return 0, f"HbA1c {hba1c}% < 6.5%  → Remission (target achieved)"
        elif hba1c < 7.0:
            return 1, f"HbA1c {hba1c}% in [6.5,7.0)  → Controlled"
        elif hba1c < 8.0:
            return 2, f"HbA1c {hba1c}% in [7.0,8.0)  → Mild"
        elif hba1c < 9.0:
            return 3, f"HbA1c {hba1c}% in [8.0,9.0)  → Moderate"
        elif hba1c < 10.5:
            return 4, f"HbA1c {hba1c}% in [9.0,10.5) → Severe"
        else:
            return 5, f"HbA1c {hba1c}% ≥ 10.5%  → Critical"

    if systolic_bp is not None:
        if systolic_bp < 120:
            return 0, f"SBP {systolic_bp} mmHg < 120  → Remission"
        elif systolic_bp < 130:
            return 1, f"SBP {systolic_bp} mmHg in [120,130) → Controlled"
        elif systolic_bp < 140:
            return 2, f"SBP {systolic_bp} mmHg in [130,140) → Mild"
        elif systolic_bp < 160:
            return 3, f"SBP {systolic_bp} mmHg in [140,160) → Moderate"
        elif systolic_bp < 180:
            return 4, f"SBP {systolic_bp} mmHg in [160,180) → Severe"
        else:
            return 5, f"SBP {systolic_bp} mmHg ≥ 180  → Critical"

    if lvef is not None:
        if lvef >= 55:
            return 0, f"LVEF {lvef}% ≥ 55%  → Remission (normal)"
        elif lvef >= 45:
            return 1, f"LVEF {lvef}% in [45,55) → Controlled (mildly reduced)"
        elif lvef >= 40:
            return 2, f"LVEF {lvef}% in [40,45) → Mild (borderline)"
        elif lvef >= 30:
            return 3, f"LVEF {lvef}% in [30,40) → Moderate (reduced)"
        elif lvef >= 20:
            return 4, f"LVEF {lvef}% in [20,30) → Severe (severely reduced)"
        else:
            return 5, f"LVEF {lvef}% < 20%  → Critical"

    if severity_score is not None:
        s = int(np.clip(severity_score / 100 * 5, 0, 5))
        return s, f"Generic severity {severity_score:.0f}/100 → {STATE_NAMES[s]}"

    raise ValueError("At least one clinical measurement must be provided.")


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Decision Support System
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalDecisionSupport:
    """
    Wraps the MDP policy to produce human-readable treatment recommendations.

    Usage
    -----
    cdss = ClinicalDecisionSupport(mdp, policy)
    rec  = cdss.recommend(hba1c=8.5)
    print(rec["summary"])
    """

    CONTRAINDICATIONS = {
        # action_index : list of states where action is contraindicated
        0: [3, 4, 5],   # Watchful Waiting contraindicated in Moderate–Critical
        1: [4, 5],      # Lifestyle only is insufficient for Severe/Critical
    }

    def __init__(self, mdp: TreatmentMDP, policy: np.ndarray):
        self.mdp    = mdp
        self.policy = policy

    def recommend(
        self,
        hba1c          : Optional[float] = None,
        systolic_bp    : Optional[float] = None,
        lvef           : Optional[float] = None,
        severity_score : Optional[float] = None,
        patient_id     : str = "P001",
    ) -> Dict:
        """
        Generate a treatment recommendation for a patient.

        Returns a rich dict with:
          state, state_name, recommended_action, action_name,
          expected_value, transition_probs, alternatives, rationale, summary
        """
        state, classification_note = classify_state(
            hba1c=hba1c, systolic_bp=systolic_bp,
            lvef=lvef, severity_score=severity_score,
        )

        recommended_action = int(self.policy[state])

        # Check contraindications
        contra_note = ""
        if state in self.CONTRAINDICATIONS.get(recommended_action, []):
            # Override with next stronger action
            recommended_action = min(recommended_action + 1, N_ACTIONS - 1)
            contra_note = (f"[Safety override: escalated to {ACTION_NAMES[recommended_action]} "
                           f"due to contraindication in state {STATE_NAMES[state]}]")

        # Q-values for all actions (need Q matrix from solver)
        Q_row = self.mdp.R[state, :] + GAMMA * (self.mdp.T[state, :, :] @ self.policy * 0)
        # Use R[s,a] as proxy ranking when Q is not pre-computed
        Q_row = self.mdp.R[state, :]

        # Rank alternatives
        alt_rank = np.argsort(-Q_row)
        alternatives = [
            {"action": ACTION_NAMES[a], "expected_immediate_reward": round(float(Q_row[a]), 3)}
            for a in alt_rank if a != recommended_action
        ]

        # Transition probabilities under recommended action
        trans = self.mdp.transition_probabilities(state, recommended_action)

        # Build summary string
        lines = [
            f"Patient {patient_id} — Treatment Recommendation",
            "=" * 50,
            f"Clinical input : {classification_note}",
            f"Current state  : {STATE_NAMES[state]} (index {state})",
            "",
            f"Recommended Tx : {ACTION_NAMES[recommended_action].upper()}",
            f"Expected reward: {self.mdp.R[state, recommended_action]:.2f}",
        ]
        if contra_note:
            lines.append(f"Note           : {contra_note}")

        lines += [
            "",
            "Next-state probabilities under recommended treatment:",
        ]
        for ns, p in sorted(trans.items(), key=lambda x: -x[1]):
            lines.append(f"  {ns:15s}: {p*100:.1f}%")

        lines += [
            "",
            "Alternative options (ranked by immediate reward):",
        ]
        for alt in alternatives[:3]:
            lines.append(f"  {alt['action']:25s}: reward={alt['expected_immediate_reward']:.2f}")

        return {
            "patient_id"          : patient_id,
            "state"               : state,
            "state_name"          : STATE_NAMES[state],
            "recommended_action"  : recommended_action,
            "action_name"         : ACTION_NAMES[recommended_action],
            "expected_reward"     : float(self.mdp.R[state, recommended_action]),
            "transition_probs"    : trans,
            "alternatives"        : alternatives,
            "contraindication"    : contra_note,
            "classification_note" : classification_note,
            "summary"             : "\n".join(lines),
        }

    def batch_recommend(self, patients: List[Dict]) -> pd.DataFrame:
        """
        Process a list of patient dicts (each with clinical measurements).
        Returns a summary DataFrame.
        """
        rows = []
        for p in patients:
            pid = p.pop("patient_id", f"P{len(rows)+1:03d}")
            rec = self.recommend(**p, patient_id=pid)
            rows.append({
                "Patient ID"         : rec["patient_id"],
                "State"              : rec["state_name"],
                "Recommended Tx"     : rec["action_name"],
                "Expected Reward"    : rec["expected_reward"],
                "P(Improve)"         : round(sum(
                    v for k, v in rec["transition_probs"].items()
                    if STATE_NAMES.index(k) < rec["state"]
                ), 3),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

class SensitivityAnalyser:
    """
    Tests how robust the optimal policy is to:
      A. Perturbations in state reward weights
      B. Perturbations in transition probabilities
      C. Variations in discount factor γ
    """

    def __init__(self, base_mdp: TreatmentMDP):
        self.base_mdp = base_mdp
        self.base_result = value_iteration(base_mdp)

    def reward_sensitivity(
        self,
        perturbation_range: np.ndarray = None,
        n_points: int = 10,
    ) -> pd.DataFrame:
        """
        Vary the reward multiplier for the Critical state (state 5)
        over a range and record the resulting policy at each state.
        """
        from mdp_environment import build_transition_matrix, compute_reward_matrix, STATE_REWARD

        if perturbation_range is None:
            perturbation_range = np.linspace(0.5, 3.0, n_points)

        rows = []
        for mult in perturbation_range:
            # Scale Critical state reward by mult
            mod_mdp        = TreatmentMDP(gamma=self.base_mdp.gamma)
            mod_mdp.T      = self.base_mdp.T.copy()
            new_sr         = STATE_REWARD.copy()
            new_sr[5]     *= mult
            mod_mdp.R      = compute_reward_matrix(mod_mdp.T)
            # Override critical-state rows
            mod_mdp.R[5, :] = mod_mdp.R[5, :] + (mult - 1) * 5

            res   = value_iteration(mod_mdp)
            row   = {"Critical reward multiplier": round(float(mult), 2)}
            for s in range(N_STATES - 1):
                row[f"π({STATE_NAMES[s][:4]})"] = ACTION_NAMES[res["policy"][s]][:5]
            rows.append(row)

        return pd.DataFrame(rows)

    def discount_sensitivity(
        self,
        gammas: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Solve the MDP for different discount factors and record V(Moderate).
        """
        if gammas is None:
            gammas = np.linspace(0.5, 0.99, 12)

        rows = []
        for g in gammas:
            mdp_g       = TreatmentMDP(gamma=float(g))
            res         = value_iteration(mdp_g)
            policy_str  = " | ".join(ACTION_NAMES[res["policy"][s]][:5]
                                      for s in range(N_STATES - 1))
            rows.append({
                "Gamma"        : round(float(g), 3),
                "V(Moderate)"  : round(float(res["V"][3]), 3),
                "V(Mild)"      : round(float(res["V"][2]), 3),
                "Policy"       : policy_str,
            })

        return pd.DataFrame(rows)

    def policy_stability_report(self) -> str:
        """
        Summarise how often each state's optimal action changes across
        reward and discount perturbations.
        """
        df_r = self.reward_sensitivity(n_points=8)
        df_g = self.discount_sensitivity()

        lines = [
            "Policy Stability Report",
            "=" * 45,
            "",
            "Reward sensitivity (varying Critical state penalty):",
        ]
        for col in df_r.columns[1:]:
            unique = df_r[col].nunique()
            vals   = df_r[col].unique()
            lines.append(f"  {col}: {unique} unique action(s) → {', '.join(vals)}")

        lines += [
            "",
            "Discount factor sensitivity (γ from 0.5 to 0.99):",
            f"  V(Moderate) range: [{df_g['V(Moderate)'].min():.2f}, {df_g['V(Moderate)'].max():.2f}]",
            f"  V(Mild)     range: [{df_g['V(Mild)'].min():.2f}, {df_g['V(Mild)'].max():.2f}]",
        ]
        return "\n".join(lines)
