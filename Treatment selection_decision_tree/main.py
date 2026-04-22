"""
main.py
=======
End-to-end demonstration of the Healthcare Treatment Selection MDP.

Runs:
  1. Environment setup + description
  2. All three solvers (VI, PI, Q-Learning)
  3. Policy visualisation & convergence plots
  4. Single patient simulation + cohort simulation
  5. Policy comparison vs baselines
  6. Clinical decision support examples
  7. Sensitivity analysis
  8. All figures saved to ./results/
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

# ── Module imports ────────────────────────────────────────────────────────────
from mdp_environment  import TreatmentMDP, STATE_NAMES, ACTION_NAMES
from mdp_solvers      import solve_all
from mdp_simulation   import (
    simulate_patient, simulate_cohort,
    compare_policies, make_baseline_policies,
)
from mdp_visualisation import (
    plot_transition_heatmap, plot_reward_matrix,
    plot_value_function, plot_policy, plot_convergence,
    plot_patient_trajectory, plot_cohort_outcomes,
    plot_policy_comparison, plot_q_values,
)
from mdp_clinical import ClinicalDecisionSupport, SensitivityAnalyser

OUT = Path("results"); OUT.mkdir(exist_ok=True)

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment
# ─────────────────────────────────────────────────────────────────────────────

section("1. MDP Environment")
mdp = TreatmentMDP(gamma=0.95, seed=42)
print(mdp.describe())

# Save environment plots
plot_transition_heatmap(mdp, save_path=str(OUT/"transition_heatmap.png"))
plot_reward_matrix(mdp,      save_path=str(OUT/"reward_matrix.png"))
print(f"\nSaved: transition_heatmap.png, reward_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Solve
# ─────────────────────────────────────────────────────────────────────────────

section("2. Solving the MDP")
t0 = time.time()
results = solve_all(mdp, verbose=True)
print(f"\nTotal solve time: {time.time()-t0:.2f}s")

# Print optimal policies
print("\nOptimal Policies (state → recommended action):")
header = f"  {'State':12s}" + "".join(f"  {m[:12]:12s}" for m in results)
print(header)
for s in range(len(STATE_NAMES) - 1):
    row = f"  {STATE_NAMES[s]:12s}"
    for res in results.values():
        row += f"  {ACTION_NAMES[res['policy'][s]][:12]:12s}"
    print(row)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualise policy & convergence
# ─────────────────────────────────────────────────────────────────────────────

section("3. Visualisations — Policy & Convergence")
plot_value_function(results, save_path=str(OUT/"value_function.png"))
plot_policy(results,         save_path=str(OUT/"optimal_policy.png"))
plot_convergence(results,    save_path=str(OUT/"convergence.png"))

# Q-value heatmap (Value Iteration)
plot_q_values(results["Value Iteration"]["Q"],
              title="Q-Values — Value Iteration",
              save_path=str(OUT/"q_values_vi.png"))
plot_q_values(results["Q-Learning"]["Q"],
              title="Q-Values — Q-Learning",
              save_path=str(OUT/"q_values_ql.png"))

print("Saved: value_function.png, optimal_policy.png, convergence.png")
print("       q_values_vi.png, q_values_ql.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Patient simulation
# ─────────────────────────────────────────────────────────────────────────────

section("4. Patient Trajectory Simulation")
optimal_policy = results["Value Iteration"]["policy"]

# Simulate patients starting from different states
for start_state, label in [(2, "Mild"), (3, "Moderate"), (4, "Severe")]:
    traj = simulate_patient(mdp, optimal_policy,
                             initial_state=start_state,
                             n_steps=24, seed=start_state * 7)
    plot_patient_trajectory(
        traj,
        title=f"Patient Trajectory — Starting State: {label}",
        save_path=str(OUT/f"trajectory_{label.lower()}.png"),
    )
    final = traj["state_name"].iloc[-1]
    total = traj["cumulative_reward"].iloc[-1]
    print(f"  {label:10s} → final={final:12s}  total_reward={total:.1f}")

print("\nSaved: trajectory_mild.png, trajectory_moderate.png, trajectory_severe.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cohort simulation
# ─────────────────────────────────────────────────────────────────────────────

section("5. Cohort Simulation (n=500 patients)")
cohort = simulate_cohort(mdp, optimal_policy,
                          n_patients=500, n_steps=24, seed=0)

print("\nFinal state distribution:")
print(cohort["summary"].to_string(index=False))
print(f"\nMean total return : {cohort['mean_return']:.2f}")
print(f"Std  total return : {cohort['std_return']:.2f}")

plot_cohort_outcomes(cohort, policy_name="Optimal (VI)",
                     save_path=str(OUT/"cohort_outcomes.png"))
print("\nSaved: cohort_outcomes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Policy comparison
# ─────────────────────────────────────────────────────────────────────────────

section("6. Policy Comparison")
baseline_policies = make_baseline_policies()
all_policies = {
    "Optimal (VI)"     : results["Value Iteration"]["policy"],
    "Optimal (Q-Learn)": results["Q-Learning"]["policy"],
    **baseline_policies,
}

cmp_df = compare_policies(mdp, all_policies, n_patients=300, n_steps=24, seed=0)
print(cmp_df.round(2).to_string())

plot_policy_comparison(cmp_df, save_path=str(OUT/"policy_comparison.png"))
print("\nSaved: policy_comparison.png")

# Save CSV
cmp_df.to_csv(str(OUT/"policy_comparison.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Clinical decision support
# ─────────────────────────────────────────────────────────────────────────────

section("7. Clinical Decision Support — Example Patients")
cdss = ClinicalDecisionSupport(mdp, optimal_policy)

patients = [
    {"patient_id": "PT-001", "hba1c": 9.2},
    {"patient_id": "PT-002", "hba1c": 7.4},
    {"patient_id": "PT-003", "systolic_bp": 165},
    {"patient_id": "PT-004", "lvef": 28.0},
    {"patient_id": "PT-005", "hba1c": 6.2},
]

for p in patients:
    pid = p["patient_id"]; kwargs = {k: v for k, v in p.items() if k != "patient_id"}
    rec = cdss.recommend(**kwargs, patient_id=pid)
    print(f"\n{rec['summary']}")

# Batch recommendation
batch_input = [
    {"patient_id": f"PT-{100+i:03d}", "hba1c": hba1c}
    for i, hba1c in enumerate(np.round(np.linspace(6.0, 12.0, 10), 1))
]
batch_df = cdss.batch_recommend(batch_input)
print("\n--- Batch Recommendations ---")
print(batch_df.to_string(index=False))
batch_df.to_csv(str(OUT/"batch_recommendations.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

section("8. Sensitivity Analysis")
analyser = SensitivityAnalyser(mdp)

print("\nDiscount factor sensitivity:")
df_gamma = analyser.discount_sensitivity()
print(df_gamma.to_string(index=False))

print("\nReward sensitivity:")
df_reward = analyser.reward_sensitivity(n_points=6)
print(df_reward.to_string(index=False))

print("\n" + analyser.policy_stability_report())

df_gamma.to_csv(str(OUT/"sensitivity_gamma.csv"),  index=False)
df_reward.to_csv(str(OUT/"sensitivity_reward.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")
print(f"""
MDP solved successfully with three algorithms:

  Value Iteration : {results['Value Iteration']['iterations']} iterations, converged={results['Value Iteration']['converged']}
  Policy Iteration: {results['Policy Iteration']['iterations']} outer loops,  converged={results['Policy Iteration']['converged']}
  Q-Learning      : {results['Q-Learning']['iterations']} episodes

Key policy insights:
  • Remission / Controlled → Watchful Waiting or Lifestyle (preserve gains)
  • Mild / Moderate        → Monotherapy or Combination (step-up approach)
  • Severe / Critical      → Intensive Therapy (maximise QoL / survival)

All outputs saved to: {OUT}/
""")
