"""Standalone verification for SalinityLSTMModel n_scenarios refactor.

Three checks:
  1. Single-scenario equivalence: re-running n_scenarios=1 with the same
     input trace must produce identical sf_mu trajectories on two runs.
  2. Ensemble member-0 equivalence: a multi-scenario run where scenario 0
     receives the baseline trace must produce sf_mu[:, 0] equal to the
     single-scenario baseline (catches batching mistakes that perturb
     scenario 0).
  3. Diverging-state check: with two scenarios receiving different flow
     traces, the per-scenario LSTM cell-state rows must diverge.

Run from the PywrDRB-ML root:
    python tests/test_salinity_lstm_n_scenarios.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PYWRDRB_ML_DIR", str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT))

from src.lstm_model import SalinityLSTMModel  # noqa: E402

MODEL_YAML = str(PROJ_ROOT / "models" / "SalinityLSTM" / "SalinityLSTM.yml")
START = "2000-01-01"
END = "2000-02-29"
N_STEPS = 30  # exercise enough steps for the 7-day-avg recurrence to engage


def _run_single(flows_T, flows_S, seed=0):
    """Run a single-scenario simulation and return (sf_mu_series, sf_sd_series, h_t)."""
    import torch
    torch.manual_seed(seed)
    model = SalinityLSTMModel(
        model_salinity=MODEL_YAML,
        start_date=START,
        end_date=END,
        debug=True,
        disable_tqdm=True,
        n_scenarios=1,
    )
    model.load_data()

    sf_mu_series = []
    sf_sd_series = []
    for step in range(N_STEPS):
        model.update(t=step, Q_Trenton=flows_T[step], Q_Schuylkill=flows_S[step])
        sf_mu_series.append(float(np.asarray(model.sf_mu).reshape(-1)[0]))
        sf_sd_series.append(float(np.asarray(model.sf_sd).reshape(-1)[0]))
    return np.array(sf_mu_series), np.array(sf_sd_series), model.lstm.h_t.detach().numpy().copy()


def _run_multi(flows_T_2d, flows_S_2d, seed=0):
    """Run a multi-scenario simulation. flows_*_2d shape: (N_STEPS, n_scenarios)."""
    import torch
    torch.manual_seed(seed)
    n_scen = flows_T_2d.shape[1]
    model = SalinityLSTMModel(
        model_salinity=MODEL_YAML,
        start_date=START,
        end_date=END,
        debug=True,
        disable_tqdm=True,
        n_scenarios=n_scen,
    )
    model.load_data()

    sf_mu_series = np.empty((N_STEPS, n_scen))
    sf_sd_series = np.empty((N_STEPS, n_scen))
    for step in range(N_STEPS):
        model.update(
            t=step,
            Q_Trenton=flows_T_2d[step, :],
            Q_Schuylkill=flows_S_2d[step, :],
        )
        sf_mu_series[step, :] = np.asarray(model.sf_mu).reshape(-1)
        sf_sd_series[step, :] = np.asarray(model.sf_sd).reshape(-1)
    return sf_mu_series, sf_sd_series, model.lstm.h_t.detach().numpy().copy()


def main():
    rng = np.random.default_rng(42)
    base_T = 4000.0 + 500.0 * rng.standard_normal(N_STEPS)
    base_S = 1500.0 + 300.0 * rng.standard_normal(N_STEPS)
    perturbed_T = base_T + 1500.0  # large perturbation to force divergence
    perturbed_S = base_S + 800.0

    print("[1/3] Single-scenario reproducibility check...")
    mu_a, _, ht_a = _run_single(base_T, base_S, seed=0)
    mu_b, _, ht_b = _run_single(base_T, base_S, seed=0)
    assert np.array_equal(mu_a, mu_b), "Two single-scenario runs with same seed must match exactly"
    assert np.allclose(ht_a, ht_b), "h_t must match across reproducibility runs"
    print(f"  OK. sf_mu[0]={mu_a[0]:.4f}, sf_mu[-1]={mu_a[-1]:.4f}, mean={mu_a.mean():.4f}")

    print("[2/3] Ensemble member-0 equivalence check...")
    flows_T_2d = np.stack([base_T, perturbed_T], axis=1)  # (N_STEPS, 2)
    flows_S_2d = np.stack([base_S, perturbed_S], axis=1)
    mu_multi, _, ht_multi = _run_multi(flows_T_2d, flows_S_2d, seed=0)
    if not np.allclose(mu_multi[:, 0], mu_a, atol=1e-5):
        max_diff = np.max(np.abs(mu_multi[:, 0] - mu_a))
        raise AssertionError(
            f"Ensemble member 0 must match single-scenario baseline. Max abs diff: {max_diff}"
        )
    print(f"  OK. max|mu_multi[:,0] - mu_single| = {np.max(np.abs(mu_multi[:, 0] - mu_a)):.2e}")

    print("[3/3] Diverging-state check (two scenarios with different flows)...")
    if np.allclose(mu_multi[:, 0], mu_multi[:, 1], atol=1e-3):
        raise AssertionError(
            "Scenarios with diverging flows must produce diverging sf_mu trajectories. "
            "Got identical outputs."
        )
    print(f"  OK. sf_mu RMSE between scenarios: {np.sqrt(np.mean((mu_multi[:, 0] - mu_multi[:, 1]) ** 2)):.4f}")

    # Cell-state divergence
    h_diff = np.linalg.norm(ht_multi[0] - ht_multi[1])
    assert h_diff > 1e-4, f"LSTM h_t rows must diverge across scenarios; got |diff|={h_diff:.2e}"
    print(f"  OK. |h_t[0] - h_t[1]| = {h_diff:.4f}")

    # The pywr-wrapper tests below cover the same record-shape and
    # per-scenario indexing semantics that UpdateSaltFrontLocation /
    # SaltFrontLocation / FlowTargetSaltFrontAdjustmentRatio rely on,
    # without spinning up a full pywrdrb ensemble (which requires several
    # staged HDF5 files unrelated to the salinity refactor itself).
    print("[4/5] Records-shape sanity for multi-scenario debug records...")
    # Re-run a 2-scenario simulation with debug=True and verify that the
    # records dict is shaped (length, n_scenarios) and populated from t=0
    # onward.
    import torch
    torch.manual_seed(0)
    n_scen = 2
    model = SalinityLSTMModel(
        model_salinity=MODEL_YAML,
        start_date=START,
        end_date=END,
        debug=True,
        disable_tqdm=True,
        n_scenarios=n_scen,
    )
    model.load_data()
    for step in range(N_STEPS):
        model.update(t=step, Q_Trenton=flows_T_2d[step, :], Q_Schuylkill=flows_S_2d[step, :])
    sf_records = model.records["sf_mu"]
    assert sf_records.ndim == 2 and sf_records.shape[1] == n_scen, (
        f"records['sf_mu'] must be 2D (length, n_scen); got {sf_records.shape}"
    )
    pop = np.isfinite(sf_records[:N_STEPS, :]).all(axis=0)
    assert pop.all(), f"all scenario columns must be populated for t in [0, {N_STEPS}); got {pop}"
    print(f"  OK. records.shape={sf_records.shape}, populated_columns={pop.sum()}/{n_scen}")

    print("[5/5] Per-scenario indexing semantics (mirrors pywr wrappers)...")
    # The pywr wrappers do: `float(np.asarray(ml.sf_mu).reshape(-1)[gid])`.
    # Verify both: (a) the indexing yields a finite scalar per scenario,
    # (b) records writes into [t, gid] don't bleed across scenarios.
    for gid in range(n_scen):
        v = float(np.asarray(model.sf_mu).reshape(-1)[gid])
        assert np.isfinite(v), f"sf_mu[gid={gid}] must be finite, got {v}"
    # Simulate a wrapper write into records[key][t, gid] and confirm
    # cross-scenario isolation.
    drought_key = "drought_idx"
    model.records[drought_key][N_STEPS - 1, 0] = 6
    model.records[drought_key][N_STEPS - 1, 1] = 0
    assert model.records[drought_key][N_STEPS - 1, 0] == 6
    assert model.records[drought_key][N_STEPS - 1, 1] == 0
    print("  OK. per-scenario indexing + isolated record writes verified.")

    print("\nAll Phase 1 + wrapper-semantics checks passed.")


if __name__ == "__main__":
    main()
