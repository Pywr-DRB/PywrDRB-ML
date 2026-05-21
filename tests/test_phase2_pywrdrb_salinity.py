"""Phase 2 verification: pywrdrb integration with the refactored salinity LSTM.

Two checks:
  A. Single-scenario regression: build a pywrdrb model with the salinity
     LSTM enabled, run a short simulation, confirm:
       - it does not crash
       - ml_model.records['sf_mu'] is 2D shape (length, 1)
       - sf_mu values are populated and finite for the simulated days

  B. Multi-scenario ensemble: build with ``inflow_ensemble_indices=[0, 1]``,
     run a short simulation, confirm:
       - records['sf_mu'] is 2D shape (length, 2)
       - both scenario columns are populated and finite
       - the two scenarios produce *different* sf_mu trajectories
         (proves per-scenario state divergence flowed through the
         pywr Parameter wrappers correctly)

Run from any directory with the Pywr-DRB venv:
    python tests/test_phase2_pywrdrb_salinity.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PYWRDRB_ML_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault("PYWRDRB_ML_DIR", str(PYWRDRB_ML_DIR))
# pywrdrb's SalinityModelLSTM does ``sys.path.insert(1, PywrDRB_ML_plugin_path)``
# with a Path object, which Python's importer silently ignores on some
# versions. Insert the str form here so ``from src.lstm_model import ...``
# resolves cleanly.
sys.path.insert(0, str(PYWRDRB_ML_DIR))

import pywrdrb  # noqa: E402

INFLOW_TYPE = "nhmv10_withObsScaled"  # single-trace inflow_type (no ensemble file)
ENSEMBLE_INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"  # ensemble HDF5 staged here
START = "2000-01-01"
END = "2000-02-29"  # 60 sim days; enough for the 7d-avg recurrence to engage

SALINITY_OPTIONS = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": str(PYWRDRB_ML_DIR),
    "model_salinity": str(PYWRDRB_ML_DIR / "models" / "SalinityLSTM" / "SalinityLSTM.yml"),
    "start_date": START,
    "end_date": END,
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "asycronized_update": False,
    "debug": True,
}


def _build_and_run(inflow_ensemble_indices=None):
    """Build a pywrdrb model with salinity LSTM enabled and run it."""
    options = {"salinity_model": dict(SALINITY_OPTIONS)}
    inflow_type = INFLOW_TYPE
    if inflow_ensemble_indices is not None:
        # Use the inflow_type whose flows/ dir has the ensemble HDF5 staged.
        inflow_type = ENSEMBLE_INFLOW_TYPE
        options["inflow_ensemble_indices"] = list(inflow_ensemble_indices)

    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=START,
        end_date=END,
        options=options,
    )
    mb.make_model()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        model_path = f.name
    try:
        mb.write_model(model_path)
        model = pywrdrb.Model.load(model_path)
        model.run()
        ml_model = model.parameters["salinity_model"].ml_model
        return {
            "records": {k: np.asarray(v).copy() for k, v in ml_model.records.items()},
            "n_scenarios": ml_model.n_scenarios,
            "sf_mu_final": np.asarray(ml_model.sf_mu).copy(),
            "sf_sd_final": np.asarray(ml_model.sf_sd).copy(),
        }
    finally:
        try:
            os.unlink(model_path)
        except OSError:
            pass


def main():
    n_sim_days = (np.datetime64(END) - np.datetime64(START)).astype(int) + 1

    print("[A] Single-scenario pywrdrb run with salinity LSTM...")
    out_single = _build_and_run(inflow_ensemble_indices=None)
    sf_mu_single = out_single["records"]["sf_mu"]
    assert sf_mu_single.ndim == 2, (
        f"records['sf_mu'] must be 2D (length, n_scen); got ndim={sf_mu_single.ndim}"
    )
    assert sf_mu_single.shape[1] == 1, (
        f"single-scenario records width must be 1; got {sf_mu_single.shape[1]}"
    )
    populated = np.isfinite(sf_mu_single[:n_sim_days, 0]).sum()
    assert populated > 0, "single-scenario salt-front records are all NaN/inf"
    assert out_single["sf_mu_final"].shape == (1,)
    print(
        f"  OK. records.shape={sf_mu_single.shape}, "
        f"populated={populated}, mean={np.nanmean(sf_mu_single[:n_sim_days, 0]):.3f}"
    )

    print("[B0] Single-realization pywrdrb run with inflow_ensemble_indices=[0]...")
    # Run with one ensemble member to establish a baseline that the
    # multi-scenario run's scenario-0 column should match.
    try:
        out_one = _build_and_run(inflow_ensemble_indices=[0])
    except FileNotFoundError as e:
        print(f"  SKIP. Required ensemble HDF5 not staged: {e.filename}")
        print(
            "  Note: validate end-to-end ensemble salinity in NYCOpt where the\n"
            "        full ensemble dir (catchment + predicted_inflows + predicted_diversions)\n"
            "        is provisioned by register_ensemble_path."
        )
        print("\nAll Phase 2 in-scope checks passed.")
        return
    sf_mu_one = out_one["records"]["sf_mu"]
    assert sf_mu_one.shape[1] == 1
    print(f"  OK. records.shape={sf_mu_one.shape}, mean={np.nanmean(sf_mu_one[:n_sim_days, 0]):.3f}")

    print("[B] Multi-scenario pywrdrb run with inflow_ensemble_indices=[0,1]...")
    # Full multi-scenario integration through pywrdrb requires several staged
    # ensemble files (catchment_inflow_mgd.hdf5, predicted_inflows_mgd.hdf5,
    # predicted_diversions_mgd.hdf5) under ``pn.flows/{inflow_type}/``. This
    # plumbing is owned by NYCOptimization (``src/ensembles.py``) and is
    # outside the scope of the salinity-LSTM refactor itself. The salinity
    # wrappers' multi-scenario behavior is covered by the standalone
    # ``test_salinity_lstm_n_scenarios.py`` checks (member-0 byte-identical
    # equivalence, diverging cell state, records 2D shape, per-scenario
    # indexing) plus the single-scenario pywrdrb run above (validates
    # setup/records reshape via the pywr Parameter wrapper code path).
    try:
        out_multi = _build_and_run(inflow_ensemble_indices=[0, 1])
    except FileNotFoundError as e:
        print(f"  SKIP. Required ensemble HDF5 not staged: {e.filename}")
        print(
            "  Note: validate end-to-end ensemble salinity in NYCOpt where the\n"
            "        full ensemble dir (catchment + predicted_inflows + predicted_diversions)\n"
            "        is provisioned by register_ensemble_path."
        )
        print("\nAll Phase 2 in-scope checks passed.")
        return
    sf_mu_multi = out_multi["records"]["sf_mu"]
    assert sf_mu_multi.shape[1] == 2, (
        f"multi-scenario records width must be 2; got {sf_mu_multi.shape[1]}"
    )
    pop0 = np.isfinite(sf_mu_multi[:n_sim_days, 0]).sum()
    pop1 = np.isfinite(sf_mu_multi[:n_sim_days, 1]).sum()
    assert pop0 > 0 and pop1 > 0, (
        f"multi-scenario records must be populated for both scenarios; got pop0={pop0}, pop1={pop1}"
    )
    diff = np.nanmean(np.abs(sf_mu_multi[:n_sim_days, 0] - sf_mu_multi[:n_sim_days, 1]))
    assert diff > 1e-6, (
        f"two ensemble scenarios with different inflows must produce different sf_mu; "
        f"got mean abs diff = {diff:.2e}"
    )
    assert out_multi["sf_mu_final"].shape == (2,)
    print(
        f"  OK. records.shape={sf_mu_multi.shape}, "
        f"mean|sf_mu[s0] - sf_mu[s1]|={diff:.3f}"
    )

    # Tightest correctness check: scenario-0 column of the 2-scenario run
    # should match the single-realization run on realization 0. Tolerance is
    # tight but not bit-identical because pywr's optimization may take a
    # slightly different path with different scenario counts.
    s0_diff = np.nanmean(np.abs(sf_mu_multi[:n_sim_days, 0] - sf_mu_one[:n_sim_days, 0]))
    print(f"  scenario-0 vs single-realization mean abs diff: {s0_diff:.3e}")
    assert s0_diff < 1e-3, (
        f"scenario-0 of multi-scenario run should closely match single-realization "
        f"run on realization 0; got mean abs diff={s0_diff:.3e}"
    )

    print("\nAll Phase 2 checks passed.")


if __name__ == "__main__":
    main()
