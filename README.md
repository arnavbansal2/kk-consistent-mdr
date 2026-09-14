# KK-Consistent MDR Project

Code and figures for "A Kramers–Kronig-consistent test of gravitational-wave
propagation: constraints from fifteen binary black hole mergers"
(Bansal & Farooq).

- `scripts/kernel.py` — the causal Lorentz susceptibility kernel
- `scripts/run_catalog.py` — the production runner (GR + kernel fit per event); the
  15-event registry states which events are in the paper and why others are not.
  Executed on the Caltech cluster; slurm logs in `outputs/run_catalog_outputs/logs/`
- `scripts/run_1a.py`, `run_1b.py`, `run_1c.py` — validation ladder
- `scripts/joint_fit.py` — importance-weighted shared-kernel combination
- `scripts/check_provenance.py` — verifies every saved result against the scripts' settings
- `scripts/results.py` — prints every number quoted in the paper from the outputs
- `figures/scripts/` — one script per paper figure; `figures/final/` — the PNGs

Posterior samples (`*_result.json`, ~1.1 GB) are not tracked here; they are archived
on Zenodo [DOI to be added]. With them placed under `outputs/`, every script runs
without resampling.
