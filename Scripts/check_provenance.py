"""Check that every saved result was produced with the settings the scripts
now define. Reads the metadata bilby stores inside each *_result.json and
compares it against run_catalog.EVENTS / the sampler and prior settings.
Prints one line per check; MISMATCH lines need explanation before the
result can be quoted in the paper.

Note: this bilby version does not store per-detector frequency bounds or
segment duration in the result file, so those are not checkable here; the
waveform minimum_frequency (stored) is the proxy for f_min.

Usage: python scripts/check_provenance.py
"""
import os
import sys
import numpy as np
import bilby

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import run_catalog as rc

n_ok = n_bad = 0


def check(where, what, got, want, tol=0.0):
    global n_ok, n_bad
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    n_ok += ok; n_bad += (not ok)
    flag = "ok      " if ok else "MISMATCH"
    print(f"  {flag} {where:34s} {what:26s} got={got!r:<26} want={want!r}")


def prior_bounds(r, key):
    p = r.priors.get(key)
    return (p.minimum, p.maximum) if hasattr(p, "minimum") else None


def check_common(r, label, sampler_want):
    sk = r.sampler_kwargs
    for k, v in sampler_want.items():
        check(label, f"sampler.{k}", sk.get(k), v)
    return r.meta_data.get("likelihood", {}).get("waveform_arguments", {})


def snr_lines(ifos):
    for det, info in ifos.items():
        mf = info.get("matched_filter_SNR", 0)
        try:
            mf = abs(complex(mf))
        except Exception:
            mf = float("nan")
        print(f"           {det}: optimal SNR {info.get('optimal_SNR', float('nan')):.1f}, "
              f"matched-filter SNR {mf:.1f}")


# ---------------------------------------------------------------- catalog
print("=" * 100)
print("CATALOG RUNS (outputs/run_catalog_outputs)")
print("=" * 100)
CAT_SAMPLER = dict(nlive=1500, sample="rwalk", walks=50, bound="multi", dlogz=0.1)
for ev in rc.QUEUE:
    cfg = rc.EVENTS[ev]
    for model in ("gr", "kernel"):
        label = f"{ev}_{model}_v1"
        path = os.path.join(rc.OUTDIR, label, f"{label}_result.json")
        if not os.path.exists(path):
            print(f"  MISSING  {label}"); n_bad += 1; continue
        r = bilby.result.read_in_result(path)
        print(f"\n{label}")
        wa = check_common(r, label, CAT_SAMPLER)
        check(label, "approximant", wa.get("waveform_approximant"), cfg["approx"])
        check(label, "reference_frequency", wa.get("reference_frequency"), cfg["ref_freq"])
        check(label, "wf.minimum_frequency", wa.get("minimum_frequency"), cfg["f_min"])
        check(label, "prior.chirp_mass", prior_bounds(r, "chirp_mass"), tuple(map(float, cfg["chirp"])))
        check(label, "prior.mass_ratio", prior_bounds(r, "mass_ratio"), tuple(map(float, cfg["q"])))
        check(label, "prior.luminosity_distance", prior_bounds(r, "luminosity_distance"), tuple(map(float, cfg["dist"])))
        check(label, "prior.mass_1 (constraint)", prior_bounds(r, "mass_1"), tuple(map(float, cfg["comp"])))
        check(label, "prior.a_1", prior_bounds(r, "a_1"), (0.0, rc.SPIN_MAX))
        if model == "kernel":
            check(label, "prior.eps", prior_bounds(r, "eps"), (1e-25, 1e-19))
            check(label, "prior.f0_res", prior_bounds(r, "f0_res"), (20.0, 200.0))
            check(label, "prior.gamma_res", prior_bounds(r, "gamma_res"), (1.0, 50.0))
        ifos = r.meta_data.get("likelihood", {}).get("interferometers", {})
        print(f"           detectors: {sorted(ifos.keys())}")
        snr_lines(ifos)
        print(f"           lnZ={r.log_evidence:.2f}  lnBF(vs noise)={r.log_bayes_factor:.2f}  "
              f"samples={len(r.posterior)}")

# ---------------------------------------------------------------- validation
print("\n" + "=" * 100)
print("VALIDATION RUNS (outputs/validation_outputs)")
print("=" * 100)
VAL = os.path.join(rc.ROOT, "outputs", "validation_outputs")
INJ = dict(mass_1=150.0, mass_2=120.0, luminosity_distance=4000.0,
           theta_jn=0.8, psi=2.659, phase=1.3, ra=1.375, dec=-1.2108,
           geocent_time=1242442967.4)
runs = {
    "1a_fixed":   dict(sampler=dict(nlive=500, dlogz=0.1), inj=INJ, kernel=None),
    "1b_v1":      dict(sampler=dict(nlive=1000, sample="act-walk", walks=100, bound="live", dlogz=0.1),
                       inj=INJ, kernel=dict(eps=1e-22, f0_res=60.0, gamma_res=10.0)),
    "1b_loud_v3": dict(sampler=dict(nlive=1500, sample="rwalk", walks=50, bound="multi", dlogz=0.5),
                       inj=INJ, kernel=dict(eps=2.5e-21, f0_res=60.0, gamma_res=10.0)),
    "1c_v1":      dict(sampler=dict(nlive=1500, sample="rwalk", walks=50, bound="multi", dlogz=0.5),
                       inj=INJ, kernel="recover_only"),
}
for i in range(10):
    runs[f"1c_seed{i}"] = runs["1c_v1"]

for label, want in runs.items():
    path = os.path.join(VAL, label, f"{label}_result.json")
    if not os.path.exists(path):
        print(f"  MISSING  {label}"); n_bad += 1; continue
    r = bilby.result.read_in_result(path)
    print(f"\n{label}")
    wa = check_common(r, label, want["sampler"])
    check(label, "approximant", wa.get("waveform_approximant"), "IMRPhenomXPHM")
    check(label, "wf.minimum_frequency", wa.get("minimum_frequency"), 20.0)
    inj = r.injection_parameters or {}
    for k, v in want["inj"].items():
        check(label, f"inj.{k}", inj.get(k), v, tol=1e-6)
    if isinstance(want["kernel"], dict):
        for k, v in want["kernel"].items():
            check(label, f"inj.{k}", inj.get(k), v, tol=abs(v) * 1e-9)
    if want["kernel"] is not None:
        check(label, "prior.eps", prior_bounds(r, "eps"), (1e-25, 1e-19))
        check(label, "prior.f0_res", prior_bounds(r, "f0_res"), (20.0, 200.0))
        check(label, "prior.gamma_res", prior_bounds(r, "gamma_res"), (1.0, 50.0))
        check(label, "inj has kernel params", "eps" in inj, isinstance(want["kernel"], dict))
    check(label, "prior.chirp_mass", prior_bounds(r, "chirp_mass"), (80.0, 160.0))
    check(label, "prior.mass_ratio", prior_bounds(r, "mass_ratio"), (0.25, 1.0))
    check(label, "prior.luminosity_distance", prior_bounds(r, "luminosity_distance"), (1000.0, 10000.0))
    ifos = r.meta_data.get("likelihood", {}).get("interferometers", {})
    check(label, "detectors", sorted(ifos.keys()), ["H1", "L1"])
    snr_lines(ifos)
    print(f"           sampled: {r.search_parameter_keys}")

print("\n" + "=" * 100)
print(f"{n_ok} checks ok, {n_bad} mismatches/missing")