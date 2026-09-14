"""Joint shared-kernel fit (importance-weighted, KDE-free).

Method:
  Sobol N_THETA kernel points x K_NUIS GR-posterior nuisance draws per event;
  log wbar_i(theta) = logsumexp_k(dlnL_ik) - log K_NUIS,
  dlnL_ik = lnL(theta, draw_k) - lnL(eps=0, draw_k) with one likelihood object
  (eps=0 is exactly the GR waveform). Flat priors in the sampled coordinates,
  so the joint posterior weight is log w(theta) = sum_i log wbar_i(theta).

This file is a reconstruction of the original joint-fit code, validated
against the surviving arrays (UL agreement within K=20 draw noise). The
per-event checkpoints outputs/joint_fit_outputs/joint_dlnL_<event>_val.npy
were produced by this code; joint_theta.npy / joint_logw.npy are the
surviving 17-event arrays of the original run and are not used for the
15-event result.

Usage:
    python scripts/joint_fit.py --validate
        reuse joint_theta.npy as the grid, K=20, tag "val"; existing
        checkpoints are loaded, missing ones computed; sums over the catalog
        events and writes joint_theta_val.npy, joint_logw_val.npy
    python scripts/joint_fit.py --validate --limit-theta 200   (smoke test)
    python scripts/joint_fit.py --k 40 --n 3000                (production)
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import qmc
from scipy.special import logsumexp
import bilby

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from run_catalog import EVENTS, QUEUE, OUTDIR, ROOT, resolve_trigger, build_ifos, kernel_bbh

JOINT = os.path.join(ROOT, "outputs", "joint_fit_outputs")
os.makedirs(JOINT, exist_ok=True)

# identical to run_catalog.add_kernel_priors (joint-fit requirement)
EPS_LO, EPS_HI = 1e-25, 1e-19
F0_LO, F0_HI = 20.0, 200.0
GAM_LO, GAM_HI = 1.0, 50.0
SOBOL_SEED = 0
MASK_LO, MASK_HI = 98.0, 114.0

BINARY_KEYS = ["mass_1", "mass_2", "luminosity_distance",
               "a_1", "tilt_1", "phi_12", "a_2", "tilt_2", "phi_jl",
               "theta_jn", "phase", "ra", "dec", "psi", "geocent_time"]


def gr_result_path(ev):
    return os.path.join(OUTDIR, f"{ev}_gr_v1", f"{ev}_gr_v1_result.json")


def sobol_theta(n):
    s = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    u = s.random(n)
    eps = 10 ** (np.log10(EPS_LO) + u[:, 0] * (np.log10(EPS_HI) - np.log10(EPS_LO)))
    f0 = F0_LO + u[:, 1] * (F0_HI - F0_LO)
    gam = 10 ** (np.log10(GAM_LO) + u[:, 2] * (np.log10(GAM_HI) - np.log10(GAM_LO)))
    return np.column_stack([eps, f0, gam])


def build_likelihood(name, cfg):
    trigger = resolve_trigger(name, cfg)
    ifos = build_ifos(name, cfg, trigger)
    wg = bilby.gw.WaveformGenerator(
        duration=cfg["duration"], sampling_frequency=cfg["fs"],
        frequency_domain_source_model=kernel_bbh,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters),
        waveform_arguments=dict(waveform_approximant=cfg["approx"],
                                reference_frequency=cfg["ref_freq"],
                                minimum_frequency=cfg["f_min"]))
    return bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wg,
        phase_marginalization=False, time_marginalization=False,
        distance_marginalization=False)


def sanity_damping(like):
    """At a fiducial point the kernel must damp, never amplify."""
    p = dict(mass_1=30.0, mass_2=25.0, luminosity_distance=1000.0,
             a_1=0.0, tilt_1=0.0, phi_12=0.0, a_2=0.0, tilt_2=0.0,
             phi_jl=0.0, theta_jn=0.9, phase=1.3, ra=1.0, dec=0.5, psi=0.3,
             geocent_time=float(like.interferometers[0].strain_data.start_time) + 6.0)
    wg = like.waveform_generator
    fa = wg.frequency_array
    on = dict(p, eps=1e-21, f0_res=60.0, gamma_res=10.0)
    off = dict(p, eps=0.0, f0_res=60.0, gamma_res=10.0)
    i0 = int(np.argmin(np.abs(fa - 60.0)))
    r = (np.abs(wg.frequency_domain_strain(on)["plus"][i0]) /
         np.abs(wg.frequency_domain_strain(off)["plus"][i0]))
    assert r < 1.0, f"kernel amplifies (ratio {r:.4f}): sign error"
    print(f"  damping at 60 Hz fiducial: {r:.4f} (<1, OK)")


def per_event_dlnl(name, theta, k_nuis, seed, tag):
    """dlnL matrix (N_theta x K) for one event, checkpointed to disk."""
    out = os.path.join(JOINT, f"joint_dlnL_{name}_{tag}.npy")
    if os.path.exists(out):
        d = np.load(out)
        if d.shape == (len(theta), k_nuis):
            print(f"  [skip] {name}: checkpoint exists")
            return d
        print(f"  [stale] {name}: checkpoint shape {d.shape} != ({len(theta)},{k_nuis}); recomputing")

    cfg = EVENTS[name]
    like = build_likelihood(name, cfg)
    sanity_damping(like)

    post = bilby.result.read_in_result(gr_result_path(name)).posterior
    if "mass_1" not in post.columns:
        Mc, q = post["chirp_mass"], post["mass_ratio"]
        post["mass_1"] = Mc * (1 + q) ** 0.2 / q ** 0.6
        post["mass_2"] = post["mass_1"] * q
    missing = [k for k in BINARY_KEYS if k not in post.columns]
    assert not missing, f"{name}: GR posterior missing columns {missing}"
    rng = np.random.default_rng(abs(hash((name, seed))) % (2 ** 32))
    rows = post.iloc[rng.choice(len(post), k_nuis, replace=False)]

    dlnl = np.empty((len(theta), k_nuis))
    t_start = time.time()
    for k, (_, row) in enumerate(rows.iterrows()):
        binary = {key: float(row[key]) for key in BINARY_KEYS}
        like.parameters = dict(binary, eps=0.0, f0_res=60.0, gamma_res=10.0)
        lnl_gr = like.log_likelihood_ratio()
        for t in range(len(theta)):
            like.parameters = dict(binary, eps=float(theta[t, 0]),
                                   f0_res=float(theta[t, 1]), gamma_res=float(theta[t, 2]))
            dlnl[t, k] = like.log_likelihood_ratio() - lnl_gr
        el = time.time() - t_start
        print(f"  {name}: draw {k+1}/{k_nuis} done ({el/60:.1f} min elapsed, "
              f"~{el/(k+1)*(k_nuis-k-1)/60:.0f} min left)")
    np.save(out, dlnl)
    print(f"  [saved] {out}")
    return dlnl


def report(theta, logw, n_events, label):
    w = np.exp(logw - logw.max())
    ess = w.sum() ** 2 / (w * w).sum()

    def wq(v, ww):
        o = np.argsort(v)
        cc = np.cumsum(ww[o]) / ww[o].sum()
        return v[o][np.searchsorted(cc, 0.95)]

    m40 = theta[:, 1] > 40
    ul = wq(theta[m40, 0], w[m40])
    mmask = m40 & ~((theta[:, 1] > MASK_LO) & (theta[:, 1] < MASK_HI))
    ulg = wq(theta[mmask, 0] / theta[mmask, 2], w[mmask])
    print("\n" + "=" * 60)
    print(f"JOINT [{label}] ({n_events} events): ESS = {ess:.0f}/{len(theta)}")
    print(f"  eps       < {ul:.3e}   (f0>40, 95%)")
    print(f"  eps/Gamma < {ulg:.3e}   (f0>40, {MASK_LO:.0f}-{MASK_HI:.0f} Hz masked)")
    return ul, ess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40, help="nuisance draws per event")
    ap.add_argument("--n", type=int, default=3000, help="Sobol theta points")
    ap.add_argument("--validate", action="store_true",
                    help="reuse joint_theta.npy grid at K=20 (tag 'val')")
    ap.add_argument("--limit-theta", type=int, default=None,
                    help="smoke test: first N theta points only")
    ap.add_argument("--seed", type=int, default=1, help="nuisance-draw seed")
    ap.add_argument("--events", nargs="*", default=None, help="restrict to these events")
    args = ap.parse_args()

    events = [e for e in QUEUE if os.path.exists(gr_result_path(e))]
    if args.events:
        events = [e for e in events if e in args.events]
    assert events, "no events with GR results found"
    print(f"{len(events)} events: {events}\n")

    if args.validate:
        th_path = os.path.join(JOINT, "joint_theta.npy")
        assert os.path.exists(th_path), "validation needs joint_theta.npy in outputs/joint_fit_outputs"
        theta = np.load(th_path)
        k_nuis, tag = 20, "val"
        if args.limit_theta:
            theta = theta[:args.limit_theta]
            tag = f"val{args.limit_theta}"
        print(f"VALIDATION GRID: {len(theta)} theta points, K={k_nuis}")
    else:
        theta = sobol_theta(args.n)
        k_nuis, tag = args.k, f"K{args.k}_N{args.n}"
        print(f"PRODUCTION: N={len(theta)} Sobol (seed {SOBOL_SEED}), K={k_nuis}")

    logw = np.zeros(len(theta))
    for i, ev in enumerate(events):
        print(f"\n[{i+1}/{len(events)}] {ev}")
        d = per_event_dlnl(ev, theta, k_nuis, args.seed, tag)
        logw += logsumexp(d, axis=1) - np.log(k_nuis)

    report(theta, logw, len(events), tag)
    np.save(os.path.join(JOINT, f"joint_theta_{tag}.npy"), theta)
    np.save(os.path.join(JOINT, f"joint_logw_{tag}.npy"), logw)
    print(f"\nsaved: joint_theta_{tag}.npy, joint_logw_{tag}.npy in {JOINT}")


if __name__ == "__main__":
    main()