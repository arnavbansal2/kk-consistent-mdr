"""Causal-kernel catalog runner.

For each event: a GR run and a kernel run with identical data conditioning.
Kernel priors are identical across events, as required for the joint
shared-parameter fit.

Usage:
    python scripts/run_catalog.py --list
    python scripts/run_catalog.py GW170104 --npool 16
    python scripts/run_catalog.py all --npool 16
    python scripts/run_catalog.py GW170104 --dry      # data + checks only

Catalog (15 enabled): GW151012, GW151226, GW170104, GW170608, GW170809,
GW170814, GW170818, GW170823, GW190408_181802, GW190512_180714,
GW190630_185205, GW190814, GW190828_063405, GW200224_222234, GW200311_115853.

Disabled events:
    GW190412, GW191204_171526, GW191216_213338  not completed (walltime)
    GW200129_065458  glitch-mitigated in LVK analyses (GWTC-3 Table XIV)
    GW191109_010717  data-quality issues in both LIGO detectors (GWTC-3)
    GW190701_203306  scattered-light treatment around the event
    GW170817, GW190425  BNS; tidal waveform campaign

Output: outputs/run_catalog_outputs/<label>/ per stage and
outputs/run_catalog_outputs/catalog_summary.json, one record per event.
Existing results are loaded rather than recomputed.
"""
import argparse
import json
import os
import sys

import numpy as np
import bilby
from gwpy.timeseries import TimeSeries

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

try:
    from gwosc.datasets import event_gps
    HAVE_GWOSC = True
except Exception:
    HAVE_GWOSC = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from kernel import chi

ROOT = os.path.dirname(SCRIPT_DIR)
OUTDIR = os.path.join(ROOT, "outputs", "run_catalog_outputs")
SUMMARY_PATH = os.path.join(OUTDIR, "catalog_summary.json")

MPC_SEC = 3.0857e22 / 2.998e8
THRESHOLD = 3.4
WINDOWS = {"123Hz": (115.0, 131.0), "68Hz": (66.0, 77.0)}
SPIN_MAX = 0.8

# Trigger times are resolved from the gwosc package; trigger_fallback is
# used only if that lookup fails.
EVENTS = {
    # O1
    "GW151012": dict(trigger_fallback=1128678900.4, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(12, 26), q=(0.25, 1.0), comp=(6, 60),
        dist=(200, 3000), test_m=(23, 13), test_dL=1000,
        pub="Mc_det~18, D~1 Gpc, SNR~10", enabled=True),
    "GW151226": dict(trigger_fallback=1135136350.6, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXP", chirp=(7, 13), q=(0.25, 1.0), comp=(3, 35),
        dist=(50, 1500), test_m=(14, 8), test_dL=440,
        pub="Mc_det~9.7, D~440, SNR~13, long inspiral", enabled=True),
    # O2
    "GW170104": dict(trigger_fallback=1167559936.6, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=512.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(15, 38), q=(0.3, 1.0), comp=(8, 80),
        dist=(200, 3000), test_m=(31, 20), test_dL=990,
        pub="Mc_det~25, D~990, SNR~13", enabled=True),
    "GW170608": dict(trigger_fallback=1180922494.5, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        fmin_override={"H1": 30.0},
        approx="IMRPhenomXP", chirp=(7, 12), q=(0.25, 1.0), comp=(3, 30),
        dist=(50, 1000), test_m=(12, 7), test_dL=320,
        pub="Mc_det~8.5, D~320, SNR~14, long inspiral; H1 f_min=30 Hz per GWTC-1",
        enabled=True),
    "GW170809": dict(trigger_fallback=1186302519.7, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=512.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(18, 45), q=(0.3, 1.0), comp=(10, 90),
        dist=(200, 3000), test_m=(35, 24), test_dL=1030,
        pub="Mc_det~30, D~1 Gpc, SNR~12", enabled=True),
    "GW170814": dict(trigger_fallback=1186741861.5, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=512.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(16, 40), q=(0.3, 1.0), comp=(10, 80),
        dist=(100, 1500), test_m=(31, 25), test_dL=540,
        pub="Mc_det~27, D~540, SNR~17", enabled=True),
    "GW170818": dict(trigger_fallback=1187058327.1, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=512.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(20, 48), q=(0.3, 1.0), comp=(12, 90),
        dist=(200, 3000), test_m=(35, 27), test_dL=1060,
        pub="Mc_det~32, D~1 Gpc, SNR~11", enabled=True),
    "GW170823": dict(trigger_fallback=1187529256.5, duration=8.0, fs=2048.0,
        f_min=20.0, f_max=512.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(22, 60), q=(0.3, 1.0), comp=(12, 120),
        dist=(400, 5000), test_m=(40, 30), test_dL=1850,
        pub="Mc_det~39, D~1.9 Gpc, SNR~12", enabled=True),
    # O3a
    "GW190408_181802": dict(trigger_fallback=1238782700.3, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(14, 34), q=(0.3, 1.0), comp=(8, 70),
        dist=(300, 4000), test_m=(25, 18), test_dL=1550,
        pub="Mc_det~24, D~1.5 Gpc, SNR~15", enabled=True),
    "GW190412": dict(trigger_fallback=1239082262.2, duration=16.0, fs=2048.0,
        f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM",
        chirp=(10, 25), q=(0.1, 0.6), comp=(5, 60),
        dist=(150, 2000), test_m=(34, 9), test_dL=740,
        pub="Mc_det~15, q~0.28, D~740, SNR~19, higher modes",
        enabled=False, note="not completed: cluster walltime limit"),
    "GW190512_180714": dict(trigger_fallback=1241719652.4, duration=16.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(11, 28), q=(0.2, 1.0), comp=(5, 60),
        dist=(300, 4000), test_m=(25, 13), test_dL=1400,
        pub="Mc_det~19, D~1.4 Gpc, SNR~13", enabled=True),
    "GW190630_185205": dict(trigger_fallback=1245955943.2, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(18, 44), q=(0.3, 1.0), comp=(10, 80),
        dist=(200, 3000), test_m=(35, 24), test_dL=890,
        pub="Mc_det~29, D~890, SNR~15, two-detector", enabled=True),
    "GW190814": dict(trigger_fallback=1249852257.0, duration=16.0, fs=2048.0,
        f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXP", chirp=(5.8, 7.0), q=(0.05, 0.25), comp=(1, 40),
        dist=(50, 800), test_m=(23, 2.6), test_dL=240,
        pub="Mc_det~6.4, q~0.11, D~240, SNR~25, extreme mass ratio",
        enabled=True),
    "GW190828_063405": dict(trigger_fallback=1251009263.8, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(20, 52), q=(0.3, 1.0), comp=(12, 100),
        dist=(500, 6000), test_m=(32, 26), test_dL=2100,
        pub="Mc_det~34, D~2.1 Gpc, SNR~16", enabled=True),
    # O3b
    "GW191204_171526": dict(trigger_fallback=1259514944.1, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXP",
        chirp=(7, 13), q=(0.25, 1.0), comp=(3, 35),
        dist=(50, 1500), test_m=(12, 8), test_dL=650,
        pub="Mc_det~9, D~650, SNR~17, low mass",
        enabled=False, note="not completed: cluster walltime limit"),
    "GW191216_213338": dict(trigger_fallback=1260567236.4, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXP",
        chirp=(7, 13), q=(0.25, 1.0), comp=(3, 35),
        dist=(50, 1200), test_m=(12, 8), test_dL=340,
        pub="Mc_det~9, D~340, SNR~18, low mass",
        enabled=False, note="not completed: cluster walltime limit"),
    "GW200224_222234": dict(trigger_fallback=1266618172.4, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(24, 62), q=(0.3, 1.0), comp=(15, 120),
        dist=(400, 5000), test_m=(40, 33), test_dL=1700,
        pub="Mc_det~41, D~1.7 Gpc, SNR~20", enabled=True),
    "GW200311_115853": dict(trigger_fallback=1267963151.4, duration=8.0,
        fs=2048.0, f_min=20.0, f_max=896.0, psd_dur=128.0, ref_freq=20.0,
        approx="IMRPhenomXPHM", chirp=(20, 52), q=(0.3, 1.0), comp=(12, 100),
        dist=(300, 4000), test_m=(34, 28), test_dL=1200,
        pub="Mc_det~33, D~1.2 Gpc, SNR~18", enabled=True),
    # disabled
    "GW200129_065458": dict(enabled=False,
        note="glitch-mitigated event (GWTC-3 Table XIV)"),
    "GW191109_010717": dict(enabled=False,
        note="data-quality issues in both LIGO detectors (GWTC-3)"),
    "GW190701_203306": dict(enabled=False,
        note="scattered-light treatment around event"),
    "GW170817": dict(enabled=False, note="BNS: tidal campaign"),
    "GW190425": dict(enabled=False, note="BNS: tidal campaign"),
}
QUEUE = [e for e, c in EVENTS.items() if c.get("enabled")]


def kernel_bbh(frequency_array, mass_1, mass_2, luminosity_distance,
               a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
               theta_jn, phase, eps, f0_res, gamma_res, **kwargs):
    wf = bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        a_1=a_1, tilt_1=tilt_1, phi_12=phi_12, a_2=a_2, tilt_2=tilt_2,
        phi_jl=phi_jl, theta_jn=theta_jn, phase=phase, **kwargs)
    if wf is None:
        return None
    d_sec = luminosity_distance * MPC_SEC
    w = 2.0 * np.pi * frequency_array
    P = np.exp(1j * w * d_sec * chi(frequency_array, eps, f0_res, gamma_res))
    return {"plus": wf["plus"] * P, "cross": wf["cross"] * P}


def resolve_trigger(name, cfg):
    if HAVE_GWOSC:
        try:
            t = float(event_gps(name))
            print(f"[gps] {name}: {t} (gwosc)")
            return t
        except Exception as e:
            print(f"[gps] {name}: gwosc lookup failed ({str(e)[:40]}); "
                  f"using fallback {cfg['trigger_fallback']}")
    else:
        print(f"[gps] gwosc package unavailable; using fallback "
              f"{cfg['trigger_fallback']}")
    return cfg["trigger_fallback"]


def build_ifos(name, cfg, trigger):
    """Fetch H1/L1/V1, keep detectors with clean data, estimate PSDs."""
    start = trigger + 2.0 - cfg["duration"]
    psd_start = start - cfg["psd_dur"] - 2.0
    ifos = bilby.gw.detector.InterferometerList([])
    for det in ["H1", "L1", "V1"]:
        try:
            data = TimeSeries.fetch_open_data(
                det, start, start + cfg["duration"], cache=True)
        except Exception:
            print(f"[data] {det}: no data at this time; dropped")
            continue
        data = data.resample(cfg["fs"])
        if np.isnan(data.value).any():
            print(f"[data] {det}: NaN in analysis segment; dropped")
            continue
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.set_strain_data_from_gwpy_timeseries(data)

        psd_data = TimeSeries.fetch_open_data(
            det, psd_start, psd_start + cfg["psd_dur"], cache=True)
        psd_data = psd_data.resample(cfg["fs"])
        v = psd_data.value
        if np.isnan(v).any():
            good = ~np.isnan(v)
            print(f"[psd] {det}: segment {100*(1-good.mean()):.0f}% NaN; "
                  f"cropping to longest clean run")
            idx = np.where(good)[0]
            sp = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            b = max(sp, key=len)
            need = int(4 * cfg["duration"] * cfg["fs"])
            if len(b) < need:
                print(f"[psd] {det}: clean stretch too short; dropped")
                continue
            psd_data = psd_data[b[0]:b[-1] + 1]
        alpha = 2 * ifo.strain_data.roll_off / cfg["duration"]
        psd = psd_data.psd(fftlength=cfg["duration"], overlap=0,
                           window=("tukey", alpha), method="median")
        assert not np.isnan(psd.value).any() and (psd.value > 0).all(), \
            f"{det}: bad PSD"
        print(f"[psd] {det}: OK ({len(psd_data)/cfg['fs']:.0f} s, "
              f"min={psd.value.min():.2e})")
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value)
        fmin = cfg.get("fmin_override", {}).get(det, cfg["f_min"])
        ifo.minimum_frequency, ifo.maximum_frequency = fmin, cfg["f_max"]
        ifos.append(ifo)
    assert len(ifos) >= 2, f"{name}: fewer than 2 clean detectors"
    print(f"[net] {name}: using {[i.name for i in ifos]}")
    return ifos


def base_priors(cfg, trigger):
    p = bilby.gw.prior.BBHPriorDict()
    p["mass_1"] = bilby.core.prior.Constraint(
        minimum=cfg["comp"][0], maximum=cfg["comp"][1])
    p["mass_2"] = bilby.core.prior.Constraint(
        minimum=cfg["comp"][0], maximum=cfg["comp"][1])
    p["chirp_mass"] = bilby.core.prior.Uniform(*cfg["chirp"], name="chirp_mass")
    p["mass_ratio"] = bilby.core.prior.Uniform(*cfg["q"], name="mass_ratio")
    p["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        *cfg["dist"], name="luminosity_distance")
    p["geocent_time"] = bilby.core.prior.Uniform(
        trigger - 0.1, trigger + 0.1, name="geocent_time")
    p["a_1"] = bilby.core.prior.Uniform(0, SPIN_MAX, name="a_1")
    p["a_2"] = bilby.core.prior.Uniform(0, SPIN_MAX, name="a_2")
    return p


def add_kernel_priors(p):
    # Identical across events; required for the joint fit.
    p["eps"] = bilby.core.prior.LogUniform(1e-25, 1e-19, name="eps")
    p["f0_res"] = bilby.core.prior.Uniform(20, 200, name="f0_res")
    p["gamma_res"] = bilby.core.prior.LogUniform(1, 50, name="gamma_res")
    return p


def run_or_load(cfg, ifos, source_model, priors, label, is_kernel,
                npool, dry=False):
    outdir = os.path.join(OUTDIR, label)
    path = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(path):
        print(f"[skip] {label}: result exists, loading.")
        return bilby.result.read_in_result(path)

    wg = bilby.gw.WaveformGenerator(
        duration=cfg["duration"], sampling_frequency=cfg["fs"],
        frequency_domain_source_model=source_model,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters),
        waveform_arguments=dict(waveform_approximant=cfg["approx"],
                                reference_frequency=cfg["ref_freq"],
                                minimum_frequency=cfg["f_min"]))

    print(f"[priors] {label}:")
    for k in sorted(priors):
        print(f"     {k}: {priors[k]}")

    if is_kernel:
        fa = wg.frequency_array
        m1t, m2t = cfg["test_m"]
        test = dict(mass_1=m1t, mass_2=m2t, luminosity_distance=cfg["test_dL"],
                    a_1=0, a_2=0, tilt_1=0, tilt_2=0, phi_12=0, phi_jl=0,
                    theta_jn=0.9, phase=1.3,
                    eps=1e-21, f0_res=60.0, gamma_res=10.0)
        t0 = dict(test); t0["eps"] = 0.0
        i0 = int(np.argmin(np.abs(fa - 60.0)))
        r = (np.abs(wg.frequency_domain_strain(test)["plus"][i0]) /
             np.abs(wg.frequency_domain_strain(t0)["plus"][i0]))
        print(f"[check] {label}: kernel damping at 60 Hz = {r:.4f} (must be < 1)")
        assert r < 1.0, "Kernel amplifies: sign error"

    if dry:
        print(f"[dry] {label}: checks passed; sampling skipped.")
        return None

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wg, priors=priors,
        phase_marginalization=False, time_marginalization=False,
        distance_marginalization=False)
    return bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler="dynesty",
        nlive=1500, sample="rwalk", walks=50, bound="multi",
        dlogz=0.1, npool=npool, outdir=outdir, label=label)


def run_event(name, npool, dry=False):
    cfg = EVENTS[name]
    if not cfg.get("enabled"):
        print(f"### {name} DISABLED: {cfg.get('note','')}"); return None
    print("\n" + "#" * 66 + f"\n# {name}   [{cfg.get('pub','')}]\n" + "#" * 66)

    trigger = resolve_trigger(name, cfg)
    ifos = build_ifos(name, cfg, trigger)
    gl, kl = f"{name}_gr_v1", f"{name}_kernel_v1"

    gr = run_or_load(cfg, ifos, bilby.gw.source.lal_binary_black_hole,
                     base_priors(cfg, trigger), gl, False, npool, dry)
    kern = run_or_load(cfg, ifos, kernel_bbh,
                       add_kernel_priors(base_priors(cfg, trigger)),
                       kl, True, npool, dry)
    if dry or gr is None or kern is None:
        return None

    p = kern.posterior
    eps, f0, gam = p["eps"].values, p["f0_res"].values, p["gamma_res"].values
    logL = p["log_likelihood"].values
    lnBF = kern.log_evidence - gr.log_evidence
    m40 = f0 > 40

    out = dict(event=name,
        gr_lnZ=float(gr.log_evidence), gr_lnZ_err=float(gr.log_evidence_err),
        gr_lnBF_vs_noise=float(gr.log_bayes_factor),
        kern_lnZ=float(kern.log_evidence),
        kern_lnZ_err=float(kern.log_evidence_err),
        lnBF_kernel_vs_gr=float(lnBF),
        lnBF_err=float(np.hypot(kern.log_evidence_err, gr.log_evidence_err)),
        verdict=("CANDIDATE" if lnBF > THRESHOLD else "NULL"),
        eps_UL_marginal=float(np.percentile(eps, 95)),
        eps_UL_f0gt40=(float(np.percentile(eps[m40], 95))
                       if m40.sum() > 50 else None),
        eps_over_gamma_UL=float(np.percentile(eps / gam, 95)),
        frac_eps_below_1e22=float((eps < 1e-22).mean()),
        corr_eps_a1=float(np.corrcoef(np.log10(eps), p["a_1"])[0, 1]),
        corr_eps_a2=float(np.corrcoef(np.log10(eps), p["a_2"])[0, 1]),
        distance_median_Mpc=float(np.median(p["luminosity_distance"])),
        detectors=[i.name for i in ifos], approx=cfg["approx"])
    for tag, (lo, hi) in WINDOWS.items():
        w = (f0 > lo) & (f0 < hi)
        out[f"n_{tag}_window"] = int(w.sum())
        out[f"dlogL_{tag}_window"] = (float(logL[w].max() - logL[~w].max())
                                      if w.sum() > 5 else None)

    print("\n" + "=" * 66)
    print(f"{name}: lnBF = {out['lnBF_kernel_vs_gr']:+.2f} +/- "
          f"{out['lnBF_err']:.2f} -> {out['verdict']}")
    print(f"eps UL: marginal {out['eps_UL_marginal']:.2e}"
          + (f" | f0>40: {out['eps_UL_f0gt40']:.2e}"
             if out['eps_UL_f0gt40'] else ""))
    for tag in WINDOWS:
        if out[f"dlogL_{tag}_window"] is not None:
            print(f"{tag} window advantage: {out[f'dlogL_{tag}_window']:+.2f}")

    summary = json.load(open(SUMMARY_PATH)) if os.path.exists(SUMMARY_PATH) else []
    summary = [s for s in summary if s["event"] != name] + [out]
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
    print(f"[saved -> {SUMMARY_PATH}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="*", help="event names, or 'all'")
    ap.add_argument(
        "--npool", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="worker processes (defaults to SLURM allocation or CPU count)")
    ap.add_argument("--dry", action="store_true",
                    help="data + checks only, no sampling")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.events:
        print(f"{'event':<20}{'status':<10}{'done?':<8}note/pub")
        for e, c in EVENTS.items():
            done = os.path.exists(
                os.path.join(OUTDIR, f"{e}_kernel_v1", f"{e}_kernel_v1_result.json"))
            print(f"{e:<20}{'ENABLED' if c.get('enabled') else 'disabled':<10}"
                  f"{'yes' if done else '':<8}{c.get('pub', c.get('note',''))}")
        return

    names = QUEUE if args.events == ["all"] else args.events
    bad = [n for n in names if n not in EVENTS]
    if bad:
        sys.exit(f"unknown events: {bad}")
    results = [r for n in names if (r := run_event(n, args.npool, args.dry))]

    if len(results) > 1:
        print("\n" + "=" * 80)
        print(f"{'event':<20}{'D[Mpc]':>8}{'lnBF':>8}{'verdict':>11}"
              f"{'eps UL(f0>40)':>15}{'123Hz':>7}{'68Hz':>7}")
        print("=" * 80)
        for r in results:
            ul = r["eps_UL_f0gt40"] or r["eps_UL_marginal"]
            w1 = (f"{r['dlogL_123Hz_window']:+.1f}"
                  if r["dlogL_123Hz_window"] is not None else "--")
            w2 = (f"{r['dlogL_68Hz_window']:+.1f}"
                  if r["dlogL_68Hz_window"] is not None else "--")
            print(f"{r['event']:<20}{r['distance_median_Mpc']:>8.0f}"
                  f"{r['lnBF_kernel_vs_gr']:>+8.2f}{r['verdict']:>11}"
                  f"{ul:>15.2e}{w1:>7}{w2:>7}")


if __name__ == "__main__":
    main()