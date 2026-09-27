"""Feature forensics for one event: tests applied to the strongest kernel
feature in a given f0 window.

    python scripts/forensics.py GW151226 --window 64 77 kk lens kink null split
    python scripts/forensics.py GW190814 --window 100 110 kk null

kk     amplitude-only / phase-only / full kernel gain at the cluster maximum,
       plus a phase-only f0 scan
lens   wave-optics point-mass lens fit (M_Lz, y grid)
kink   boson-cloud phase kink scan (f_res, dphi, dt)
null   event-local null: best-fit GR signal injected into adjacent off-source
       segments, kernel scanned across the window
split  per-detector kernel gain at the cluster maximum

Results are written to outputs/forensics/<EVENT>.json.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import bilby
from gwpy.timeseries import TimeSeries

warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from kernel import chi
from run_catalog import EVENTS, OUTDIR, ROOT, MPC_SEC, resolve_trigger, build_ifos

FOUT = os.path.join(ROOT, "outputs", "forensics")
os.makedirs(FOUT, exist_ok=True)

BINARY = ["mass_1", "mass_2", "luminosity_distance", "a_1", "a_2", "tilt_1", "tilt_2",
          "phi_12", "phi_jl", "theta_jn", "phase", "ra", "dec", "psi", "geocent_time"]
KERNEL = ["eps", "f0_res", "gamma_res"]


# ---------------------------------------------------------------- models
def make_model(mode):
    def model(fa, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12,
              a_2, tilt_2, phi_jl, theta_jn, phase, eps, f0_res, gamma_res, **kw):
        wf = bilby.gw.source.lal_binary_black_hole(
            fa, mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
            a_1=a_1, tilt_1=tilt_1, phi_12=phi_12, a_2=a_2, tilt_2=tilt_2,
            phi_jl=phi_jl, theta_jn=theta_jn, phase=phase, **kw)
        if wf is None:
            return None
        x = chi(fa, eps, f0_res, gamma_res)
        w = 2 * np.pi * fa * luminosity_distance * MPC_SEC
        P = {"full": np.exp(1j * w * x),
             "amp": np.exp(-w * np.imag(x)),
             "phase": np.exp(1j * w * np.real(x))}[mode]
        return {"plus": wf["plus"] * P, "cross": wf["cross"] * P}
    return model


_FC = {}


def lens_F(M, y, fc, wsw=15.0):
    import mpmath as mp
    GMsun_c3 = 4.925491e-6
    w = 8 * np.pi * GMsun_c3 * M * fc
    sq = np.sqrt(y * y + 4)
    mup, mum = .5 + (y * y + 2) / (2 * y * sq), .5 - (y * y + 2) / (2 * y * sq)
    dT = y * sq / 2 + np.log((sq + y) / (sq - y))
    xm = (y + sq) / 2
    ph = (xm - y) ** 2 / 2 - np.log(xm)
    F = np.empty(len(w), complex)
    for i, wi in enumerate(w):
        if wi < wsw:
            z = mp.mpc(0, wi / 2)
            F[i] = complex(mp.e ** (np.pi * wi / 4 + 1j * (wi / 2) * (mp.log(wi / 2) - 2 * ph))
                           * mp.gamma(1 - z) * mp.hyp1f1(z, 1, 1j * wi * y * y / 2))
        else:
            F[i] = np.sqrt(abs(mup)) - 1j * np.sqrt(abs(mum)) * np.exp(1j * wi * dT)
    return F / np.sqrt(abs(mup) + abs(mum))


def combo_model(fa, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12,
                a_2, tilt_2, phi_jl, theta_jn, phase, M_Lz, y_lens, f_res, dphi, dt_ms, **kw):
    from scipy.interpolate import interp1d
    wf = bilby.gw.source.lal_binary_black_hole(
        fa, mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
        a_1=a_1, tilt_1=tilt_1, phi_12=phi_12, a_2=a_2, tilt_2=tilt_2,
        phi_jl=phi_jl, theta_jn=theta_jn, phase=phase, **kw)
    if wf is None:
        return None
    h = {k: wf[k] for k in ("plus", "cross")}
    if M_Lz > 0:
        key = (round(float(M_Lz), 3), round(float(y_lens), 4))
        if key not in _FC:
            fc = np.linspace(20, 896, 120)
            F = lens_F(*key, fc)
            _FC[key] = (interp1d(fc, F.real, bounds_error=False, fill_value=1.0),
                        interp1d(fc, F.imag, bounds_error=False, fill_value=0.0))
        re, im = _FC[key]
        h = {k: v * (re(fa) + 1j * im(fa)) for k, v in h.items()}
    if f_res > 0:
        dPsi = np.where(fa > f_res, dphi + 2 * np.pi * fa * (dt_ms * 1e-3), 0.0)
        h = {k: v * np.exp(1j * dPsi) for k, v in h.items()}
    return h


# ---------------------------------------------------------------- setup
def wavegen(cfg, model):
    return bilby.gw.WaveformGenerator(
        duration=cfg["duration"], sampling_frequency=cfg["fs"],
        frequency_domain_source_model=model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant=cfg["approx"],
                                reference_frequency=cfg["ref_freq"],
                                minimum_frequency=cfg["f_min"]))


def likelihood(ifos, wg):
    return bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg)


def gain(like, params, null=None):
    like.parameters = dict(params)
    la = like.log_likelihood_ratio()
    like.parameters = dict(params, **(null or dict(eps=1e-25)))
    return la - like.log_likelihood_ratio()


def cluster_best(name, lo, hi):
    lab = f"{name}_kernel_v1"
    r = bilby.result.read_in_result(os.path.join(OUTDIR, lab, f"{lab}_result.json"))
    P = r.posterior
    cl = (P["f0_res"] > lo) & (P["f0_res"] < hi) & (P["eps"] > 1e-22)
    best = P[cl].iloc[np.argmax(P["log_likelihood"].values[cl])]
    Mc, q = best["chirp_mass"], best["mass_ratio"]
    m1 = Mc * (1 + q) ** 0.2 / q ** 0.6
    bp = dict(mass_1=float(m1), mass_2=float(q * m1),
              **{k: float(best[k]) for k in BINARY[2:] + KERNEL})
    d_sec = bp["luminosity_distance"] * MPC_SEC
    tau = bp["eps"] * (2 * np.pi * bp["f0_res"]) ** 2 * d_sec / (2 * np.pi * bp["gamma_res"])
    return bp, dict(cluster_fraction=float(cl.mean()), n_cluster=int(cl.sum()),
                    eps=bp["eps"], f0=bp["f0_res"], gamma=bp["gamma_res"], tau=float(tau),
                    eps_median_cluster=float(np.median(P["eps"][cl])))


def gr_best(name):
    lab = f"{name}_gr_v1"
    g = bilby.result.read_in_result(os.path.join(OUTDIR, lab, f"{lab}_result.json")).posterior
    gb = g.iloc[g["log_likelihood"].idxmax()]
    Mc, q = gb["chirp_mass"], gb["mass_ratio"]
    m1 = Mc * (1 + q) ** 0.2 / q ** 0.6
    return dict(mass_1=float(m1), mass_2=float(q * m1),
                **{k: float(gb[k]) for k in BINARY[2:]})


# ---------------------------------------------------------------- tests
def test_kk(cfg, ifos, bp, lo, hi):
    out = {}
    for mode in ("full", "amp", "phase"):
        out[mode] = float(gain(likelihood(ifos, wavegen(cfg, make_model(mode))), bp))
        print(f"  {mode:6s} {out[mode]:+.2f}")
    likeP = likelihood(ifos, wavegen(cfg, make_model("phase")))
    fg = np.arange(20.0, 200.01, 0.5)
    sc = np.array([gain(likeP, dict(bp, f0_res=float(f))) for f in fg])
    npk = int(((sc[1:-1] > sc[:-2]) & (sc[1:-1] > sc[2:]) & (sc[1:-1] > 2)).sum())
    out["phase_scan"] = dict(f0=fg.tolist(), dlnL=sc.tolist(),
                             peak=float(sc.max()), peak_f0=float(fg[np.argmax(sc)]),
                             n_peaks_above_2=npk)
    print(f"  phase-only scan: peak {sc.max():+.2f} at {fg[np.argmax(sc)]:.0f} Hz; peaks>+2: {npk}")
    return out


def test_lens(cfg, ifos, gp, comparator):
    like = likelihood(ifos, wavegen(cfg, combo_model))
    NEU = dict(M_Lz=0.0, y_lens=1.0, f_res=0.0, dphi=0.0, dt_ms=0.0)
    like.parameters = dict(gp, **NEU)
    L0 = like.log_likelihood_ratio()
    like.parameters = dict(gp, **{**NEU, "M_Lz": 1000.0, "y_lens": 0.05})
    gate = like.log_likelihood_ratio() - L0
    assert gate < -5, f"strong-lens gate failed ({gate:+.2f})"
    grid = {}
    for M in np.geomspace(20, 1000, 10):
        for y in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
            like.parameters = dict(gp, **{**NEU, "M_Lz": float(M), "y_lens": float(y)})
            grid[f"{M:.1f},{y}"] = float(like.log_likelihood_ratio() - L0)
    best = max(grid.values())
    print(f"  lens gate {gate:+.2f}; best lens {best:+.2f} (kernel full {comparator:+.2f})")
    return dict(gate=float(gate), best=best, grid=grid)


def test_kink(cfg, ifos, gp, comparator):
    like = likelihood(ifos, wavegen(cfg, combo_model))
    NEU = dict(M_Lz=0.0, y_lens=1.0, f_res=0.0, dphi=0.0, dt_ms=0.0)
    like.parameters = dict(gp, **NEU)
    L0 = like.log_likelihood_ratio()
    best, arg = -np.inf, None
    for fr in np.linspace(60, 160, 26):
        for dt in np.linspace(-5, 5, 17):
            for ph in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                like.parameters = dict(gp, **{**NEU, "f_res": float(fr), "dphi": float(ph),
                                              "dt_ms": float(dt)})
                v = like.log_likelihood_ratio() - L0
                if v > best:
                    best, arg = v, (float(fr), float(ph), float(dt))
    print(f"  best kink {best:+.2f} at f={arg[0]:.0f} Hz (phase-only comparator {comparator:+.2f})")
    return dict(best=float(best), f_res=arg[0], dphi=arg[1], dt_ms=arg[2])


def test_null(cfg, name, trigger, bp, lo, hi, cluster_eps, offsets):
    wg = wavegen(cfg, make_model("full"))
    inj = {k: bp[k] for k in BINARY[:-1]}
    peaks = {}
    for off in offsets:
        st = trigger + off + 2.0 - cfg["duration"]
        try:
            ifn = build_ifos(name, cfg, trigger + off)
        except AssertionError:
            continue
        ip = dict(inj, eps=1e-25, f0_res=float(bp["f0_res"]), gamma_res=float(bp["gamma_res"]),
                  geocent_time=st + cfg["duration"] - 2.0)
        ifn.inject_signal(waveform_generator=wg, parameters=ip)
        lk = likelihood(ifn, wg)
        lk.parameters = dict(ip)
        l0 = lk.log_likelihood_ratio()
        mx = -np.inf
        for fv in np.linspace(lo - 10, hi + 10, 36):
            lk.parameters = dict(ip, eps=float(cluster_eps), f0_res=float(fv))
            mx = max(mx, lk.log_likelihood_ratio() - l0)
        peaks[str(int(off))] = float(mx)
        print(f"  {off:+6.0f}s: {mx:+.2f}")
    v = np.array(list(peaks.values()))
    print(f"  null n={len(v)} mean {v.mean():+.2f} max {v.max():+.2f}")
    return dict(peaks=peaks, n=int(len(v)), mean=float(v.mean()), max=float(v.max()))


def test_split(cfg, ifos, bp):
    out = {}
    for ifo in ifos:
        one = bilby.gw.detector.InterferometerList([ifo])
        out[ifo.name] = float(gain(likelihood(one, wavegen(cfg, make_model("full"))), bp))
        print(f"  {ifo.name}: {out[ifo.name]:+.2f}")
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("tests", nargs="+", choices=["kk", "lens", "kink", "null", "split"])
    ap.add_argument("--window", nargs=2, type=float, required=True, metavar=("LO", "HI"))
    ap.add_argument("--offsets", nargs="*", type=float,
                    default=[o for o in range(-1200, 1201, 200) if abs(o) >= 200])
    args = ap.parse_args()

    name, cfg = args.event, EVENTS[args.event]
    lo, hi = args.window
    path = os.path.join(FOUT, f"{name}.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    R["window"] = [lo, hi]

    bp, info = cluster_best(name, lo, hi)
    R["cluster"] = info
    print(f"{name} cluster {lo:.0f}-{hi:.0f} Hz: eps={info['eps']:.2e} f0={info['f0']:.1f} "
          f"gamma={info['gamma']:.2f} tau={info['tau']:.2f} ({100*(1-np.exp(-info['tau'])):.0f}% notch)")

    trigger = resolve_trigger(name, cfg)
    ifos = build_ifos(name, cfg, trigger)
    gp = gr_best(name)

    if "kk" in args.tests:
        print("KK decomposition"); R["kk"] = test_kk(cfg, ifos, bp, lo, hi)
    full = R.get("kk", {}).get("full", float("nan"))
    phase = R.get("kk", {}).get("phase", float("nan"))
    if "lens" in args.tests:
        print("lens"); R["lens"] = test_lens(cfg, ifos, gp, full)
    if "kink" in args.tests:
        print("kink"); R["kink"] = test_kink(cfg, ifos, gp, phase)
    if "null" in args.tests:
        print("event-local null"); R["null"] = test_null(cfg, name, trigger, bp, lo, hi,
                                                        info["eps_median_cluster"], args.offsets)
    if "split" in args.tests:
        print("per-detector split"); R["split"] = test_split(cfg, ifos, bp)

    json.dump(R, open(path, "w"), indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()