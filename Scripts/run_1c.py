"""Validation gate 1c: pure-GR injection, kernel recovery (null test).

Same binary and conditioning as gate 1a (150 + 120 Msun, d_L = 4000 Mpc,
H1 + L1 design sensitivity, 4 s at 1024 Hz, 20-256 Hz, IMRPhenomXPHM,
fixed extrinsics). The injected signal is GR; the recovery model is the
kernel model with the catalog priors (eps log-uniform [1e-25, 1e-19],
f0 uniform [20, 200] Hz, Gamma log-uniform [1, 50] Hz).
Sampler: dynesty, nlive 1500, rwalk, walks 50, bound multi, dlogz 0.5.

Two modes:
    single  one realization, seed 88170237 -> 1c_v1/
    null    ten realizations, seeds 88170237..88170246 -> 1c_seed{0..9}/
            For each, the spurious kernel gain is max over 45 f0 bins of
            max logL minus the median over bins; the distribution of these
            gains calibrates the detection threshold used for the catalog.
            Note seed0 is the same realization as `single`.

Usage:
    python scripts/run_1c.py single
    python scripts/run_1c.py null

Output: outputs/validation_outputs/1c_v1/, 1c_seed{N}/, 1c_null_gains.npy.
Existing results are loaded without resampling.
"""
import os
import sys
import numpy as np
import bilby

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from kernel import chi

ROOT = os.path.dirname(SCRIPT_DIR)
OUTBASE = os.path.join(ROOT, "outputs", "validation_outputs")

MPC_SEC = 3.0857e22 / 2.998e8
duration, sampling_frequency = 4.0, 1024.0
f_min, f_max = 20.0, 256.0
SEEDS = [88170237, 88170238, 88170239, 88170240, 88170241,
         88170242, 88170243, 88170244, 88170245, 88170246]
SAMPLER = dict(nlive=1500, sample="rwalk", walks=50, bound="multi", dlogz=0.5)

injection = dict(
    mass_1=150.0, mass_2=120.0,
    a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0,
    luminosity_distance=4000.0,
    theta_jn=0.8, psi=2.659, phase=1.3,
    ra=1.375, dec=-1.2108,
    geocent_time=1242442967.4,
)
wf_args = dict(waveform_approximant="IMRPhenomXPHM",
               reference_frequency=20.0, minimum_frequency=f_min)


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


def make_priors():
    priors = bilby.gw.prior.BBHPriorDict()
    priors["mass_1"] = bilby.core.prior.Constraint(minimum=50, maximum=400)
    priors["mass_2"] = bilby.core.prior.Constraint(minimum=50, maximum=400)
    priors["chirp_mass"] = bilby.core.prior.Uniform(80, 160, name="chirp_mass")
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1.0, name="mass_ratio")
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        1000, 10000, name="luminosity_distance")
    priors["geocent_time"] = bilby.core.prior.Uniform(
        injection["geocent_time"] - 0.1, injection["geocent_time"] + 0.1)
    priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, boundary="periodic")
    for key in ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
                "theta_jn", "psi", "ra", "dec"]:
        priors[key] = injection[key]
    priors["eps"] = bilby.core.prior.LogUniform(1e-25, 1e-19, name="eps")
    priors["f0_res"] = bilby.core.prior.Uniform(20, 200, name="f0_res")
    priors["gamma_res"] = bilby.core.prior.LogUniform(1, 50, name="gamma_res")
    return priors


def run_or_load(label, seed):
    outdir = os.path.join(OUTBASE, label)
    path = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(path):
        print(f"[load] {path}")
        return bilby.result.read_in_result(path)

    np.random.seed(seed)
    gr_gen = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=wf_args)
    kern_gen = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=kernel_bbh,
        waveform_arguments=wf_args)

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = f_min, f_max
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection["geocent_time"] - duration + 2.0)
    ifos.inject_signal(waveform_generator=gr_gen, parameters=injection)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=kern_gen,
        priors=make_priors(), phase_marginalization=False,
        time_marginalization=False, distance_marginalization=False)
    return bilby.run_sampler(
        likelihood=likelihood, priors=make_priors(), sampler="dynesty",
        npool=4, injection_parameters=injection,
        outdir=outdir, label=label, **SAMPLER)


def spurious_gain(result):
    """Max over 45 f0 bins of max logL, minus the median over bins."""
    f0 = result.posterior["f0_res"].values
    logL = result.posterior["log_likelihood"].values
    edges = np.linspace(20, 200, 46)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f0 >= lo) & (f0 < hi)
        prof.append(logL[m].max() if m.sum() > 5 else np.nan)
    prof = np.array(prof)
    return float(np.nanmax(prof) - np.nanmedian(prof))


def gate_single(result):
    eps = result.posterior["eps"].values
    print("\n===== GATE 1c (single realization) =====")
    print(f"eps 95th percentile (upper limit): {np.percentile(eps, 95):.3e}")
    print(f"fraction of eps posterior below 1e-22: {(eps < 1e-22).mean():.3f} (prior 0.50)")
    print(f"fraction of eps posterior below 1e-23: {(eps < 1e-23).mean():.3f} (prior 0.33)")
    for k, v in [("chirp_mass", (150 * 120) ** 0.6 / 270 ** 0.2),
                 ("luminosity_distance", 4000.0)]:
        lo, hi = np.percentile(result.posterior[k], [5, 95])
        print(f"{k}: truth={v:.2f}, 90% CI=[{lo:.2f}, {hi:.2f}] "
              f"{'PASS' if lo <= v <= hi else 'FAIL'}")
    print(f"spurious kernel gain: {spurious_gain(result):.2f}")
    print(f"lnBF (kernel vs noise) = {result.log_bayes_factor:.2f}")


def gate_null():
    gains = []
    for i, seed in enumerate(SEEDS):
        r = run_or_load(f"1c_seed{i}", seed)
        g = spurious_gain(r)
        gains.append(g)
        print(f"  seed{i} ({seed}): gain = {g:.2f}   lnBF = {r.log_bayes_factor:.2f}")
    gains = np.array(gains)
    print("\n===== GATE 1c: null distribution (10 realizations) =====")
    print(f"gains:        {np.round(gains, 2).tolist()}")
    print(f"mean:         {gains.mean():.2f}   (Wilks expectation ~1.5)")
    print(f"max:          {gains.max():.2f}")
    print(f"95th pctile:  {np.percentile(gains, 95):.2f}")
    out = os.path.join(OUTBASE, "1c_null_gains.npy")
    if os.path.exists(out):
        saved = np.load(out)
        print(f"saved 1c_null_gains.npy agrees: "
              f"{np.allclose(np.sort(saved), np.sort(gains), atol=1e-6)}")
    else:
        np.save(out, gains)
        print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("single", "null"):
        sys.exit("usage: python scripts/run_1c.py single|null")
    if sys.argv[1] == "single":
        gate_single(run_or_load("1c_v1", SEEDS[0]))
    else:
        gate_null()