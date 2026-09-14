"""Validation gate 1b: kernel injection, kernel recovery, fixed extrinsics.

Same binary and conditioning as gate 1a (150 + 120 Msun, d_L = 4000 Mpc,
H1 + L1 design sensitivity, 4 s at 1024 Hz, 20-256 Hz, IMRPhenomXPHM),
with the causal kernel applied at f0 = 60 Hz, Gamma = 10 Hz.

Two stages:
    marginal  eps = 1e-22   (tau ~ 0.09; below detection threshold)
              sampler: bilby dynesty defaults (nlive 1000, act-walk,
              walks 100, bound live, dlogz 0.1)
    loud      eps = 2.5e-21 (tau ~ 2.3; ~90% absorption notch)
              sampler: nlive 1500, rwalk, walks 50, bound multi, dlogz 0.5
              (adopted after the defaults were found to stall on a sharp
              resonance; used for all subsequent analyses)

Kernel priors (identical to the catalog): eps log-uniform [1e-25, 1e-19],
f0 uniform [20, 200] Hz, Gamma log-uniform [1, 50] Hz.

Pass criteria:
    marginal  kernel parameters consistent with the null; no localized f0.
    loud      f0 recovered near 60 Hz; binary parameters within 90% CI.
              (eps and Gamma are individually unconstrained: Gamma rails
              to the 1 Hz prior floor.)

Usage:
    python scripts/run_1b.py marginal
    python scripts/run_1b.py loud

Output: outputs/validation_outputs/1b_v1/ (marginal), 1b_loud_v3/ (loud).
Existing results are loaded and the gate table printed without resampling.
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

STAGES = {
    "marginal": dict(label="1b_v1", eps=1e-22, seed=88170236,
                     sampler=dict(nlive=1000, sample="act-walk", walks=100,
                                  bound="live", dlogz=0.1)),
    "loud": dict(label="1b_loud_v3", eps=2.5e-21, seed=88170236,
                 sampler=dict(nlive=1500, sample="rwalk", walks=50,
                              bound="multi", dlogz=0.5)),
}

injection_base = dict(
    mass_1=150.0, mass_2=120.0,
    a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0,
    luminosity_distance=4000.0,
    theta_jn=0.8, psi=2.659, phase=1.3,
    ra=1.375, dec=-1.2108,
    geocent_time=1242442967.4,
    f0_res=60.0, gamma_res=10.0,
)


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


def tau_of(inj):
    return (inj["eps"] * (2 * np.pi * inj["f0_res"]) ** 2
            * inj["luminosity_distance"] * MPC_SEC
            / (2 * np.pi * inj["gamma_res"]))


def run(stage):
    cfg = STAGES[stage]
    injection = dict(injection_base, eps=cfg["eps"])
    outdir = os.path.join(OUTBASE, cfg["label"])
    np.random.seed(cfg["seed"])

    wf_args = dict(waveform_approximant="IMRPhenomXPHM",
                   reference_frequency=20.0, minimum_frequency=f_min)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=kernel_bbh,
        waveform_arguments=wf_args)

    # kernel sign and depth check against the analytic tau
    fa = waveform_generator.frequency_array
    p_off = dict(injection); p_off["eps"] = 0.0
    h_off = waveform_generator.frequency_domain_strain(p_off)["plus"]
    h_on = waveform_generator.frequency_domain_strain(injection)["plus"]
    i0 = np.argmin(np.abs(fa - injection["f0_res"]))
    r = np.abs(h_on[i0]) / np.abs(h_off[i0])
    tau = tau_of(injection)
    print(f"[check] damping at {fa[i0]:.1f} Hz: {r:.4f}   "
          f"analytic exp(-tau): {np.exp(-tau):.4f}   (tau = {tau:.3f})")
    assert r < 1.0, "Kernel amplifies at resonance: sign error"
    assert abs(r - np.exp(-tau)) < 0.02, "Damping depth disagrees with analytic tau"

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency, ifo.maximum_frequency = f_min, f_max
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection["geocent_time"] - duration + 2.0)
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection)

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

    print("priors going into sampler:")
    for k in ["mass_1", "mass_2", "chirp_mass", "mass_ratio",
              "eps", "f0_res", "gamma_res"]:
        print(f"    {k}: {priors[k]}")

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator,
        priors=priors, phase_marginalization=False,
        time_marginalization=False, distance_marginalization=False)

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler="dynesty",
        npool=4, injection_parameters=injection,
        outdir=outdir, label=cfg["label"], **cfg["sampler"])
    result.plot_corner()
    return result


def gate(stage, result):
    inj = result.injection_parameters
    p = result.posterior
    print(f"\n===== GATE 1b-{stage}  (injected tau = {tau_of(inj):.3f}) =====")
    truth = dict(
        chirp_mass=(inj["mass_1"] * inj["mass_2"]) ** 0.6
                   / (inj["mass_1"] + inj["mass_2"]) ** 0.2,
        mass_ratio=inj["mass_2"] / inj["mass_1"],
        luminosity_distance=inj["luminosity_distance"],
        geocent_time=inj["geocent_time"], phase=inj["phase"],
        eps=inj["eps"], f0_res=inj["f0_res"], gamma_res=inj["gamma_res"])
    for k, v in truth.items():
        lo, hi = np.percentile(p[k], [5, 95])
        print(f"{k:22s} truth={v:12.4e}  90% CI=[{lo:.4e}, {hi:.4e}]  "
              f"{'inside' if lo <= v <= hi else 'OUTSIDE'}")
    eps = p["eps"].values
    print(f"fraction of eps posterior below 1e-22: {(eps < 1e-22).mean():.2f} "
          f"(prior: 0.50)")
    print(f"eps/gamma median {np.median(eps / p['gamma_res'].values):.3e}  "
          f"(truth {inj['eps'] / inj['gamma_res']:.3e})")
    print(f"lnBF(kernel vs noise) = {result.log_bayes_factor:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit("usage: python scripts/run_1b.py marginal|loud")
    stage = sys.argv[1]
    cfg = STAGES[stage]
    path = os.path.join(OUTBASE, cfg["label"], f"{cfg['label']}_result.json")
    if os.path.exists(path):
        print(f"[load] {path}")
        result = bilby.result.read_in_result(path)
    else:
        result = run(stage)
    gate(stage, result)