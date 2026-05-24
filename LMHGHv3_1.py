"""
Lepton Mass Hierarchy — Pipeline v3.1
======================================
2-parameter models with Bayesian model comparison + vacuum-shift sensitivity.

Models (both 2-D, uniform priors):
    M0 : H = alpha * A + r    * L            (combinatorial Laplacian)
    M1 : H = alpha * A + beta * A^2          (second-neighbor interactions)

Decision metric: ln K_{10} = ln Z_1 - ln Z_0  (Jeffreys 1961, Kass-Raftery 1995).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from scipy.optimize import differential_evolution
from scipy.stats import wasserstein_distance, gaussian_kde, norm as scipy_norm

# Optional statsmodels fallback ------------------------------------------------
try:
    from statsmodels.stats.proportion import proportion_confint
except ImportError:
    def proportion_confint(successes: int, n: int, method: str = "wilson",
                           alpha: float = 0.05):
        """Inline Wilson CI fallback if statsmodels is unavailable."""
        if n == 0:
            return (0.0, 1.0)
        z = scipy_norm.ppf(1.0 - alpha / 2.0)
        p_hat = successes / n
        denom = 1.0 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
        return (max(0.0, center - half), min(1.0, center + half))


# ============================================================================
# 0) CONFIGURATION + LOGGING
# ============================================================================
ModelID = Literal["M0", "M1"]


@dataclass
class Config:
    # Graph
    n1: int = 16
    n2: int = 20
    p_bipartite: float = 0.30
    p_intra_noise: float = 0.05
    seed: int = 20260524

    # Targets
    target_r1: float = 5.0
    target_r2: float = 4.0
    feasibility_rel_tol: float = 0.05

    # 2-D uniform priors -- same support magnitudes => fair Occam factor
    prior_M0: tuple = ((-5.0, 5.0), (-5.0, 5.0))    # (alpha, r)
    prior_M1: tuple = ((-5.0, 5.0), (-2.0, 2.0))    # (alpha, beta)

    # Likelihood
    sigma_log: float = 0.05

    # Vacuum shift (the numerical regularizer for positive masses).
    # We sweep this in Step 3.5 to prove M1 is robust to its choice.
    vacuum_eps: float = 1e-6
    eps_sweep_values: tuple = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)

    # Nested sampling -- relaxed for Bayes-factor purposes
    nlive: int = 2000
    dlogz_stop: float = 0.5            # |lnK| ~ 1e3+ expected; 0.5 is overkill
    ns_maxiter: int = 200_000          # hard cap for safety
    bound_method: str = "multi"
    sample_method: str = "rslice"

    # DE feasibility
    de_maxiter: int = 400
    de_popsize: int = 30

    # Null test
    null_n_graphs: int = 10_000
    null_batch: int = 250
    null_tolerance: float = 0.20
    null_topologies: tuple = ("erdos_renyi", "watts_strogatz", "barabasi_albert")
    null_trim_quantile: float = 0.95

    output_dir: Path = Path("./output_v3p1")


CFG = Config()
CFG.output_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CFG.output_dir / "run.log", mode="w"),
    ],
)
log = logging.getLogger("LMHv3.1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

LOG_T1 = np.log(CFG.target_r1)
LOG_T2 = np.log(CFG.target_r2)


# ============================================================================
# 1) GRAPH + PRECOMPUTED OPERATORS
# ============================================================================
def build_asymmetric_graph(cfg: Config = CFG) -> torch.Tensor:
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n1 + cfg.n2
    A = np.zeros((N, N), dtype=np.float64)

    bip = (rng.random((cfg.n1, cfg.n2)) < cfg.p_bipartite).astype(np.float64)
    A[:cfg.n1, cfg.n1:] = bip
    A[cfg.n1:, :cfg.n1] = bip.T

    for size, offset in [(cfg.n1, 0), (cfg.n2, cfg.n1)]:
        noise = (rng.random((size, size)) < cfg.p_intra_noise).astype(np.float64)
        noise = np.triu(noise, k=1)
        noise = noise + noise.T
        A[offset:offset+size, offset:offset+size] = noise

    return torch.tensor(A, dtype=DTYPE, device=DEVICE)


A_TENSOR  = build_asymmetric_graph(CFG)
A2_TENSOR = A_TENSOR @ A_TENSOR
L_TENSOR  = torch.diag(A_TENSOR.sum(dim=1)) - A_TENSOR
EIGVALS_A = torch.linalg.eigvalsh(A_TENSOR).cpu().numpy()


# ============================================================================
# 2) MODEL DISPATCH + SPECTRAL OBSERVABLES
# ============================================================================
def build_hamiltonian(theta, model: ModelID) -> torch.Tensor:
    if model == "M0":
        alpha, r = float(theta[0]), float(theta[1])
        H = alpha * A_TENSOR + r * L_TENSOR
    elif model == "M1":
        alpha, beta = float(theta[0]), float(theta[1])
        H = alpha * A_TENSOR + beta * A2_TENSOR
    else:
        raise ValueError(f"Unknown model: {model}")
    return 0.5 * (H + H.T)


def extract_masses(eigvals: torch.Tensor, k: int = 3, eps: float = None):
    if eps is None:
        eps = CFG.vacuum_eps
    if eigvals.ndim == 1:
        shifted = eigvals - eigvals.min() + eps
        sorted_, _ = torch.sort(shifted)
        return sorted_[:k]
    else:
        m_min = eigvals.min(dim=-1, keepdim=True).values
        shifted = eigvals - m_min + eps
        sorted_, _ = torch.sort(shifted, dim=-1)
        return sorted_[..., :k]


def ratios_from_masses(m: torch.Tensor):
    if m.ndim == 1:
        return (m[1] / m[0]).item(), (m[2] / m[1]).item()
    return m[..., 1] / m[..., 0], m[..., 2] / m[..., 1]


# ============================================================================
# 3) LOG-COST AND LIKELIHOOD
# ============================================================================
def log_cost(theta_np: np.ndarray, model: ModelID, eps: float = None) -> float:
    H = build_hamiltonian(theta_np, model)
    eigvals = torch.linalg.eigvalsh(H)
    m = extract_masses(eigvals, k=3, eps=eps)
    if torch.any(m <= 0):
        return 1e6
    r1, r2 = ratios_from_masses(m)
    if r1 <= 0 or r2 <= 0:
        return 1e6
    return (np.log(r1) - LOG_T1) ** 2 + (np.log(r2) - LOG_T2) ** 2


def make_loglike(model: ModelID, eps: float = None):
    def _ll(theta_np: np.ndarray) -> float:
        val = log_cost(theta_np, model, eps=eps)
        if val >= 1e6:
            return -1e10
        return -0.5 * val / (CFG.sigma_log ** 2)
    return _ll


def make_prior_transform(model: ModelID):
    bounds = CFG.prior_M0 if model == "M0" else CFG.prior_M1
    def _pt(u: np.ndarray) -> np.ndarray:
        return np.array([b[0] + (b[1] - b[0]) * ui for ui, b in zip(u, bounds)])
    return _pt


# ============================================================================
# 4) FEASIBILITY
# ============================================================================
def feasibility_test(model: ModelID, cfg: Config = CFG):
    log.info(f"  Feasibility test ({model}) ...")
    bounds = cfg.prior_M0 if model == "M0" else cfg.prior_M1
    t0 = time.time()
    res = differential_evolution(
        func=lambda th: log_cost(th, model),
        bounds=list(bounds),
        seed=cfg.seed, maxiter=cfg.de_maxiter, popsize=cfg.de_popsize,
        tol=1e-9, polish=True, workers=1, updating="immediate",
        mutation=(0.5, 1.5), recombination=0.9,
    )
    elapsed = time.time() - t0
    theta_map = res.x
    H = build_hamiltonian(theta_map, model)
    m = extract_masses(torch.linalg.eigvalsh(H), k=3)
    r1, r2 = ratios_from_masses(m)
    err1 = abs(r1 - cfg.target_r1) / cfg.target_r1
    err2 = abs(r2 - cfg.target_r2) / cfg.target_r2
    feasible = (err1 < cfg.feasibility_rel_tol) and (err2 < cfg.feasibility_rel_tol)

    log.info(f"    [{model}] elapsed={elapsed:.1f}s | theta_MAP={np.array2string(theta_map, precision=4)}")
    log.info(f"    [{model}] log_cost*={res.fun:.4e}  (r1,r2)=({r1:.4f},{r2:.4f})  err=({err1*100:.2f}%,{err2*100:.2f}%)")
    log.info(f"    [{model}] feasible within ±{cfg.feasibility_rel_tol*100:.0f}%: {feasible}")
    return theta_map, float(res.fun), feasible, (float(r1), float(r2))


# ============================================================================
# 5) NESTED SAMPLING
# ============================================================================
def run_nested(model: ModelID, cfg: Config = CFG, eps: float = None):
    log.info(f"  Nested sampling ({model}) | nlive={cfg.nlive} | dlogz={cfg.dlogz_stop}"
             f" | eps={eps if eps is not None else cfg.vacuum_eps:.0e}")
    sampler = dynesty.NestedSampler(
        loglikelihood=make_loglike(model, eps=eps),
        prior_transform=make_prior_transform(model),
        ndim=2,
        nlive=cfg.nlive,
        bound=cfg.bound_method,
        sample=cfg.sample_method,
    )
    t0 = time.time()
    sampler.run_nested(
        dlogz=cfg.dlogz_stop,
        maxiter=cfg.ns_maxiter,           # safety cap
        print_progress=True,
    )
    elapsed = time.time() - t0
    res = sampler.results

    logz, logz_err = float(res.logz[-1]), float(res.logzerr[-1])
    samples = res.samples
    weights = np.exp(res.logwt - res.logz[-1])
    posterior_eq = dyfunc.resample_equal(samples, weights)
    means, cov = dyfunc.mean_and_cov(samples, weights)
    stds = np.sqrt(np.diag(cov))
    idx_best = int(np.argmax(res.logl))
    theta_best = res.samples[idx_best]

    log.info(f"    [{model}] elapsed={elapsed:.1f}s | ln(Z)={logz:.4f} ± {logz_err:.4f}")
    p_names = ["alpha", "r"] if model == "M0" else ["alpha", "beta"]
    for nm, mu, sd in zip(p_names, means, stds):
        log.info(f"    [{model}]    {nm:<6s} = {mu:+.4f} ± {sd:.4f}")
    log.info(f"    [{model}] theta_best (max-L sample) = {np.array2string(theta_best, precision=4)}")

    return dict(
        results=res, posterior=posterior_eq, theta_best=theta_best,
        logz=logz, logz_err=logz_err, means=means, stds=stds,
        p_names=p_names, elapsed=elapsed,
    )


# ============================================================================
# 6) VACUUM-SHIFT SENSITIVITY SWEEP
# ============================================================================
def epsilon_sensitivity_sweep(cfg: Config = CFG) -> dict:
    """
    Re-run NS for both models across a range of vacuum-shift values.
    A model whose ln(Z) depends strongly on eps is numerically fragile.
    """
    log.info("=" * 72)
    log.info("STEP 3.5 — VACUUM-SHIFT SENSITIVITY SWEEP")
    log.info("=" * 72)

    # Lighter NS for the sweep -- we only need order-of-magnitude trends.
    saved_nlive, saved_dlogz = cfg.nlive, cfg.dlogz_stop
    cfg.nlive = max(500, cfg.nlive // 2)
    cfg.dlogz_stop = max(1.0, cfg.dlogz_stop * 4)

    sweep_data = {"eps": list(cfg.eps_sweep_values), "M0_lnZ": [], "M1_lnZ": []}
    try:
        for eps in cfg.eps_sweep_values:
            log.info(f"  -- eps = {eps:.0e} --")
            r0 = run_nested("M0", cfg, eps=eps)
            r1 = run_nested("M1", cfg, eps=eps)
            sweep_data["M0_lnZ"].append((r0["logz"], r0["logz_err"]))
            sweep_data["M1_lnZ"].append((r1["logz"], r1["logz_err"]))
    finally:
        cfg.nlive, cfg.dlogz_stop = saved_nlive, saved_dlogz

    # Report stability
    m0 = np.array([z for z, _ in sweep_data["M0_lnZ"]])
    m1 = np.array([z for z, _ in sweep_data["M1_lnZ"]])
    log.info("  ----- SUMMARY -----")
    log.info(f"  ln(Z_M0) across eps : "
             f"min={m0.min():.2f}  max={m0.max():.2f}  range={m0.ptp():.2f}")
    log.info(f"  ln(Z_M1) across eps : "
             f"min={m1.min():.2f}  max={m1.max():.2f}  range={m1.ptp():.2f}")
    log.info(f"  Interpretation: a wider range across eps => more fragility to "
             f"the vacuum regularization choice.")
    return sweep_data


def plot_eps_sweep(sweep_data: dict, out_dir: Path):
    eps = sweep_data["eps"]
    m0 = np.array(sweep_data["M0_lnZ"])
    m1 = np.array(sweep_data["M1_lnZ"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(eps, m0[:, 0], yerr=m0[:, 1], fmt="o-",
                label=r"$\mathcal{M}_0:\ \alpha A + r L$", color="indianred", capsize=4)
    ax.errorbar(eps, m1[:, 0], yerr=m1[:, 1], fmt="s-",
                label=r"$\mathcal{M}_1:\ \alpha A + \beta A^2$", color="steelblue", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel(r"vacuum shift $\varepsilon$")
    ax.set_ylabel(r"$\ln(Z)$")
    ax.set_title("Bayesian evidence vs vacuum-shift regularization")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_sweep.png", dpi=140)
    plt.close(fig)
    log.info(f"  saved {out_dir/'epsilon_sweep.png'}")


# ============================================================================
# 7) POSTERIOR ANALYSIS
# ============================================================================
def posterior_kde_map(posterior: np.ndarray, grid_size: int = 200):
    kde = gaussian_kde(posterior.T, bw_method="scott")
    x_min, x_max = posterior[:, 0].min(), posterior[:, 0].max()
    y_min, y_max = posterior[:, 1].min(), posterior[:, 1].max()
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(grid_size, grid_size)
    i, j = np.unravel_index(np.argmax(Z), Z.shape)
    return np.array([xs[j], ys[i]]), Z, xs, ys


def identify_eigenvalue_triplet(theta, model: ModelID):
    if model != "M1":
        return None
    alpha, beta = theta
    mapped = alpha * EIGVALS_A + beta * EIGVALS_A**2
    shifted = mapped - mapped.min() + CFG.vacuum_eps
    order = np.argsort(shifted)
    return tuple(int(i) for i in order[:3])


# ============================================================================
# 8) BAYES FACTOR INTERPRETATION
# ============================================================================
def interpret_bayes_factor(ln_K: float) -> str:
    a = abs(ln_K)
    fav = "M1" if ln_K > 0 else "M0"
    if a < 1.0:
        return "Inconclusive (not worth more than a bare mention)"
    if a < 2.5:
        return f"Substantial evidence in favor of {fav}"
    if a < 5.0:
        return f"Strong evidence in favor of {fav}"
    return f"Decisive evidence in favor of {fav}"


# ============================================================================
# 9) NULL HYPOTHESIS TEST
# ============================================================================
def generate_random_graphs(topology, n_graphs, n_nodes, rng):
    adjs = np.zeros((n_graphs, n_nodes, n_nodes), dtype=np.float32)
    if topology == "erdos_renyi":
        p = 2.0 * CFG.p_bipartite * CFG.n1 * CFG.n2 / (n_nodes * (n_nodes - 1))
        for i in range(n_graphs):
            mat = (rng.random((n_nodes, n_nodes)) < p)
            mat = np.triu(mat, k=1)
            adjs[i] = (mat + mat.T).astype(np.float32)
    elif topology == "watts_strogatz":
        for i in range(n_graphs):
            G = nx.watts_strogatz_graph(n_nodes, k=4, p=0.3,
                seed=int(rng.integers(0, 2**31 - 1)))
            adjs[i] = nx.to_numpy_array(G, dtype=np.float32)
    elif topology == "barabasi_albert":
        for i in range(n_graphs):
            G = nx.barabasi_albert_graph(n_nodes, m=3,
                seed=int(rng.integers(0, 2**31 - 1)))
            adjs[i] = nx.to_numpy_array(G, dtype=np.float32)
    else:
        raise ValueError(topology)
    return torch.from_numpy(adjs)


def null_test_robust(theta_best, model: ModelID, cfg: Config = CFG) -> dict:
    log.info(f"  Null test using theta from {model}: "
             f"{np.array2string(theta_best, precision=4)}")
    n_nodes = cfg.n1 + cfg.n2
    rng_master = np.random.default_rng(cfg.seed + 7)
    all_topo = {}

    for topo in cfg.null_topologies:
        log.info(f"    Topology: {topo}")
        rng_topo = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        r1_all, r2_all = [], []
        successes = 0
        t0 = time.time()
        n_done = 0
        while n_done < cfg.null_n_graphs:
            bs = min(cfg.null_batch, cfg.null_n_graphs - n_done)
            A_batch = generate_random_graphs(topo, bs, n_nodes, rng_topo)
            A_batch = A_batch.to(device=DEVICE, dtype=DTYPE)

            if model == "M0":
                deg = A_batch.sum(dim=-1)
                second = torch.diag_embed(deg) - A_batch
            else:
                second = torch.matmul(A_batch, A_batch)

            alpha_b, beta_b = float(theta_best[0]), float(theta_best[1])
            H = alpha_b * A_batch + beta_b * second
            H = 0.5 * (H + H.transpose(-2, -1))

            eigvals = torch.linalg.eigvalsh(H)
            m = extract_masses(eigvals, k=3)
            r1 = (m[..., 1] / m[..., 0]).cpu().numpy()
            r2 = (m[..., 2] / m[..., 1]).cpu().numpy()
            r1_all.append(r1); r2_all.append(r2)

            ok = (np.abs(r1 - cfg.target_r1) / cfg.target_r1 < cfg.null_tolerance) & \
                 (np.abs(r2 - cfg.target_r2) / cfg.target_r2 < cfg.null_tolerance)
            successes += int(ok.sum())
            n_done += bs

        r1_all = np.concatenate(r1_all)
        r2_all = np.concatenate(r2_all)

        # ---- Robust metrics ----
        dist_2d = np.sqrt((r1_all - cfg.target_r1)**2 + (r2_all - cfg.target_r2)**2)
        q_clip = cfg.null_trim_quantile
        cap = np.quantile(dist_2d, q_clip)
        trimmed_dist = dist_2d[dist_2d <= cap]
        median_dist = float(np.median(dist_2d))
        trimmed_w1 = float(trimmed_dist.mean())

        r1_cap = np.quantile(np.abs(r1_all - cfg.target_r1), q_clip)
        r2_cap = np.quantile(np.abs(r2_all - cfg.target_r2), q_clip)
        r1_trim = r1_all[np.abs(r1_all - cfg.target_r1) <= r1_cap]
        r2_trim = r2_all[np.abs(r2_all - cfg.target_r2) <= r2_cap]
        w1_r1 = float(wasserstein_distance(r1_trim, [cfg.target_r1]))
        w1_r2 = float(wasserstein_distance(r2_trim, [cfg.target_r2]))

        p_lo, p_hi = proportion_confint(successes, cfg.null_n_graphs, method="wilson")
        p_value = successes / cfg.null_n_graphs
        elapsed = time.time() - t0

        log.info(f"      successes={successes}/{cfg.null_n_graphs}  "
                 f"p={p_value:.5f}  Wilson95%=[{p_lo:.5f}, {p_hi:.5f}]")
        log.info(f"      median_dist(joint)={median_dist:.4f}  "
                 f"trim_W1={trimmed_w1:.4f}  marg W1=({w1_r1:.4f}, {w1_r2:.4f})")
        log.info(f"      elapsed={elapsed:.2f}s")

        all_topo[topo] = dict(
            r1=r1_all, r2=r2_all,
            p_value=p_value, p_ci=(float(p_lo), float(p_hi)),
            successes=int(successes),
            median_dist=median_dist, trimmed_w1=trimmed_w1,
            w1_r1=w1_r1, w1_r2=w1_r2,
        )

    return all_topo


# ============================================================================
# 10) PLOTTING
# ============================================================================
def plot_posterior_comparison(M0_data, M1_data, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, name in [
        (axes[0], M0_data, r"$\mathcal{M}_0$: $\alpha A + r L$"),
        (axes[1], M1_data, r"$\mathcal{M}_1$: $\alpha A + \beta A^2$"),
    ]:
        post = data["posterior"]
        h = ax.hist2d(post[:, 0], post[:, 1], bins=80, cmap="magma",
                      norm=LogNorm(vmin=1))
        kde_map, _, _, _ = posterior_kde_map(post)
        ax.scatter(*kde_map, c="cyan", s=200, marker="*",
                   edgecolors="white", linewidths=2,
                   label=f"KDE-MAP=({kde_map[0]:+.3g}, {kde_map[1]:+.3g})")
        ax.set_xlabel(data["p_names"][0]); ax.set_ylabel(data["p_names"][1])
        ax.set_title(f"{name}\n ln(Z) = {data['logz']:.3f} ± {data['logz_err']:.3f}")
        ax.legend(loc="upper right", fontsize=9)
        plt.colorbar(h[3], ax=ax, label="counts (log)")
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_comparison.png", dpi=140)
    plt.close(fig)
    log.info(f"  saved {out_dir/'posterior_comparison.png'}")


def plot_dynesty_per_model(data, model: ModelID, out_dir: Path):
    labels = [rf"${data['p_names'][0]}$", rf"${data['p_names'][1]}$"]
    results = data["results"]
    fig, _ = dyplot.runplot(results)
    fig.savefig(out_dir / f"ns_runplot_{model}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    fig, _ = dyplot.cornerplot(results, labels=labels, show_titles=True,
                               title_fmt=".4f", quantiles=[0.16, 0.5, 0.84])
    fig.savefig(out_dir / f"ns_cornerplot_{model}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {out_dir/f'ns_*_{model}.png'}")


def plot_null(null_results, out_dir: Path, cfg: Config = CFG):
    n = len(null_results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5 * n))
    if n == 1: axes = axes[np.newaxis, :]
    for row, (topo, res) in enumerate(null_results.items()):
        r1, r2 = res["r1"], res["r2"]
        ax = axes[row, 0]
        ax.hist(np.clip(r1, 0, 20), bins=80, range=(0, 20),
                alpha=0.7, color="steelblue", label=r"$r_1$")
        ax.hist(np.clip(r2, 0, 20), bins=80, range=(0, 20),
                alpha=0.6, color="indianred", label=r"$r_2$")
        ax.axvline(cfg.target_r1, color="steelblue", ls="--")
        ax.axvline(cfg.target_r2, color="indianred", ls="--")
        ax.set_title(f"{topo} | p={res['p_value']:.4f}  "
                     f"CI=[{res['p_ci'][0]:.4f},{res['p_ci'][1]:.4f}]  "
                     f"median={res['median_dist']:.3f}  trim_W1={res['trimmed_w1']:.3f}")
        ax.set_xlabel("ratio (clipped @20)"); ax.set_ylabel("counts")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        mask = (r1 < 30) & (r2 < 30)
        h = ax.hist2d(r1[mask], r2[mask], bins=80, range=[[0, 15], [0, 15]],
                      cmap="magma", norm=LogNorm(vmin=1))
        ax.scatter([cfg.target_r1], [cfg.target_r2], c="cyan", s=200,
                   marker="*", edgecolors="white", label="target")
        ax.set_xlabel(r"$r_1$"); ax.set_ylabel(r"$r_2$")
        ax.set_title(f"{topo} — joint H0 density (log)")
        ax.legend(); plt.colorbar(h[3], ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "null_test.png", dpi=140)
    plt.close(fig)
    log.info(f"  saved {out_dir/'null_test.png'}")


# ============================================================================
# 11) MAIN PIPELINE
# ============================================================================
def main():
    log.info(f"Torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Output: {CFG.output_dir.resolve()}")

    # ----- STEP 1: feasibility -----
    log.info("=" * 72); log.info("STEP 1 — FEASIBILITY"); log.info("=" * 72)
    map_M0, cost_M0, feas_M0, ratios_M0 = feasibility_test("M0")
    map_M1, cost_M1, feas_M1, ratios_M1 = feasibility_test("M1")
    if not feas_M1:
        log.error("M1 not feasible. Aborting.")
        sys.exit(1)
    if not feas_M0:
        log.warning("M0 not feasible -- continuing for fair Bayes comparison.")

    # ----- STEP 2: nested sampling for both -----
    log.info("=" * 72); log.info("STEP 2 — NESTED SAMPLING"); log.info("=" * 72)
    M0 = run_nested("M0")
    M1 = run_nested("M1")

    # ----- STEP 3: Bayes factor -----
    log.info("=" * 72); log.info("STEP 3 — BAYESIAN MODEL COMPARISON"); log.info("=" * 72)
    ln_K = M1["logz"] - M0["logz"]
    ln_K_err = float(np.sqrt(M1["logz_err"]**2 + M0["logz_err"]**2))
    K = float(np.exp(min(ln_K, 700.0)))    # cap to avoid overflow in exp
    verdict = interpret_bayes_factor(ln_K)
    log.info(f"  ln(Z_0)  [M0: αA + rL]    = {M0['logz']:+.4f} ± {M0['logz_err']:.4f}")
    log.info(f"  ln(Z_1)  [M1: αA + βA²]   = {M1['logz']:+.4f} ± {M1['logz_err']:.4f}")
    log.info(f"  ln(K_10) = ln(Z_1/Z_0)    = {ln_K:+.4f} ± {ln_K_err:.4f}")
    log.info(f"  K_10                      ≈ {K:.3e}" +
             (" (capped at exp(700))" if ln_K > 700 else ""))
    log.info(f"  Jeffreys verdict          : {verdict}")

    # ----- STEP 3.5: epsilon-sensitivity sweep -----
    sweep_data = epsilon_sensitivity_sweep(CFG)
    plot_eps_sweep(sweep_data, CFG.output_dir)

    # ----- STEP 4: posterior analysis on M1 -----
    log.info("=" * 72); log.info("STEP 4 — POSTERIOR ANALYSIS (M1)"); log.info("=" * 72)
    kde_map_M1, _, _, _ = posterior_kde_map(M1["posterior"])
    log.info(f"  KDE-MAP (alpha, beta) = ({kde_map_M1[0]:+.4f}, {kde_map_M1[1]:+.4f})")
    triplet_map = identify_eigenvalue_triplet(kde_map_M1, "M1")
    log.info(f"  Eigenvalue triplet at KDE-MAP: indices {triplet_map}")
    log.info(f"     A-spectrum: {[float(EIGVALS_A[i]) for i in triplet_map]}")

    triplets = np.array([identify_eigenvalue_triplet(s, "M1")
                         for s in M1["posterior"]])
    unique, counts = np.unique(triplets, axis=0, return_counts=True)
    log.info(f"  Posterior support across {len(unique)} distinct triplets:")
    for trip, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
        log.info(f"     triplet={tuple(trip.tolist())}  "
                 f"fraction={cnt/len(triplets)*100:.2f}%")

    # ----- STEP 5: plots -----
    plot_dynesty_per_model(M0, "M0", CFG.output_dir)
    plot_dynesty_per_model(M1, "M1", CFG.output_dir)
    plot_posterior_comparison(M0, M1, CFG.output_dir)

    # ----- STEP 6: null hypothesis test -----
    log.info("=" * 72); log.info("STEP 6 — NULL TEST (M1 KDE-MAP)"); log.info("=" * 72)
    null_results = null_test_robust(kde_map_M1, "M1")
    plot_null(null_results, CFG.output_dir)

    # ----- STEP 7: persist -----
    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(CFG).items()},
        "feasibility": {
            "M0": {"theta_MAP": map_M0.tolist(), "cost": cost_M0,
                   "feasible": feas_M0, "ratios": list(ratios_M0)},
            "M1": {"theta_MAP": map_M1.tolist(), "cost": cost_M1,
                   "feasible": feas_M1, "ratios": list(ratios_M1)},
        },
        "evidence": {
            "ln_Z_M0": M0["logz"], "ln_Z_M0_err": M0["logz_err"],
            "ln_Z_M1": M1["logz"], "ln_Z_M1_err": M1["logz_err"],
            "ln_K_10": ln_K, "ln_K_10_err": ln_K_err,
            "K_10": K, "jeffreys_verdict": verdict,
        },
        "epsilon_sweep": sweep_data,
        "posterior_M1": {
            "kde_MAP": kde_map_M1.tolist(),
            "means": M1["means"].tolist(),
            "stds":  M1["stds"].tolist(),
            "triplet_at_MAP": list(triplet_map),
        },
        "null_test": {
            topo: {k: (v if not isinstance(v, np.ndarray) else None)
                   for k, v in res.items()}
            for topo, res in null_results.items()
        },
    }
    with open(CFG.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  saved {CFG.output_dir/'summary.json'}")

    # ----- FINAL -----
    log.info("=" * 72); log.info("PIPELINE v3.1 COMPLETE"); log.info("=" * 72)
    log.info(f"  M0 theta_MAP={map_M0}  ratios={ratios_M0}")
    log.info(f"  M1 theta_MAP={map_M1}  ratios={ratios_M1}")
    log.info(f"  ln(K_10)    = {ln_K:+.4f}  -->  {verdict}")
    log.info(f"  M1 KDE-MAP  = ({kde_map_M1[0]:+.4f}, {kde_map_M1[1]:+.4f})")
    for topo, res in null_results.items():
        log.info(f"  null[{topo:<15s}]: p={res['p_value']:.5f}  "
                 f"trim_W1={res['trimmed_w1']:.3f}  median={res['median_dist']:.3f}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()