"""
LMH Rescue Script
=================
Recovers from the v3.1 crash at line 329 by:
  1) Plotting the epsilon sweep from logged values (no recompute).
  2) Re-running ONLY M1 nested sampling (light) to recover posterior.
  3) Completing posterior analysis, null test, JSON summary.

Run time: ~8-12 minutes (vs 3.5 h re-running everything).
"""

from __future__ import annotations
import json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import dynesty
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from scipy.stats import wasserstein_distance, gaussian_kde, norm as scipy_norm
try:
    from statsmodels.stats.proportion import proportion_confint
except ImportError:
    def proportion_confint(s, n, method="wilson", alpha=0.05):
        z = scipy_norm.ppf(1 - alpha/2)
        if n == 0: return (0.0, 1.0)
        p = s/n; d = 1 + z**2/n
        c = (p + z**2/(2*n)) / d
        h = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
        return (max(0.0, c-h), min(1.0, c+h))

# ----------------------------------------------------------------------------
# Hard-coded sweep results from the previous run's log
# ----------------------------------------------------------------------------
SWEEP_DATA = {
    "eps":    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    "M0_lnZ": [(-39.4600, 0.5894), (-35.0070, 0.5596), (-30.2164, 0.5191),
               (-25.7109, 0.4744), (-21.0522, 0.4323)],
    "M1_lnZ": [(-28.8120, 0.5031), (-24.3401, 0.4618), (-18.9596, 0.4077),
               (-14.5922, 0.3556), (-10.5609, 0.3008)],
}
MAIN_RUN = {
    "M0_lnZ": (-30.1059, 0.2355),
    "M1_lnZ": (-19.3440, 0.1883),
    "ln_K_10": 10.7619,
    "ln_K_10_err": 0.3015,
    "verdict": "Decisive evidence in favor of M1",
    # theta_MAP from differential evolution (Step 1)
    "theta_MAP_M0": [4.1537e-05, 2.5254e-05],
    "theta_MAP_M1": [0.0029, 0.0169],
}

# ----------------------------------------------------------------------------
# Config + paths
# ----------------------------------------------------------------------------
N1, N2 = 16, 20
N_NODES = N1 + N2
SEED = 20260524
TARGET_R1, TARGET_R2 = 5.0, 4.0
VACUUM_EPS = 1e-6
OUTPUT_DIR = Path("./output_v3p1_rescue")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUTPUT_DIR/"rescue.log", mode="w", encoding="utf-8")],
)
log = logging.getLogger("rescue")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Rebuild graph exactly as before
# ----------------------------------------------------------------------------
def build_graph():
    rng = np.random.default_rng(SEED)
    N = N1 + N2; A = np.zeros((N, N))
    bip = (rng.random((N1, N2)) < 0.30).astype(np.float64)
    A[:N1, N1:] = bip; A[N1:, :N1] = bip.T
    for size, off in [(N1, 0), (N2, N1)]:
        noise = (rng.random((size, size)) < 0.05).astype(np.float64)
        noise = np.triu(noise, k=1); noise = noise + noise.T
        A[off:off+size, off:off+size] = noise
    return torch.tensor(A, dtype=DTYPE, device=DEVICE)

A_T = build_graph()
A2_T = A_T @ A_T
EIGVALS_A = torch.linalg.eigvalsh(A_T).cpu().numpy()

def extract_masses(eigvals, k=3, eps=VACUUM_EPS):
    if eigvals.ndim == 1:
        s = eigvals - eigvals.min() + eps
        return torch.sort(s)[0][:k]
    s = eigvals - eigvals.min(dim=-1, keepdim=True).values + eps
    return torch.sort(s, dim=-1)[0][..., :k]

# ============================================================================
# 1) Plot epsilon sweep (no recompute)
# ============================================================================
def plot_eps_sweep():
    eps = SWEEP_DATA["eps"]
    m0 = np.array(SWEEP_DATA["M0_lnZ"])
    m1 = np.array(SWEEP_DATA["M1_lnZ"])
    lnK = m1[:, 0] - m0[:, 0]
    lnK_err = np.sqrt(m0[:, 1]**2 + m1[:, 1]**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.errorbar(eps, m0[:, 0], yerr=m0[:, 1], fmt="o-",
                label=r"$\mathcal{M}_0:\ \alpha A + r L$",
                color="indianred", capsize=5, lw=2, ms=8)
    ax.errorbar(eps, m1[:, 0], yerr=m1[:, 1], fmt="s-",
                label=r"$\mathcal{M}_1:\ \alpha A + \beta A^2$",
                color="steelblue", capsize=5, lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"vacuum shift $\varepsilon$", fontsize=12)
    ax.set_ylabel(r"$\ln Z$", fontsize=12)
    ax.set_title("Bayesian evidence vs vacuum-shift regularization")
    ax.grid(alpha=0.3); ax.legend(fontsize=11)

    ax = axes[1]
    ax.errorbar(eps, lnK, yerr=lnK_err, fmt="D-",
                color="darkgreen", capsize=5, lw=2, ms=10,
                label=r"$\ln K_{10} = \ln Z_1 - \ln Z_0$")
    ax.axhline(5.0, color="orange", ls="--", label="Jeffreys 'decisive' threshold")
    ax.axhline(np.mean(lnK), color="darkgreen", ls=":", alpha=0.5,
               label=f"mean = {np.mean(lnK):.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"vacuum shift $\varepsilon$", fontsize=12)
    ax.set_ylabel(r"$\ln K_{10}$", fontsize=12)
    ax.set_title(f"Bayes factor robustness  |  range = {np.ptp(lnK):.2f}")
    ax.grid(alpha=0.3); ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"epsilon_sweep.png", dpi=140)
    plt.close(fig)
    log.info(f"saved {OUTPUT_DIR/'epsilon_sweep.png'}")
    log.info(f"ln(K_10) sweep stats: mean={np.mean(lnK):.3f}  "
             f"range={np.ptp(lnK):.3f}  min={lnK.min():.3f}  max={lnK.max():.3f}")

# ============================================================================
# 2) Re-run ONLY M1 nested sampling (light)
# ============================================================================
LOG_T1, LOG_T2 = np.log(TARGET_R1), np.log(TARGET_R2)

def log_cost_M1(theta):
    alpha, beta = float(theta[0]), float(theta[1])
    H = alpha * A_T + beta * A2_T
    H = 0.5*(H + H.T)
    e = torch.linalg.eigvalsh(H)
    m = extract_masses(e)
    if torch.any(m <= 0): return 1e6
    r1, r2 = (m[1]/m[0]).item(), (m[2]/m[1]).item()
    if r1 <= 0 or r2 <= 0: return 1e6
    return (np.log(r1)-LOG_T1)**2 + (np.log(r2)-LOG_T2)**2

def ll_M1(theta):
    v = log_cost_M1(theta)
    if v >= 1e6: return -1e10
    return -0.5 * v / (0.05**2)

def pt_M1(u):
    bounds = [(-5.0, 5.0), (-2.0, 2.0)]
    return np.array([b[0] + (b[1]-b[0])*ui for ui, b in zip(u, bounds)])

def run_M1_light():
    log.info("Re-running M1 nested sampling (nlive=800, dlogz=1.0)")
    sampler = dynesty.NestedSampler(
        ll_M1, pt_M1, ndim=2, nlive=800,
        bound="multi", sample="rslice")
    t0 = time.time()
    sampler.run_nested(dlogz=1.0, maxiter=80_000, print_progress=True)
    elapsed = time.time() - t0
    res = sampler.results
    samples = res.samples
    weights = np.exp(res.logwt - res.logz[-1])
    post = dyfunc.resample_equal(samples, weights)
    means, cov = dyfunc.mean_and_cov(samples, weights)
    log.info(f"elapsed={elapsed:.1f}s  ln(Z)={res.logz[-1]:.3f} ± {res.logzerr[-1]:.3f}")
    log.info(f"alpha={means[0]:+.4f} ± {np.sqrt(cov[0,0]):.4f}")
    log.info(f"beta ={means[1]:+.4f} ± {np.sqrt(cov[1,1]):.4f}")
    return res, post

# ============================================================================
# 3) Posterior analysis
# ============================================================================
def kde_map(posterior):
    kde = gaussian_kde(posterior.T, bw_method="scott")
    xs = np.linspace(posterior[:,0].min(), posterior[:,0].max(), 200)
    ys = np.linspace(posterior[:,1].min(), posterior[:,1].max(), 200)
    XX, YY = np.meshgrid(xs, ys)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(200, 200)
    i, j = np.unravel_index(np.argmax(Z), Z.shape)
    return np.array([xs[j], ys[i]])

def identify_triplet(theta):
    a, b = theta
    mapped = a * EIGVALS_A + b * EIGVALS_A**2
    s = mapped - mapped.min() + VACUUM_EPS
    return tuple(int(i) for i in np.argsort(s)[:3])

# ============================================================================
# 4) Null hypothesis test
# ============================================================================
def gen_random_graphs(topo, n, N, rng):
    A = np.zeros((n, N, N), dtype=np.float32)
    if topo == "erdos_renyi":
        p = 2*0.30*N1*N2 / (N*(N-1))
        for i in range(n):
            m = rng.random((N, N)) < p
            m = np.triu(m, k=1); A[i] = (m + m.T).astype(np.float32)
    elif topo == "watts_strogatz":
        for i in range(n):
            G = nx.watts_strogatz_graph(N, k=4, p=0.3, seed=int(rng.integers(0, 2**31-1)))
            A[i] = nx.to_numpy_array(G, dtype=np.float32)
    elif topo == "barabasi_albert":
        for i in range(n):
            G = nx.barabasi_albert_graph(N, m=3, seed=int(rng.integers(0, 2**31-1)))
            A[i] = nx.to_numpy_array(G, dtype=np.float32)
    return torch.from_numpy(A)

def null_test(theta_best, n_graphs=10000, batch=250, tol=0.20):
    log.info(f"Null test with theta_best={theta_best}")
    rng_master = np.random.default_rng(SEED + 7)
    out = {}
    for topo in ("erdos_renyi", "watts_strogatz", "barabasi_albert"):
        log.info(f"  topology: {topo}")
        rng = np.random.default_rng(rng_master.integers(0, 2**31-1))
        r1a, r2a, succ, t0, done = [], [], 0, time.time(), 0
        while done < n_graphs:
            bs = min(batch, n_graphs-done)
            A = gen_random_graphs(topo, bs, N_NODES, rng).to(DEVICE, DTYPE)
            A2 = torch.matmul(A, A)
            H = theta_best[0]*A + theta_best[1]*A2
            H = 0.5*(H + H.transpose(-2,-1))
            e = torch.linalg.eigvalsh(H)
            m = extract_masses(e)
            r1 = (m[...,1]/m[...,0]).cpu().numpy()
            r2 = (m[...,2]/m[...,1]).cpu().numpy()
            r1a.append(r1); r2a.append(r2)
            ok = (np.abs(r1-TARGET_R1)/TARGET_R1 < tol) & \
                 (np.abs(r2-TARGET_R2)/TARGET_R2 < tol)
            succ += int(ok.sum()); done += bs
        r1a, r2a = np.concatenate(r1a), np.concatenate(r2a)
        d2 = np.sqrt((r1a-TARGET_R1)**2 + (r2a-TARGET_R2)**2)
        cap = np.quantile(d2, 0.95)
        median_d = float(np.median(d2))
        trim_w1 = float(d2[d2 <= cap].mean())
        p_lo, p_hi = proportion_confint(succ, n_graphs, method="wilson")
        p = succ/n_graphs
        log.info(f"    succ={succ}/{n_graphs}  p={p:.5f}  Wilson95%=[{p_lo:.5f},{p_hi:.5f}]")
        log.info(f"    median_dist={median_d:.4f}  trim_W1={trim_w1:.4f}  elapsed={time.time()-t0:.1f}s")
        out[topo] = dict(r1=r1a, r2=r2a, p_value=p,
                         p_ci=(float(p_lo), float(p_hi)),
                         successes=succ, median_dist=median_d, trim_W1=trim_w1)
    return out

def plot_null(null_results):
    n = len(null_results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.5*n))
    if n == 1: axes = axes[np.newaxis, :]
    for row, (topo, res) in enumerate(null_results.items()):
        r1, r2 = res["r1"], res["r2"]
        ax = axes[row, 0]
        ax.hist(np.clip(r1, 0, 20), bins=80, range=(0, 20),
                alpha=0.7, color="steelblue", label=r"$r_1$")
        ax.hist(np.clip(r2, 0, 20), bins=80, range=(0, 20),
                alpha=0.6, color="indianred", label=r"$r_2$")
        ax.axvline(TARGET_R1, color="steelblue", ls="--")
        ax.axvline(TARGET_R2, color="indianred", ls="--")
        ax.set_title(f"{topo} | p={res['p_value']:.4f}  "
                     f"CI=[{res['p_ci'][0]:.4f},{res['p_ci'][1]:.4f}]")
        ax.legend(); ax.grid(alpha=0.3)
        ax = axes[row, 1]
        mk = (r1<30)&(r2<30)
        h = ax.hist2d(r1[mk], r2[mk], bins=80, range=[[0,15],[0,15]],
                      cmap="magma", norm=LogNorm(vmin=1))
        ax.scatter([TARGET_R1],[TARGET_R2], c="cyan", s=200, marker="*",
                   edgecolors="white", label="target")
        ax.set_xlabel(r"$r_1$"); ax.set_ylabel(r"$r_2$")
        ax.set_title(f"{topo}  joint H0 (log)")
        ax.legend(); plt.colorbar(h[3], ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"null_test.png", dpi=140)
    plt.close(fig)
    log.info(f"saved {OUTPUT_DIR/'null_test.png'}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    log.info(f"Torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    log.info("=" * 72)
    log.info("STAGE A — Plot epsilon sweep from cached values")
    log.info("=" * 72)
    plot_eps_sweep()

    log.info("=" * 72)
    log.info("STAGE B — Re-run M1 NS (light)")
    log.info("=" * 72)
    res_M1, post_M1 = run_M1_light()
    np.save(OUTPUT_DIR/"posterior_M1.npy", post_M1)

    log.info("=" * 72)
    log.info("STAGE C — Posterior analysis")
    log.info("=" * 72)
    kmap = kde_map(post_M1)
    log.info(f"KDE-MAP (alpha, beta) = ({kmap[0]:+.4f}, {kmap[1]:+.4f})")
    trip = identify_triplet(kmap)
    log.info(f"Triplet at KDE-MAP: indices {trip} | A-eigvals: "
             f"{[float(EIGVALS_A[i]) for i in trip]}")
    trips = np.array([identify_triplet(s) for s in post_M1])
    uniq, cnts = np.unique(trips, axis=0, return_counts=True)
    log.info(f"Posterior support across {len(uniq)} distinct triplets:")
    for t, c in sorted(zip(uniq, cnts), key=lambda x: -x[1])[:5]:
        log.info(f"   triplet={tuple(t.tolist())}  frac={c/len(trips)*100:.2f}%")

    # M1 corner + run plot
    fig, _ = dyplot.cornerplot(res_M1, labels=[r"$\alpha$", r"$\beta$"],
                               show_titles=True, title_fmt=".4f",
                               quantiles=[0.16, 0.5, 0.84])
    fig.savefig(OUTPUT_DIR/"ns_cornerplot_M1.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved {OUTPUT_DIR/'ns_cornerplot_M1.png'}")

    log.info("=" * 72)
    log.info("STAGE D — Null hypothesis test (M1 KDE-MAP)")
    log.info("=" * 72)
    null_results = null_test(kmap)
    plot_null(null_results)

    log.info("=" * 72)
    log.info("STAGE E — Persist summary.json")
    log.info("=" * 72)
    summary = {
        "main_run": MAIN_RUN,
        "sweep": {
            "eps": SWEEP_DATA["eps"],
            "M0_lnZ": SWEEP_DATA["M0_lnZ"],
            "M1_lnZ": SWEEP_DATA["M1_lnZ"],
            "ln_K_per_eps": [(m1[0]-m0[0]) for m0, m1 in
                             zip(SWEEP_DATA["M0_lnZ"], SWEEP_DATA["M1_lnZ"])],
        },
        "M1_posterior_rescue": {
            "kde_MAP": kmap.tolist(),
            "triplet_at_MAP": list(trip),
            "ln_Z_light": float(res_M1.logz[-1]),
        },
        "null_test": {
            topo: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in r.items() if k not in ("r1", "r2")}
            for topo, r in null_results.items()
        },
    }
    with open(OUTPUT_DIR/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"saved {OUTPUT_DIR/'summary.json'}")

    log.info("=" * 72)
    log.info("RESCUE COMPLETE")
    log.info("=" * 72)
    lnK = [m1[0]-m0[0] for m0, m1 in zip(SWEEP_DATA["M0_lnZ"], SWEEP_DATA["M1_lnZ"])]
    log.info(f"Main ln(K_10) = {MAIN_RUN['ln_K_10']:+.4f} ± {MAIN_RUN['ln_K_10_err']:.4f}")
    log.info(f"Sweep ln(K_10): mean={np.mean(lnK):.3f}  "
             f"range={np.ptp(lnK):.3f}  all decisively > 5")
    log.info(f"M1 KDE-MAP = ({kmap[0]:+.4f}, {kmap[1]:+.4f})")
    for topo, r in null_results.items():
        log.info(f"  null[{topo:<15s}]: p={r['p_value']:.5f}  trim_W1={r['trim_W1']:.3f}")

if __name__ == "__main__":
    main()