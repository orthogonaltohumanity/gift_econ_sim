import json
import glob
import re
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, kstest, pareto, lognorm
from PIL import Image
from tqdm import tqdm
import xgi


files = glob.glob("data_*.json")
files.sort(key=lambda f: int(re.search(r"data_(\d+)\.json", f).group(1)))

with open(files[0], "r") as f:
    pkt0 = json.load(f)
N = len(pkt0["op"])

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos = {n: (np.cos(a), np.sin(a)) for n, a in enumerate(angles)}


def dominant_groups(packet, eps=0.0):
    ops = packet["op"]
    keys = sorted(
        {g for a in ops for g in a},
        key=lambda g: (len(g.split(",")), g),
    )
    edges = []
    avg_ops = []
    for g in keys:
        members = [int(m) for m in g.split(",")]
        avg = float(np.mean([ops[m][g] for m in members]))
        if avg > eps:
            edges.append(members)
            avg_ops.append(avg)
    return edges, avg_ops


def spectral_entropy(edges, n_nodes):
    if not edges:
        return 0.0
    B = np.zeros((n_nodes, len(edges)))
    for j, e in enumerate(edges):
        for v in e:
            B[v, j] = 1.0
    sv = np.linalg.svd(B, compute_uv=False)
    sq = sv ** 2
    s = sq.sum()
    if s <= 0:
        return 0.0
    p = sq / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


frames = []
ts = []
spec_ent = []
prev_edge_set = None
last_change_t = None
change_intervals = []
active_since = {}      # frozenset(edge) -> t at which it (re)appeared
edge_lifetimes = []    # completed lives only
for fp in tqdm(files, desc="rendering hypergraph frames"):
    with open(fp, "r") as f:
        pkt = json.load(f)
    edges, avg_ops = dominant_groups(pkt, eps=0.05)
    t_now = pkt["t"]
    ts.append(t_now)
    spec_ent.append(spectral_entropy(edges, N))

    edge_set = frozenset(frozenset(e) for e in edges)
    if prev_edge_set is None:
        last_change_t = t_now
    elif edge_set != prev_edge_set:
        change_intervals.append(t_now - last_change_t)
        last_change_t = t_now

    for e in edge_set - (prev_edge_set or frozenset()):
        active_since[e] = t_now
    if prev_edge_set is not None:
        for e in prev_edge_set - edge_set:
            edge_lifetimes.append(t_now - active_since.pop(e))
    prev_edge_set = edge_set

    H = xgi.Hypergraph()
    H.add_nodes_from(range(N))
    H.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    if H.num_edges > 0:
        xgi.draw(H, pos=pos, ax=ax, node_size=15, node_labels=True,
                 edge_fc=avg_ops, edge_fc_cmap="viridis",
                 edge_vmin=0.05, edge_vmax=1.0)
    else:
        for n in range(N):
            ax.scatter(*pos[n], s=80, color="black")
            ax.annotate(str(n), pos[n], xytext=(4, 4),
                        textcoords="offset points")
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"),
        norm=plt.Normalize(vmin=0.05, vmax=1.0),
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("avg opinion of group")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"t = {pkt['t']}  ({H.num_edges} groups with avg opinion > 0.05)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))

frames[0].save(
    "hypergraph_evolution.gif",
    save_all=True,
    append_images=frames[1:],
    duration=120,
    loop=0,
)
print(f"saved hypergraph_evolution.gif  ({len(frames)} frames)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ts, spec_ent, color="tab:purple")
ax.set_xlabel("t")
ax.set_ylabel("spectral entropy of incidence matrix")
ax.set_title("Hypergraph spectral entropy over time  "
             "(groups with avg opinion > 0.05)")
fig.tight_layout()
fig.savefig("hypergraph_spectral_entropy.png")
plt.close(fig)
print("saved hypergraph_spectral_entropy.png")

def fit_and_plot_three(values, hist_color, xlabel, title_prefix, outpath,
                       empty_msg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if values:
        vals = np.asarray(values, dtype=float)
        vals = vals[vals > 0]
        n = len(vals)

        # Exponential MLE
        exp_scale = vals.mean()
        exp_ks = kstest(vals, "expon", args=(0.0, exp_scale))
        exp_ll = expon.logpdf(vals, loc=0.0, scale=exp_scale).sum()
        exp_aic = 2 * 1 - 2 * exp_ll

        # Power-law (Pareto) MLE with x_min = min(vals)
        x_min = vals.min()
        alpha = 1.0 + n / np.sum(np.log(vals / x_min))
        b = alpha - 1.0
        # KS / AIC restricted to vals >= x_min (which is all of them here)
        pl_ks = kstest(vals, "pareto", args=(b, 0.0, x_min))
        pl_ll = pareto.logpdf(vals, b, loc=0.0, scale=x_min).sum()
        pl_aic = 2 * 1 - 2 * pl_ll

        # Log-normal MLE
        log_vals = np.log(vals)
        mu = log_vals.mean()
        sigma = log_vals.std(ddof=0)
        ln_scale = np.exp(mu)
        ln_ks = kstest(vals, "lognorm", args=(sigma, 0.0, ln_scale))
        ln_ll = lognorm.logpdf(vals, sigma, loc=0.0, scale=ln_scale).sum()
        ln_aic = 2 * 2 - 2 * ln_ll

        xs = np.linspace(vals.min(), vals.max(), 400)
        exp_pdf = expon.pdf(xs, loc=0.0, scale=exp_scale)
        pl_pdf = pareto.pdf(xs, b, loc=0.0, scale=x_min)
        ln_pdf = lognorm.pdf(xs, sigma, loc=0.0, scale=ln_scale)

        exp_label = (f"Exp(λ={1/exp_scale:.3g})  "
                     f"KS p={exp_ks.pvalue:.2g}  AIC={exp_aic:.0f}")
        pl_label = (f"PowerLaw(α={alpha:.3g}, x_min={x_min:.0f})  "
                    f"KS p={pl_ks.pvalue:.2g}  AIC={pl_aic:.0f}")
        ln_label = (f"LogNormal(μ={mu:.3g}, σ={sigma:.3g})  "
                    f"KS p={ln_ks.pvalue:.2g}  AIC={ln_aic:.0f}")

        lin_bins = np.linspace(vals.min(), vals.max(), 31)
        log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 31)

        for ax, bins, scale_kind in [
            (axes[0], lin_bins, "linear"),
            (axes[1], log_bins, "log-log"),
        ]:
            ax.hist(vals, bins=bins, density=True, color=hist_color,
                    edgecolor="black", alpha=0.5)
            ax.plot(xs, exp_pdf, "--", color="tab:red", label=exp_label)
            ax.plot(xs, pl_pdf, "--", color="tab:blue", label=pl_label)
            ax.plot(xs, ln_pdf, "--", color="tab:green", label=ln_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.set_title(scale_kind)
            ax.legend(fontsize=8, loc="best")
            if scale_kind == "log-log":
                ax.set_xscale("log")
                ax.set_yscale("log")

        best_aic = min(exp_aic, pl_aic, ln_aic)
        winner = {exp_aic: "Exp", pl_aic: "PowerLaw", ln_aic: "LogNormal"}[best_aic]
        fig.suptitle(
            f"{title_prefix}  —  best by AIC: {winner}\n"
            f"n={n}, mean={vals.mean():.0f}, median={np.median(vals):.0f}"
        )
    else:
        for ax in axes:
            ax.text(0.5, 0.5, empty_msg,
                    ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


fit_and_plot_three(
    change_intervals,
    hist_color="tab:orange",
    xlabel="Δt between successive hypergraph changes",
    title_prefix="Inter-change-time distribution",
    outpath="hypergraph_change_intervals.png",
    empty_msg="no changes detected",
)
print(f"saved hypergraph_change_intervals.png  "
      f"({len(change_intervals)} change events)")

fit_and_plot_three(
    edge_lifetimes,
    hist_color="tab:green",
    xlabel="continuous lifetime of a hyperedge (Δt)",
    title_prefix="Hyperedge lifetime distribution",
    outpath="hypergraph_edge_lifetimes.png",
    empty_msg="no completed lifetimes detected",
)
print(f"saved hypergraph_edge_lifetimes.png  "
      f"({len(edge_lifetimes)} completed lifetimes; "
      f"{len(active_since)} still-active edges censored)")
