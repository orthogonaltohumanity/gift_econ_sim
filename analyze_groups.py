"""
Analysis of the gift-economy simulation.

Sign convention: in main.py, epsilon>0 means agent n gave a gift to group g
(gifts[n][g] += epsilon). So:
    accumulated sum_gift > 0  =>  net giver
    accumulated sum_gift < 0  =>  net taker
"""

import json
import glob
import re
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from PIL import Image
from tqdm import tqdm
import xgi


# ---------------------------------------------------------------------------
# Load files

files = glob.glob("data_*.json")
files.sort(key=lambda f: int(re.search(r"data_(\d+)\.json", f).group(1)))
last_path = files[-1]

with open(last_path) as f:
    last_packet = json.load(f)
opinions = last_packet["op"]
gifts = last_packet["gf"]
N = len(opinions)

groups = sorted(
    {g for a in opinions for g in a},
    key=lambda g: (len(g.split(",")), g),
)
group_members = {g: [int(m) for m in g.split(",")] for g in groups}
group_sizes = np.array([len(group_members[g]) for g in groups])
unique_sizes = sorted(set(group_sizes.tolist()))
groups_by_size = {s: [g for g in groups if len(group_members[g]) == s]
                  for s in unique_sizes}


# ---------------------------------------------------------------------------
# Per-group statistics (last timestep)

def per_group_stats(packet):
    ops = packet["op"]
    gfs = packet["gf"]
    keys = sorted(
        {g for a in ops for g in a},
        key=lambda g: (len(g.split(",")), g),
    )
    tg, cp, opavg = [], [], []
    for g in keys:
        members = [int(m) for m in g.split(",")]
        tg.append(sum(gfs[m][g] for m in members))
        probs = []
        for m in members:
            s = sum(ops[m].values())
            probs.append(ops[m][g] / s if s > 0 else 0.0)
        cp.append(np.mean(probs))
        opavg.append(np.mean([ops[m][g] for m in members]))
    sizes = np.array([len(g.split(",")) for g in keys])
    return np.array(tg), np.array(cp), np.array(opavg), sizes


total_gifts, avg_choice_prob, avg_opinion, _ = per_group_stats(last_packet)


# ---------------------------------------------------------------------------
# Static scatters (one per metric)

metrics = [
    ("choice_prob", avg_choice_prob, "avg P(member chooses group)", "choice prob"),
    ("opinion",     avg_opinion,     "avg opinion of group",        "avg opinion"),
]


def scatter_colored_by_size(y, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(total_gifts, y, c=group_sizes, cmap="viridis")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("sum of gifts in group  (>0 = givers, <0 = takers)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, ticks=unique_sizes)
    cbar.set_label("group size")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def scatter_by_size_grid(y, ylabel, title, outpath):
    ncols = 5
    nrows = int(np.ceil(len(unique_sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    axes = axes.reshape(-1)
    for ax, size in zip(axes, unique_sizes):
        m = group_sizes == size
        x, ys = total_gifts[m], y[m]
        ax.scatter(x, ys, color="tab:blue")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("sum of gifts in group")
        ax.set_ylabel(ylabel)
        if len(x) >= 2 and np.ptp(x) > 0:
            r = linregress(x, ys)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, r.slope * xs + r.intercept, color="tab:red")
            ax.set_title(f"size {size}  (n={len(x)})\n"
                         f"slope={r.slope:.2e}  R²={r.rvalue**2:.3f}")
        else:
            ax.set_title(f"size {size}  (n={len(x)})")
    for ax in axes[len(unique_sizes):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def scatter_with_marginals(x, y, ylabel, title, outpath, fit=True):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(x, y, color="tab:blue")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("sum of gifts in group  (>0 = givers, <0 = takers)")
    ax.set_ylabel(ylabel)
    if fit and len(x) >= 2 and np.ptp(x) > 0:
        r = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, r.slope * xs + r.intercept, color="tab:red",
                label=f"slope={r.slope:.2e}  R²={r.rvalue**2:.3f}")
        ax.legend(loc="best")
    ax_top.hist(x, bins=30, color="tab:blue", edgecolor="black")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("count")
    ax_right.hist(y, bins=30, orientation="horizontal",
                  color="tab:blue", edgecolor="black")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("count")
    fig.suptitle(title)
    fig.savefig(outpath)
    plt.close(fig)


for tag, y, ylabel, short in metrics:
    scatter_colored_by_size(y, ylabel,
        f"{last_path} — {short} vs total in-group gifts",
        f"groups_scatter_{tag}.png")
    scatter_by_size_grid(y, ylabel,
        f"{last_path} — {short} vs total gifts, by group size",
        f"groups_scatter_by_size_{tag}.png")
    scatter_with_marginals(total_gifts, y, ylabel,
        f"{last_path} — {short} vs total gifts, with marginals",
        f"groups_scatter_marginals_{tag}.png", fit=False)
    for size in unique_sizes:
        m = group_sizes == size
        scatter_with_marginals(total_gifts[m], y[m], ylabel,
            f"{last_path} — {short}, size {size}  (n={m.sum()})",
            f"groups_scatter_marginals_{tag}_size{size}.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=30, color="tab:blue", edgecolor="black")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("number of groups")
    ax.set_title(f"{last_path} — distribution of {short}")
    fig.tight_layout()
    fig.savefig(f"groups_{tag}_hist.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time-series load (per-group and per-agent trajectories)

per_group_x = {g: [] for g in groups}
per_group_y_cp = {g: [] for g in groups}
per_group_y_op = {g: [] for g in groups}
per_agent_total = []
per_agent_entropy = []
per_agent_others = []
ts = []
total_abs_debt = []

for fp in tqdm(files, desc="loading snapshots"):
    with open(fp) as f:
        pkt = json.load(f)
    ts.append(pkt["t"])
    ops = pkt["op"]
    gfs = pkt["gf"]
    total_abs_debt.append(sum(abs(v) for a in gfs for v in a.values()))

    for g in groups:
        members = group_members[g]
        per_group_x[g].append(sum(gfs[m][g] for m in members))
        probs = []
        for m in members:
            s = sum(ops[m].values())
            probs.append(ops[m][g] / s if s > 0 else 0.0)
        per_group_y_cp[g].append(np.mean(probs))
        per_group_y_op[g].append(np.mean([ops[m][g] for m in members]))

    own = np.zeros(N)
    others = np.zeros(N)
    ent = np.zeros(N)
    for n in range(N):
        own[n] = sum(gfs[n].values())
        vals = np.array(list(ops[n].values()), dtype=float)
        s = vals.sum()
        if s > 0:
            p = vals / s
            p = p[p > 0]
            ent[n] = -np.sum(p * np.log(p))
        for g in ops[n]:
            for m in group_members[g]:
                if m != n:
                    others[n] += gfs[m][g]
    per_agent_total.append(own)
    per_agent_entropy.append(ent)
    per_agent_others.append(others)

ts_arr = np.array(ts, dtype=float)
debt_arr = np.array(total_abs_debt, dtype=float)
per_agent_total = np.array(per_agent_total)        # (T, N)
per_agent_entropy = np.array(per_agent_entropy)    # (T, N)
per_agent_others = np.array(per_agent_others)      # (T, N)


# ---------------------------------------------------------------------------
# Total absolute debt over time + power-law fit

mask = (ts_arr > 0) & (debt_arr > 0)
res = linregress(np.log(ts_arr[mask]), np.log(debt_arr[mask]))
b = res.slope
a = np.exp(res.intercept)
fit = a * ts_arr[mask] ** b

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ts, total_abs_debt, label="data")
ax.plot(ts_arr[mask], fit, "--", color="tab:red",
        label=f"fit: {a:.3g}·t^{b:.3f}  (R²={res.rvalue**2:.3f})")
ax.set_xlabel("t")
ax.set_ylabel("sum |gifts[n][g]|")
ax.set_title("Total absolute debt over time")
ax.legend()
fig.tight_layout()
fig.savefig("total_abs_debt.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(ts_arr[mask], debt_arr[mask], label="data")
ax.loglog(ts_arr[mask], fit, "--", color="tab:red",
          label=f"fit: {a:.3g}·t^{b:.3f}  (R²={res.rvalue**2:.3f})")
ax.set_xlabel("t")
ax.set_ylabel("sum |gifts[n][g]|")
ax.set_title("Total absolute debt over time (log-log)")
ax.legend()
fig.tight_layout()
fig.savefig("total_abs_debt_loglog.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Group trajectories: 3 samples per size in (gift-sum, choice-prob) phase plane

rng = np.random.default_rng(0)
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=min(unique_sizes), vmax=max(unique_sizes))

fig, ax = plt.subplots(figsize=(9, 7))
for size in unique_sizes:
    pool = groups_by_size[size]
    sample = rng.choice(pool, size=min(3, len(pool)), replace=False)
    for g in sample:
        xs = per_group_x[g]
        ys = per_group_y_cp[g]
        ax.plot(xs, ys, color=cmap(norm(size)), alpha=0.85, linewidth=1.2)
        ax.scatter([xs[0]], [ys[0]], color=cmap(norm(size)),
                   marker="o", s=30, edgecolor="black", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], color=cmap(norm(size)),
                   marker="X", s=60, edgecolor="black", zorder=3)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("sum of gifts in group  (>0 = givers, <0 = takers)")
ax.set_ylabel("avg P(member chooses group)")
ax.set_title("3 sampled trajectories per group size  (○ = start, ✕ = end)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, ticks=unique_sizes)
cbar.set_label("group size")
fig.tight_layout()
fig.savefig("groups_trajectories.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Per-group: time spent above/below 0; zero-crossings

T = len(files)
time_above = np.array([int((np.array(per_group_x[g]) > 0).sum()) for g in groups])
time_below = np.array([int((np.array(per_group_x[g]) < 0).sum()) for g in groups])
zero_crossings = np.zeros(len(groups), dtype=int)
for i, g in enumerate(groups):
    s = np.sign(per_group_x[g])
    s = s[s != 0]
    zero_crossings[i] = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0


def overlay_hist(values_blue, values_red, label_b, label_r, bins, xlabel,
                 title, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values_blue, bins=bins, color="tab:blue", alpha=0.6,
            edgecolor="black", label=label_b)
    ax.hist(values_red, bins=bins, color="tab:red", alpha=0.6,
            edgecolor="black", label=label_r)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of groups")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


overlay_hist(time_above, time_below, "time as giver (sum>0)", "time as taker (sum<0)",
             np.linspace(0, T, 31), "number of snapshots",
             "Time each group spends as giver / taker (all sizes)",
             "groups_time_sign_hist.png")

ncols = 5
nrows = int(np.ceil(len(unique_sizes) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         squeeze=False)
axes = axes.reshape(-1)
bins = np.linspace(0, T, 31)
for ax, size in zip(axes, unique_sizes):
    keys = groups_by_size[size]
    a_arr = np.array([time_above[groups.index(g)] for g in keys])
    b_arr = np.array([time_below[groups.index(g)] for g in keys])
    ax.hist(a_arr, bins=bins, color="tab:blue", alpha=0.6,
            edgecolor="black", label="giver")
    ax.hist(b_arr, bins=bins, color="tab:red", alpha=0.6,
            edgecolor="black", label="taker")
    ax.set_xlabel("snapshots")
    ax.set_ylabel("# groups")
    ax.set_title(f"size {size}  (n={len(keys)})")
    ax.legend(fontsize=8)
for ax in axes[len(unique_sizes):]:
    ax.axis("off")
fig.suptitle("Time spent giving / taking, by group size")
fig.tight_layout()
fig.savefig("groups_time_sign_hist_by_size.png")
plt.close(fig)

max_c = max(zero_crossings.max(), 1)
zc_bins = np.arange(0, max_c + 2) - 0.5
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(zero_crossings, bins=zc_bins, color="tab:purple", edgecolor="black")
ax.set_xlabel("number of zero-crossings of sum-gift over time")
ax.set_ylabel("number of groups")
ax.set_title("Zero-crossings of sum-gift per group (all sizes)")
fig.tight_layout()
fig.savefig("groups_zero_crossings_hist.png")
plt.close(fig)

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                         squeeze=False)
axes = axes.reshape(-1)
for ax, size in zip(axes, unique_sizes):
    keys = groups_by_size[size]
    cs = np.array([zero_crossings[groups.index(g)] for g in keys])
    ax.hist(cs, bins=zc_bins, color="tab:purple", edgecolor="black")
    ax.set_xlabel("zero-crossings")
    ax.set_ylabel("# groups")
    ax.set_title(f"size {size}  (n={len(keys)}, mean={cs.mean():.1f})")
for ax in axes[len(unique_sizes):]:
    ax.axis("off")
fig.suptitle("Zero-crossings of sum-gift over time, by group size")
fig.tight_layout()
fig.savefig("groups_zero_crossings_hist_by_size.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# Group-level scatter+marginals GIFs (choice prob and opinion)

all_x = [v for series in per_group_x.values() for v in series]
x_lo = min(all_x) - 0.05 * (max(all_x) - min(all_x))
x_hi = max(all_x) + 0.05 * (max(all_x) - min(all_x))
y_lo, y_hi = 0.0, 1.0
x_bins = np.linspace(x_lo, x_hi, 31)
y_bins = np.linspace(y_lo, y_hi, 31)


def render_groups_gif(per_group_y, ylabel, outpath, desc):
    frs = []
    for i, t in enumerate(tqdm(ts, desc=desc)):
        tg = np.array([per_group_x[g][i] for g in groups])
        yv = np.array([per_group_y[g][i] for g in groups])

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        sc = ax.scatter(tg, yv, c=group_sizes, cmap="viridis",
                        vmin=min(unique_sizes), vmax=max(unique_sizes))
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("sum of gifts in group  (>0 = givers, <0 = takers)")
        ax.set_ylabel(ylabel)
        ax_top.hist(tg, bins=x_bins, color="tab:blue", edgecolor="black")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_ylabel("count")
        ax_right.hist(yv, bins=y_bins, orientation="horizontal",
                      color="tab:blue", edgecolor="black")
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_xlabel("count")
        cbar = fig.colorbar(sc, ax=ax_right, ticks=unique_sizes)
        cbar.set_label("group size")
        fig.suptitle(f"t = {t}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frs.append(Image.open(buf).convert("RGB"))
    frs[0].save(outpath, save_all=True, append_images=frs[1:],
                duration=120, loop=0)


render_groups_gif(per_group_y_cp, "avg P(member chooses group)",
                  "groups_scatter_marginals.gif",
                  "rendering groups gif (choice prob)")
render_groups_gif(per_group_y_op, "avg opinion of group",
                  "groups_scatter_marginals_opinion.gif",
                  "rendering groups gif (opinion)")


# ---------------------------------------------------------------------------
# Per-agent analyses

# Static: entropy vs sum_gift at last timestep
ent_last = per_agent_entropy[-1]
own_last = per_agent_total[-1]
others_last = per_agent_others[-1]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(own_last, ent_last, c=np.arange(N), cmap="tab10", s=80,
                edgecolor="black")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("agent's sum of gifts  (>0 = giver, <0 = taker)")
ax.set_ylabel("entropy of group-choice probabilities")
ax.set_title(f"{last_path} — agents: choice entropy vs total gifts")
cbar = fig.colorbar(sc, ax=ax, ticks=range(N))
cbar.set_label("agent id")
fig.tight_layout()
fig.savefig("agents_entropy_vs_gifts.png")
plt.close(fig)


# Agent classification: zero-crossings + mean sign
crossings_per_agent = np.zeros(N, dtype=int)
mean_sign = np.zeros(N)
for n in range(N):
    s = np.sign(per_agent_total[:, n])
    s2 = s[s != 0]
    crossings_per_agent[n] = int((np.diff(s2) != 0).sum()) if len(s2) > 1 else 0
    mean_sign[n] = s.mean()

threshold = np.median(crossings_per_agent)
labels, colors = [], []
for n in range(N):
    if crossings_per_agent[n] > threshold:
        labels.append("inbetween"); colors.append("tab:gray")
    elif mean_sign[n] > 0:
        labels.append("giver");     colors.append("tab:blue")
    else:
        labels.append("taker");     colors.append("tab:red")

print("\nagent  crossings  mean_sign  category")
for n in range(N):
    print(f"  {n:>2}     {crossings_per_agent[n]:>4}     "
          f"{mean_sign[n]:+.2f}      {labels[n]}")

handles = [
    plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="giver (low crossings, +)"),
    plt.Rectangle((0, 0), 1, 1, color="tab:red", label="taker (low crossings, −)"),
    plt.Rectangle((0, 0), 1, 1, color="tab:gray", label="inbetween (high crossings)"),
]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(N), crossings_per_agent, color=colors, edgecolor="black")
ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8,
           label=f"median = {threshold:.0f}")
ax.set_xticks(range(N))
ax.set_xlabel("agent id")
ax.set_ylabel("# zero-crossings of total gift")
ax.set_title("Givers / Takers / Inbetweens by zero-crossings")
ax.legend(handles=handles, loc="best")
fig.tight_layout()
fig.savefig("agents_crossings.png")
plt.close(fig)


# Outgoing vs incoming (last timestep)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(own_last, others_last, c=colors, s=120, edgecolor="black")
for n in range(N):
    ax.annotate(str(n), (own_last[n], others_last[n]),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.axvline(0, color="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.5)
lo = min(own_last.min(), others_last.min())
hi = max(own_last.max(), others_last.max())
ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("agent's own total gifts  (>0 = giver)")
ax.set_ylabel("total gifts from others into agent's groups")
ax.set_title(f"{last_path} — outgoing vs incoming (per agent)")
ax.legend(handles=handles + [plt.Line2D([0], [0], color="gray", linestyle=":",
          label="y = x")], loc="best", fontsize=9)
fig.tight_layout()
fig.savefig("agents_outgoing_vs_incoming.png")
plt.close(fig)


# Entropy vs sum_gift GIF
ax_xlo = per_agent_total.min() - 0.05 * (per_agent_total.max() - per_agent_total.min())
ax_xhi = per_agent_total.max() + 0.05 * (per_agent_total.max() - per_agent_total.min())
ax_yhi = per_agent_entropy.max() * 1.05
ax_ylo = -0.02 * ax_yhi

frames = []
for i, t in enumerate(tqdm(ts, desc="rendering agents-entropy gif")):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(per_agent_total[i], per_agent_entropy[i],
                    c=np.arange(N), cmap="tab10", s=80,
                    edgecolor="black", vmin=0, vmax=N - 1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(ax_xlo, ax_xhi)
    ax.set_ylim(ax_ylo, ax_yhi)
    ax.set_xlabel("agent's sum of gifts  (>0 = giver, <0 = taker)")
    ax.set_ylabel("entropy of group-choice probabilities")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(N))
    cbar.set_label("agent id")
    ax.set_title(f"agents: choice entropy vs total gifts  —  t = {t}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
frames[0].save("agents_entropy_vs_gifts.gif", save_all=True,
               append_images=frames[1:], duration=120, loop=0)


# Out vs in GIF
xlo = per_agent_total.min() - 0.05 * (per_agent_total.max() - per_agent_total.min())
xhi = per_agent_total.max() + 0.05 * (per_agent_total.max() - per_agent_total.min())
ylo = per_agent_others.min() - 0.05 * (per_agent_others.max() - per_agent_others.min())
yhi = per_agent_others.max() + 0.05 * (per_agent_others.max() - per_agent_others.min())

frames = []
for i, t in enumerate(tqdm(ts, desc="rendering out-vs-in gif")):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(per_agent_total[i], per_agent_others[i],
               c=colors, s=120, edgecolor="black")
    for n in range(N):
        ax.annotate(str(n), (per_agent_total[i, n], per_agent_others[i, n]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    diag_lo = min(xlo, ylo); diag_hi = max(xhi, yhi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("agent's own total gifts  (>0 = giver)")
    ax.set_ylabel("total gifts from others into agent's groups")
    ax.set_title(f"outgoing vs incoming  —  t = {t}")
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
frames[0].save("agents_outgoing_vs_incoming.gif", save_all=True,
               append_images=frames[1:], duration=120, loop=0)


# ---------------------------------------------------------------------------
# Hypergraph of dominant groups (last timestep): avg opinion > mean + 2*std

threshold_op = np.mean(avg_opinion) + 2 * np.std(avg_opinion)
dominant = [g for g, a in zip(groups, avg_opinion) if a > threshold_op]
print(f"\nthreshold (mean + 2*std) = {threshold_op:.4f}")
print(f"dominant groups ({len(dominant)}):")
for g, a in zip(groups, avg_opinion):
    if a > threshold_op:
        print(f"  {{{g}}}  avg_opinion={a:.4f}")

H = xgi.Hypergraph()
H.add_nodes_from(range(N))
H.add_edges_from([group_members[g] for g in dominant])

fig, ax = plt.subplots(figsize=(8, 8))
xgi.draw(H, ax=ax)
ax.set_title(f"{last_path} — dominant groups "
             f"(avg opinion > mean+2*std = {threshold_op:.3f})")
fig.tight_layout()
fig.savefig("dominant_hypergraph.png")
plt.close(fig)
