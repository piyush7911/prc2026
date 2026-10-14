"""Draw the paper's figures as vector PDF into docs/paper/figures/ from docs/paper/figure_data.json and the documented pipeline numbers.
Usage (repository root): python docs/paper/make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
D = json.load(open(os.path.join(HERE, "figure_data.json")))
W = 6.4                                                     # printed text width, inches

# Okabe-Ito colour-blind-safe palette, one colour per pipeline part
C = dict(data="#4D4D4D", feat="#E9ECEF", base="#0072B2", adsb="#009E73", nb="#E69F00", daily="#CC79A7", bounds="#7F7F7F",
         reject="#D55E00", ink="#222222", light="#F4F6F8", grid="#D9D9D9")
LANE = dict(covered="#009E73", uncovered="#E69F00", unm_nr="#D55E00", unm_rome="#8C2D04")
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink"], "axes.labelcolor": C["ink"], "xtick.color": C["ink"],
    "ytick.color": C["ink"], "text.color": C["ink"], "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "custom", "mathtext.rm": "Helvetica", "mathtext.it": "Helvetica:italic", "mathtext.bf": "Helvetica:bold",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig); print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=7.5, bold=False, lw=0.8, r=0.6, ls="-", ha="center", va="center", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc, lw=lw, ls=ls, zorder=z))
    tx = x + w / 2 if ha == "center" else x + 0.8
    ax.text(tx, y + h / 2 if va == "center" else y + h - 0.8, text, ha=ha, va=va, color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=z + 1, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=None, lw=0.8, ls="-", style="-|>", conn="arc3", z=1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=z,
                arrowprops=dict(arrowstyle=style, color=color or C["ink"], lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7, connectionstyle=conn))


def canvas(w, h, xmax, ymax):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off"); return fig, ax


def panel(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------ Figure 1: pipeline
def fig1():
    fig, ax = canvas(W, 3.9, 100, 61)
    for x, t in ((0.5, "Inputs"), (22, "Features"), (43, "Stage 0: base"), (68, "Corrections and constraints")):
        ax.text(x, 59.6, t, fontsize=7.6, fontweight="bold", va="top")
    inputs = ["Movement records\n2025 with labels\n2026 to predict", "Network Manager clocks\nEOBT$_1$ · LOBT · IOBT\nAOBT$_3$",
              "METAR weather\n(Iowa Environmental\nMesonet)", "adsb.lol ADS-B\ndaily archives\n(open, ODbL)", "EUROCONTROL daily\npre-departure delay\n(published)"]
    ys = [47, 37, 27, 17, 7]
    for y, t in zip(ys, inputs):
        box(ax, 0.5, y, 18.5, 8.4, t, C["data"], fs=6.1)
    box(ax, 22, 31.5, 18, 23.9, "Feature table\n≈ 240 columns\n\ncalendar · aircraft\nstand · runway\nweather · traffic\nqueue state · arrivals\ndefect rates\ntarget encodings",
        C["feat"], ec="#B8BEC4", tc=C["ink"], fs=6.1)
    box(ax, 22, 17, 18, 12.4, "ADS-B ground-track\ntiming\n\nleft the assigned stand\nfirst movement · last stop\nstops · coverage", "#E3F2EC", ec=C["adsb"], tc=C["ink"], fs=6.1)
    arrow(ax, 19, 51.2, 22, 51.2); arrow(ax, 19, 41.2, 22, 41.2); arrow(ax, 19, 31.2, 22, 33.5); arrow(ax, 19, 21.2, 22, 22.5)
    # stage 0
    ax.add_patch(FancyBboxPatch((43, 17), 21, 38.4, boxstyle="round,pad=0,rounding_size=0.8", fc="#E6F0F8", ec=C["base"], lw=0.9, zorder=1))
    ax.text(53.5, 53.6, "residual to the best clock, $y - b$", ha="center", va="top", fontsize=6.1, color=C["base"])
    for i, t in enumerate(["LightGBM, lr 0.05", "LightGBM, lr 0.02", "CatBoost, depth 8"]):
        box(ax, 45, 44.5 - i * 5.4, 17, 4.4, t, C["base"], fs=6.1)
    ax.text(53.5, 30.2, "+ 4 decorrelators\n(other clock anchors, 0.05 each)", ha="center", va="center", fontsize=5.9)
    ax.text(53.5, 23.2, "+ mixed-target members\nfor $d \\leq 1$ h: CatBoost 0.5,\nLightGBM 0.2", ha="center", va="center", fontsize=5.9)
    arrow(ax, 40, 43.5, 43, 43.5); arrow(ax, 40, 26, 43, 26)
    # corrections column
    stages = [("Stage 1: ADS-B stacker (covered rows)", C["adsb"]), ("Stage 1b: ADS-B corrector (no NM clock)", C["adsb"]),
              ("Stage 2: runway and airport neighbours", C["nb"]), ("Stage 2b: Istanbul runway queue", C["nb"]),
              ("Stage 3: daily reconciliation", C["daily"]), ("Stage 4: assigned-stand readout", C["adsb"]), ("Bounds: join window, airport maxima", C["bounds"])]
    sy = [51.6, 46.6, 41.6, 36.6, 31.6, 26.6, 21.6]
    for y, (t, col) in zip(sy, stages):
        box(ax, 68, y, 31.5, 4.2, t, col, fs=6.1)
    for y0, y1 in zip(sy[:-1], sy[1:]):
        arrow(ax, 83.75, y0, 83.75, y1 + 4.2, lw=0.6)
    arrow(ax, 64, 53.2, 68, 53.2)
    box(ax, 68, 15.2, 31.5, 4.8, "Prediction for every departure", "white", ec=C["ink"], tc=C["ink"], fs=6.4, bold=True)
    arrow(ax, 83.75, 21.6, 83.75, 20.0, lw=0.6)
    # ADS-B timing -> stages 1 and 1b, routed below stage 0 through the channel at x = 65.2
    ax.plot([31, 31, 65.2, 65.2], [17, 13.6, 13.6, 47.6], color=C["adsb"], lw=0.7, zorder=1)
    arrow(ax, 65.2, 47.6, 68, 47.6, color=C["adsb"], lw=0.7)
    ax.plot([65.2, 65.2], [47.6, 53.2], color=C["adsb"], lw=0.7, zorder=1)
    # EUROCONTROL -> stage 3, channel at x = 66.6
    ax.plot([19, 66.6, 66.6], [11.2, 11.2, 33.7], color=C["daily"], lw=0.7, zorder=1)
    arrow(ax, 66.6, 33.7, 68, 33.7, color=C["daily"], lw=0.7)
    # gate band
    ax.add_patch(Rectangle((0.5, 0.4), 99, 3.0, fc=C["light"], ec="#C9CED3", lw=0.6, zorder=0))
    ax.text(50, 1.9, "Gate on the 2025 out-of-fold predictions: every analogue month better · weight checked fold by fold · "
            "100 most-improved rows < 100 % of the gain", ha="center", va="center", fontsize=5.8)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ Figure 2: clocks of one real departure
def fig2():
    E = D["example"]; ck = E["clocks_s"]; ev = E["adsb_events_s"]; m = lambda v: v / 60.0
    fig, ax = plt.subplots(figsize=(W, 2.45))
    ax.set_xlim(-21.6, 1.2); ax.set_ylim(0, 6.1); ax.spines["left"].set_visible(False); ax.set_yticks([])
    ax.set_xlabel(f"minutes relative to take-off (a Zurich departure, {E['month']}; take-off at {E['takeoff_utc']} UTC)")
    def mark(v, h, col, ls="-"):
        x = m(v); ax.plot([x, x], [0, h], color=col, lw=1.0, ls=ls, zorder=2, clip_on=False); ax.plot(x, 0, "o", ms=3.6, color=col, zorder=3, clip_on=False)
    mark(ck["schedule"], 2.05, "#7F7F7F"); ax.text(m(ck["schedule"]) + 0.15, 1.95, "schedule = EOBT$_1$ = LOBT = IOBT", fontsize=6.6, color="#7F7F7F", va="top")
    mark(ck["AOBT3"], 2.05, C["base"]); ax.text(m(ck["AOBT3"]) + 0.15, 1.95, "NM actual off-block AOBT$_3$", fontsize=6.6, color=C["base"], va="top")
    mark(ck["recorded_offblock"], 1.25, C["ink"], "--")
    ax.text(m(ck["recorded_offblock"]) - 0.18, 1.2, "recorded off-block\n(the label)", fontsize=6.6, ha="right", va="top")
    mark(ev["ads_dwell_end"], 1.25, C["adsb"])
    off = ev["ads_dwell_end"] - ck["recorded_offblock"]
    ax.text(m(ev["ads_dwell_end"]) + 0.18, 1.2, f"ADS-B: end of\nlast stop ({off:+.0f} s)", fontsize=6.6, color=C["adsb"], va="top",
            bbox=dict(fc="white", ec="none", pad=0.5), zorder=4)
    mark(0.0, 1.25, C["ink"]); ax.text(-0.18, 1.2, "take-off", fontsize=6.6, ha="right", va="top")
    def bracket(x0, x1, yy, txt, col):
        ax.annotate("", xy=(x0, yy), xytext=(x1, yy), arrowprops=dict(arrowstyle="<->", color=col, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6))
        ax.text((x0 + x1) / 2, yy + 0.12, txt, ha="center", va="bottom", fontsize=6.6, color=col)
    bracket(m(ck["schedule"]), 0, 3.05, f"d = take-off − schedule = {abs(ck['schedule']):.0f} s", "#7F7F7F")
    bracket(m(ck["recorded_offblock"]), 0, 4.0, f"y = take-off − recorded off-block = {E['y_s']:.0f} s (the label)", C["ink"])
    bracket(m(ck["AOBT3"]), 0, 4.95, f"b = take-off − AOBT$_3$ = {abs(ck['AOBT3']):.0f} s;  residual y − b = {E['y_s'] - abs(ck['AOBT3']):+.0f} s", C["base"])
    save(fig, "fig2_clocks")


# ------------------------------------------------------------------ Figure 3: lanes
def fig3():
    n = D["lanes_2026"]; R = D["lanes_rmse"]
    fig, ax = canvas(W, 3.0, 100, 47)
    box(ax, 36, 39.5, 28, 5.5, f"Departure  ({sum(n.values()):,} in the 2026 file)", "white", ec=C["ink"], tc=C["ink"], fs=7, bold=True)
    q = lambda x, y, t: box(ax, x, y, 22, 4.6, t, C["light"], ec="#B8BEC4", tc=C["ink"], fs=6.8)
    q(39, 31.5, "NM clock present?"); q(16, 23, "own ADS-B ground track seen?"); q(62, 23, "Rome (LIRF)?")
    arrow(ax, 50, 39.5, 50, 36.1)
    arrow(ax, 44, 31.5, 27, 27.6); arrow(ax, 56, 31.5, 73, 27.6)
    ax.text(33.5, 30.4, "yes", fontsize=6.5, ha="center"); ax.text(66.5, 30.4, "no", fontsize=6.5, ha="center")
    leaves = [("covered", 1, "ADS-B stacker · neighbours\nstand readout"), ("uncovered", 26, "neighbours · Istanbul queue"),
              ("unm_nr", 51, "ADS-B corrector where seen"), ("unm_rome", 76, "two-mode daily posterior")]
    for (k, x, extra) in leaves:
        col = LANE[k]
        box(ax, x, 9.5, 23, 9.5, f"{k}\n{n[k]:,} departures\n{extra}", col, fs=6.6)
        vals = " · ".join(f"{R[m][k]:,.0f}" for m in ("1", "2", "6", "7"))
        ax.text(x + 11.5, 5.4, vals, ha="center", fontsize=6.8, color=col, fontweight="bold")
    arrow(ax, 21, 23, 12.5, 19); arrow(ax, 33, 23, 37.5, 19); arrow(ax, 67, 23, 62.5, 19); arrow(ax, 79, 23, 87.5, 19)
    for x, t in ((15.5, "yes"), (36.8, "no"), (63.8, "no"), (86.0, "yes")):
        ax.text(x, 21.3, t, fontsize=6.5, ha="center")
    ax.text(50, 0.3, "Coloured numbers: 2025 out-of-fold RMSE of the lane in seconds (January · February · June · July).\n"
            "All lanes also receive the base learners and members, the daily reconciliation and the bounds.", ha="center", fontsize=6.0,
            color="#555555", linespacing=1.3)
    save(fig, "fig3_lanes")


# ------------------------------------------------------------------ Figure 4: ADS-B chain and a real ground-speed trace
def fig4():
    E = D["example"]; T = E["track"]; ev = E["adsb_events_s"]; ck = E["clocks_s"]
    fig = plt.figure(figsize=(W, 3.8)); gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 2.2], hspace=0.22)
    ax0 = fig.add_subplot(gs[0]); ax0.set_xlim(0, 100); ax0.set_ylim(0, 14); ax0.axis("off")
    steps = ["Download\nadsb.lol daily archives\n~3 GB per day, 485 days",
             "Extract\npositions within ±0.15°\nof each airport",
             "Match to the departure\nby callsign, aircraft\naddress or lift-off time",
             "Features per departure\nleft the assigned stand\nfirst movement · last stop"]
    for i, t in enumerate(steps):
        box(ax0, 1 + i * 25, 1, 22, 12, t, "#E3F2EC", ec=C["adsb"], tc=C["ink"], fs=6.2)
        if i < 3: arrow(ax0, 23 + i * 25, 7, 26 + i * 25, 7, color=C["adsb"])
    ax = fig.add_subplot(gs[1]); t = np.array(T["t_s"]) / 60; g = np.array(T["gs_kn"])
    ax.plot(t, np.minimum(g, 45), color=C["adsb"], lw=1.1, zorder=3); ax.plot(t, np.minimum(g, 45), "o", ms=1.6, color=C["adsb"], zorder=3)
    ax.set_xlim(-31, 0.8); ax.set_ylim(0, 47); ax.set_ylabel("ground speed (kn)"); ax.set_xlabel("minutes relative to take-off")
    ax.text(-0.7, 44.5, "take-off roll (clipped)", fontsize=6.2, ha="right", va="top", color="#555555", bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    st = (g <= 2) & (t > ev["ads_dwell_end"] / 60 + 0.2) & (t < -0.6); i = 0      # holds during the taxi, from the track itself
    while i < len(t):
        if st[i]:
            j = i
            while j + 1 < len(t) and st[j + 1]: j += 1
            if (t[j] - t[i]) * 60 >= 60:
                ax.annotate("", xy=(t[i], 3.2), xytext=(t[j], 3.2), arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=5))
                lab = f"hold {(t[j] - t[i]) * 60:.0f} s" if (t[j] - t[i]) > 3 else f"{(t[j] - t[i]) * 60:.0f} s"
                ax.text((t[i] + t[j]) / 2, 5.2, lab, fontsize=6.0, ha="center", color="#555555",
                        bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
            i = j + 1
        else: i += 1
    ax.axvspan(-31, ev["ads_dwell_end"] / 60, color=C["light"], zorder=0); ax.text(-30.5, 4.0, "at the stand", fontsize=6.6, color="#555555")
    # four events lie within about one minute of each other: they are named in a legend in the empty area at the stand, not next to the lines
    near = [(ck["recorded_offblock"], "recorded off-block (label)", C["ink"], "--", 1.0), (ev.get("ads_as_leave"), "left the assigned stand", C["adsb"], "-", 1.3),
            (ev["ads_dwell_end"], "end of the last stop", "#7FBFA6", (0, (5, 1.5)), 1.0), (ev["ads_first_mov"], "first movement > 5 kn", C["adsb"], ":", 1.1)]
    hs = []
    for s_, lab, col, ls, lw in near:
        if s_ is None: continue
        hs.append(ax.axvline(s_ / 60, color=col, lw=lw, ls=ls, zorder=2, label=lab))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.012, 0.985), fontsize=6.4, frameon=True, framealpha=1.0, edgecolor="#C9CED3",
              handlelength=3.2, borderpad=0.6, labelspacing=0.45)
    for s_, lab, col, ls, ty in [(ck["AOBT3"], "NM AOBT$_3$", C["base"], "-.", 13), (ev["ads_first_fast"], "first fast > 15 kn", "#7FBFA6", ":", 31)]:
        x = s_ / 60; ax.axvline(x, color=col, lw=0.9, ls=ls, zorder=2)
        ax.text(x + 0.25, ty, lab, fontsize=6.4, color=col, va="center", ha="left", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    for a_, letter in ((ax0, "a"), (ax, "b")):     # both letters at the same distance from the left edge of the figure
        fig.text(0.012, a_.get_position().y1 + 0.004, letter, fontsize=9.5, fontweight="bold", va="bottom", ha="left")
    save(fig, "fig4_adsb")


# ------------------------------------------------------------------ Figure 5: daily total identity and gain
def fig5():
    I = D["identity"]; G = D["daily_gain"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))
    x = np.array(I["label_sum_min"]); y = np.array(I["published_min"]); keep = (x > 0) & (y > 0)
    a.scatter(x[keep], y[keep], s=4, color=C["daily"], alpha=0.45, lw=0, rasterized=True)
    lo, hi = 200, max(x.max(), y.max()) * 1.3; a.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.7, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel(r"$\sum\ \max(0,\ d - y)$ from 2025 labels (min)"); a.set_ylabel("published DLY_ALL_PRE_2 (min)")
    a.text(0.04, 0.95, f"{keep.sum():,} airport-days\ncorrelation {I['corr']:.4f}\nmedian ratio {I['median_ratio']:.4f}", transform=a.transAxes,
           fontsize=6.8, va="top")
    panel(a, "a")
    labels = ["Jan", "Feb", "Jun", "Jul", "four\nmonths"]; vals = [G[k] for k in ("1", "2", "6", "7", "four")]
    bars = b.bar(range(5), vals, color=[C["daily"]] * 4 + ["#8E4F78"], width=0.62)
    b.axhline(0, color=C["ink"], lw=0.6); b.set_xticks(range(5)); b.set_xticklabels(labels)
    b.set_ylabel("stage 3: change of 2025 out-of-fold RMSE (s)"); b.set_ylim(min(vals) * 1.18, 1.0)
    for r, v in zip(bars, vals): b.text(r.get_x() + r.get_width() / 2, v - 0.4, f"{v:.1f}".replace("-", "\u2212"), ha="center", va="top", fontsize=6.6)
    panel(b, "b")
    save(fig, "fig5_daily")


# ------------------------------------------------------------------ Figure 6: folds and gate
def fig6():
    fig = plt.figure(figsize=(W, 3.75))
    a = fig.add_axes([0.0, 0.47, 1.0, 0.53]); a.set_xlim(-2.0, 15.6); a.set_ylim(-0.25, 7.1); a.axis("off")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(); folds = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
    for j, mth in enumerate(months):
        a.text(j + 0.5, 6.45, mth, ha="center", va="center", fontsize=6.8, fontweight="bold" if j in (0, 1, 5, 6) else "normal")
    for i, fo in enumerate(folds):
        y = 5 - i
        a.text(-0.2, y + 0.42, f"fold {i + 1}", ha="right", va="center", fontsize=6.8)
        for j in range(12):
            a.add_patch(Rectangle((j + 0.05, y + 0.07), 0.9, 0.7, fc=C["base"] if j in fo else "#DCE8F3", ec="none"))
    for j in (0, 1, 5, 6):
        a.add_patch(Rectangle((j, -0.03), 1, 6.85, fc="none", ec=C["reject"], lw=1.0, zorder=5))
    a.add_patch(Rectangle((12.45, 4.55), 0.4, 0.4, fc=C["base"])); a.text(13.0, 4.75, "held out", fontsize=6.5, va="center")
    a.add_patch(Rectangle((12.45, 3.85), 0.4, 0.4, fc="#DCE8F3")); a.text(13.0, 4.05, "training", fontsize=6.5, va="center")
    a.add_patch(Rectangle((12.45, 3.15), 0.4, 0.4, fc="none", ec=C["reject"], lw=1.0)); a.text(13.0, 3.35, "analogues of the\n2026 test months", fontsize=6.5, va="center")
    a.text(12.45, 1.6, "Out-of-fold predictions\nof each stage are the\ntraining input of the next.", fontsize=6.2, va="center", color="#555555")
    a.text(-2.0, 7.0, "a", fontsize=9.5, fontweight="bold", va="top")
    b = fig.add_axes([0.0, 0.0, 1.0, 0.44]); b.set_xlim(0, 100); b.set_ylim(0, 26); b.axis("off")
    b.text(0, 25.6, "b", fontsize=9.5, fontweight="bold", va="top")
    grey = dict(fc=C["light"], ec="#B8BEC4", tc=C["ink"], fs=6.1)
    box(b, 0.5, 7, 16, 13, "Candidate component\n\nscope fixed in advance:\nall rows, $d \\leq 1$ h,\nor uncovered rows", **grey)
    box(b, 19.5, 7, 16.5, 13, "Added to the complete\npipeline, on top of the\naccepted components,\nfor a grid of weights", **grey)
    box(b, 39, 7, 16.5, 13, "Weight $w^*$: pooled\n12-month optimum;\nre-chosen on 5 folds\nand scored on the 6th", **grey)
    b.text(70.5, 21.4, "all three must hold", ha="center", fontsize=6.1, color="#555555")
    checks = ["each of Jan, Feb, Jun, Jul better", "four-month RMSE better", "top-100 rows carry < 100 % of gain"]
    for i, t in enumerate(checks):
        box(b, 58.5, 15.2 - i * 5.0, 24, 4.0, t, "white", ec=C["ink"], tc=C["ink"], fs=5.9)
    box(b, 87, 13.2, 12.5, 6.0, "keep\n(decision file)", C["adsb"], fs=6.2, bold=True)
    box(b, 87, 4.2, 12.5, 5.0, "reject", C["reject"], fs=6.2, bold=True)
    for x0, x1 in ((16.5, 19.5), (36, 39), (55.5, 58.5)): arrow(b, x0, 13.5, x1, 13.5)
    arrow(b, 82.5, 13.5, 87, 16.2, color=C["adsb"]); arrow(b, 82.5, 8.5, 87, 6.7, color=C["reject"])
    b.text(93.25, 21.6, "then: size of the change\n2026 / 2025 at most 1.5", ha="center", fontsize=5.8, color="#555555")
    save(fig, "fig6_gate")


# ------------------------------------------------------------------ Figure 7: concentration and component gains
def fig7():
    K = D["concentration"]
    fig = plt.figure(figsize=(W, 2.75)); a = fig.add_axes([0.075, 0.2, 0.33, 0.72]); b = fig.add_axes([0.71, 0.2, 0.28, 0.72])
    k = np.array(K["k"]); s = np.array(K["share"]) * 100
    a.plot(k, s, color=C["base"], lw=1.3); a.set_xscale("log"); a.set_xlim(1, K["n_rows"] * 1.2); a.set_ylim(0, 102)
    a.set_xlabel(f"worst rows (of {K['n_rows']:,})"); a.set_ylabel("share of squared error (%)")
    for kk, dx in (("100", 2.2), ("1000", 2.2), ("10000", 2.2)):
        v = K["top"][kk] * 100; a.plot(int(kk), v, "o", ms=3.5, color=C["base"])
        a.text(int(kk) * dx, v - 1.5, f"top {int(kk):,}: {v:.0f} %", fontsize=6.6, va="top")
    a.grid(axis="y", color=C["grid"], lw=0.5); a.set_axisbelow(True); panel(a, "a")
    comps = [("ADS-B stacker (stage 1)", -13.74, C["adsb"]), ("daily reconciliation (stage 3)", -8.72, C["daily"]),
             ("assigned-stand timing", -0.97, C["adsb"]), ("runway neighbours (stage 2)", -0.69, C["nb"]), ("mixed-target CatBoost", -0.69, C["base"]),
             ("ADS-B corrector (stage 1b)", -0.67, C["adsb"]), ("Istanbul queue (stage 2b)", -0.42, C["nb"]),
             ("decorrelators", -0.32, C["base"]), ("bounds", -0.19, C["bounds"]), ("mixed-target LightGBM", -0.16, C["base"]),
             ("arrival-located stands", -0.12, C["adsb"]), ("stand readout (stage 4)", -0.08, C["adsb"])]
    y = np.arange(len(comps))[::-1]
    b.barh(y, [c[1] for c in comps], color=[c[2] for c in comps], height=0.66)
    b.set_yticks(y); b.set_yticklabels([c[0] for c in comps]); b.axvline(0, color=C["ink"], lw=0.6)
    b.set_xlabel("four-month change of 2025 RMSE (s)"); b.set_xlim(-17.5, 0.6)
    for yy, c in zip(y, comps): b.text(c[1] - 0.25, yy, f"{c[1]:.2f}".replace("-", "\u2212"), ha="right", va="center", fontsize=6.3)
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.text(1.0, -0.27, "each on top of the components accepted before it; not additive", transform=b.transAxes, fontsize=5.8,
           color="#555555", va="top", ha="right")
    panel(b, "b", x=-0.95)
    save(fig, "fig7_error_concentration")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7): f()
