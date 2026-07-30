#!/usr/bin/env python3
"""
plot_resonance.py  --  FEF-trend plots for hybridbar & bowtie resonance scans
=============================================================================


Pure matplotlib + stdlib -- imports NO meep, so it runs on a laptop.

Two ways to feed data (you never have to split the summary into several files):

  1) PASTE the rows you want straight into the call, as a triple-quoted string.
     Great for "FEF vs W only for the L_bar=1600 runs" -- paste just those rows:

         plot_fef_vs('''
         HybridBar_3  30  1600  150  160  6075.26  32172.28  32728.94  44491.69
         HybridBar_3  30  1600  150  200  6076.71  29166.10  29675.69  40199.56
         HybridBar_3  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
         HybridBar_3  30  1600  150  320  6172.06  15560.93  16591.80  23517.27
         ''', xvar="W", kind="hybrid")

  2) Pass a FILE PATH and (optionally) filter inline with `fixed=`:

         plot_fef_vs("results/Hybrid/summary1.txt", xvar="W", kind="hybrid",
                     fixed={"L_bar": 1600, "L_tip": 150, "gap": 30})

Column layout is known from `kind`, so pasted rows may include or omit the
header line; comments after '#' and rows whose numbers are corrupted are skipped.

  kind="hybrid":  Geometria  gap  L_bar  L_tip  W  lambda  FEF_mean  FEF_center  FEF_max
  kind="bowtie":  Geometria  gap  L      W      lambda  FEF_mean  FEF_center  FEF_max
"""

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

XINFO = {
    "l_bar":  ("Bar length [nm]",          "#8B0000", 50),    # dark red
    "l_tip":  ("Tip length [nm]",          "#B8860B", 50),    # dark goldenrod
    "w":      ("Width [nm]",               "#00008B", 50),    # dark blue
    "gap":    ("Gap size [nm]",            "#006400", 10),    # dark green
    "l":      ("Length [nm]",              "#8B0000", 50),    # dark red
    "lambda": ("Resonant wavelength [nm]", "#4B0082", None),  # indigo (auto minor)
}
FEF_YLABEL = {"mean": "Mean FEF in the gap region",
              "center": "FEF at the gap centre",
              "max": "Max-pixel FEF in the gap"}
KIND_LABEL = {"hybrid": "HybridBar", "bowtie": "BowTie"}


# ======================================================================
# parsing
# ======================================================================
def _load_text(data):
    """`data` is either a path to a .txt or the pasted text itself."""
    if isinstance(data, str) and os.path.exists(data):
        with open(data, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return str(data)


def parse_rows(data, kind):
    """
    Parse pasted text or a file into a list of dict rows keyed by canonical
    names (gap, l_bar/l/l_tip, w, lambda, fef_mean, fef_center, fef_max).

    Positional by `kind`; a header line, comments and rows with
    non-numeric cells are skipped automatically.
    """
    if kind not in ("hybrid", "bowtie"):
        raise ValueError("kind must be 'hybrid' or 'bowtie'")
    n_num = 8 if kind == "hybrid" else 7

    rows = []
    for raw in _load_text(data).splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5 or toks[0].lower() == "geometria":
            continue
        nums = toks[1:]
        try:
            vals = [float(t) for t in nums[:max(n_num, len(nums))]]
        except ValueError:
            vals = []
            for t in nums:
                try:
                    vals.append(float(t))
                except ValueError:
                    break
        if kind == "hybrid":
            if len(vals) < 6:              # need gap,L_bar,L_tip,W,lambda,+1 FEF
                continue
            rec = dict(gap=vals[0], l_bar=vals[1], l_tip=vals[2], w=vals[3],
                       lam=vals[4], fef_mean=vals[5],
                       fef_center=vals[6] if len(vals) > 6 else vals[5],
                       fef_max=vals[7] if len(vals) > 7 else vals[5])
        else:  # bowtie
            if len(vals) < 5:              # need gap,L,W,lambda,+1 FEF
                continue
            rec = dict(gap=vals[0], l=vals[1], w=vals[2], lam=vals[3],
                       fef_mean=vals[4],
                       fef_center=vals[5] if len(vals) > 5 else vals[4],
                       fef_max=vals[6] if len(vals) > 6 else vals[4])
        rows.append(rec)
    return rows


_XKEY = {"l_bar": "l_bar", "l_tip": "l_tip", "w": "w", "gap": "gap",
         "l": "l", "lambda": "lam"}
_FEFKEY = {"mean": "fef_mean", "center": "fef_center", "max": "fef_max"}


# ======================================================================
# plotting
# ======================================================================
def plot_fef_vs(data, xvar, kind="hybrid", fef="mean", fixed=None,
                color=None, connect=None, title=None, note=None,
                save=None, outdir="plots", show_peak=False):
    """
    Make figure of FEF versus `xvar`.

    data      : pasted rows (string) OR path to a summary .txt
    xvar      : "L_bar" | "L_tip" | "W" | "gap" | "lambda"   (bowtie: "L"|"W"|"gap"|"lambda")
    kind      : "hybrid" | "bowtie"
    fef       : "mean" (default) | "center" | "max"
    fixed     : optional {"L_bar":1600, ...} to keep params fixed when you pass
                a whole file instead of pasting a subset
    connect   : line+markers if True, scatter if False (default: scatter only for lambda)
    save      : output PNG path (default: <outdir>/<kind>_FEF_vs_<xvar>.png)

    Returns PNG.
    """
    xvar_l = xvar.lower()
    if xvar_l not in _XKEY:
        raise ValueError(f"xvar must be one of {list(_XKEY)} (got {xvar!r})")
    xkey = _XKEY[xvar_l]
    fkey = _FEFKEY[fef]
    rows = parse_rows(data, kind)

    if fixed:
        for k, v in fixed.items():
            fk = _XKEY[k.lower()]
            rows = [r for r in rows if abs(r[fk] - float(v)) < 1e-6]

    pts = [(r[xkey], r[fkey]) for r in rows]
    if not pts:
        print(f"[skip] FEF vs {xvar}: no matching rows")
        return None

    acc = {}
    for x, y in pts:
        acc.setdefault(x, []).append(y)
    xs = sorted(acc)
    ys = [sum(acc[x]) / len(acc[x]) for x in xs]

    if connect is None:
        connect = True   # dashed line by default; pass connect=False for a pure cloud
    if color is None:
        color = XINFO[xvar_l][1]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    ax.plot(xs, ys, marker="o", markersize=9, color=color, linewidth=1.5,
            linestyle=("--" if connect else "none"))

    if show_peak:
        i = max(range(len(ys)), key=lambda k: ys[k])
        ax.annotate(f"{ys[i]:,.0f}", (xs[i], ys[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=12, color="black")

    ax.set_xlabel(XINFO[xvar_l][0], fontsize=16)
    ax.set_ylabel(FEF_YLABEL[fef], fontsize=16)

    # title "FEF vs Bar length (gap=30, L_tip=150, W=240 nm)"
    if title is None:
        paren = note if note is not None else (
            ", ".join(f"{k}={float(v):g}" for k, v in fixed.items()) + " nm"
            if fixed else "")
        xname = XINFO[xvar_l][0].replace(" [nm]", "")
        title = f"FEF vs {xname}" + (f" ({paren})" if paren else "")
    ax.set_title(title, fontsize=16, pad=15)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if xvar_l != "lambda":
        ax.set_xticks(xs)          # majors on data points (lambda -> keep auto ticks)
    xstep = XINFO[xvar_l][2]
    ax.xaxis.set_minor_locator(
        MultipleLocator(xstep) if xstep is not None else AutoMinorLocator())

    ax.tick_params(which="major", length=6, labelsize=14, width=1.2)
    ax.tick_params(which="minor", length=3, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if save is None:
        os.makedirs(outdir, exist_ok=True)
        save = os.path.join(outdir, f"{kind}_FEF_vs_{xvar_l}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    print(f"[ok] FEF vs {xvar:<8} ({len(xs)} pts) -> {save}")
    return fig


def summarize(data, kind="hybrid"):
    rows = parse_rows(data, kind)
    keys = (["gap", "l_bar", "l_tip", "w", "lam"] if kind == "hybrid"
            else ["gap", "l", "w", "lam"])
    print(f"{KIND_LABEL[kind]}: {len(rows)} rows")
    for k in keys:
        vals = sorted({r[k] for r in rows})
        show = ", ".join(f"{v:g}" for v in vals)
        print(f"  {k:8s}: {show}")


if __name__ == "__main__":

    Hybrid_FEFvsW = """
    HybridBar_3  30  1600  150  160  6075.26  32172.28  32728.94  44491.69
    HybridBar_3  30  1600  150  200  6076.71  29166.10  29675.69  40199.56
    HybridBar_4  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
    HybridBar_4  30	 1600  150	280	 6126.21  16434.88	17574.56  25465.27
    HybridBar_3  30  1600  150  320  6172.06  15560.93  16591.80  23517.27
    """
    Hybrid_FEFvsL_bar = """
    HybridBar_4  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
    HybridBar_3  30  1400  150  240  5712.79  13751.78  14704.41  21319.87
    HybridBar_4  30  1800  150  240  6673.50  21653.03  23152.43  33596.56
    HybridBar_3	 30	 2000  150	240	 7265.28  25737.27	27519.15  39941.45
    HybridBar_3  30  2200  150  240  7849.04  30147.20  32234.07  46796.24
    HybridBar_3  30  2400  150  280  8462.19  32440.33  34688.21  50305.77
    """
    Hybrid_FEFvsL_tip = """
    HybridBar_4  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
    HybridBar_3	 30	 1600  50  240  6089.70  9805.60  9950.34  15770.22
    HybridBar_3  30  1600  100  240  6063.44  17332.22  18481.73  26139.68
    HybridBar_3	 30	 1600  200	240	 6141.02  18646.39	19936.33  29141.29
    HybridBar_3	 30	 1600  250	240	 6253.80  35667.73	88897.58  86427.98
    HybridBar_3	 30	 1600  300	240	 6327.40  37014.86	92472.25  87130.65
    """
    Hybrid_FEFvsgap = """
    HybridBar_4  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
    HybridBar_3	 50	 1600  150	240	 6006.25  5596.38  3910.63  12596.19
    HybridBar_3	 70	 1600  150	240	 5956.49  2834.02  1440.63  8796.85
    HybridBar_3	 90	 1600  150	240	 5917.82  1766.93  703.59  7065.64
    """

    Hybrid_FEFvslambda = """
    HybridBar  30  1400  150  240  5712.79  13751.78  14704.41  21319.87
    HybridBar  30  1600  150  240  6082.20  17896.85  19136.39  27764.71
    HybridBar  30  1800  150  240  6673.50  21653.03  23152.43  33596.56
    HybridBar  30  2000  150  240  7265.28  25737.27  27519.15  39941.45
    HybridBar  30  2200  150  240  7849.04  30147.20  32234.07  46796.24
    HybridBar  30  2400  150  240  8421.60  34876.60  37290.54  54151.44
    """
    
    # plot_fef_vs(Hybrid_FEFvsW, xvar="W", kind="hybrid", fef="mean",
    #             note="fixed: L_bar=1600, L_tip=150, gap=30 nm")

    # plot_fef_vs(Hybrid_FEFvsL_bar, xvar="L_bar", kind="hybrid", fef="mean",
    #             note="fixed: L_tip=150, W = 240, gap=30 nm")

    # plot_fef_vs(Hybrid_FEFvsL_tip, xvar="L_tip", kind="hybrid", fef="mean",
    #             note="fixed: L_bar=1600, W = 240, gap=30 nm")

    # plot_fef_vs(Hybrid_FEFvsgap, xvar="gap", kind="hybrid", fef="mean",
    #             note="fixed: L_bar=1600, L_tip=150, W = 240")

    plot_fef_vs(Hybrid_FEFvslambda, xvar="lambda", kind="hybrid", fef="mean",
                note="L_tip=150, W=240, gap=30 nm")

    # ----------------whole file + inline filter -------------------------
    # plot_fef_vs("results/Hybrid/resonant_peaks_hybrid_summary1.txt",
    #             xvar="L_tip", kind="hybrid", fef="mean",
    #             fixed={"L_bar": 1600, "W": 240, "gap": 30})

    # -------peek at what's in a file before plotting------------------------
    # summarize("results/Hybrid/resonant_peaks_hybrid_summary1.txt", "hybrid")

    plt.show()
