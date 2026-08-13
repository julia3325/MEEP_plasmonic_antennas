"""
LOKALNY postprocessing map wzmocnienia - BEZ importu meep.

Na Aresie compute_dft_enhancement_maps zapisuje tylko male pliki
dft_enhancement_<plane>_e2.h5 + field_vs_time_gap.dat (rendering PNG jest tam
wylaczony, bo sie wywala). Ten modul renderuje z nich PNG lokalnie.

Odpowiednik postprocess_dft_efe / postprocess_dft_enhancement_maps z
experiments.py / meep_utils.py, ale bez `import meep` - tamte moduly ciagna
meep w pierwszej linii, mimo ze sam postprocessing nie uzywa z niego NICZEGO.
Dziala w gołym srodowisku z h5py + numpy + matplotlib (np. conda `base`).

Uzycie:

    python postprocess_maps.py                 # foldery z SIMULATIONS ponizej
    python postprocess_maps.py <folder> [...]  # albo wprost z linii polecen

Wzorowane na plot_resonance.py: czysty stdlib + h5py/numpy/matplotlib.
"""

import os
import sys

import h5py
import numpy as np
import matplotlib
# Ten modul wylacznie zapisuje PNG (nigdy plt.show()), wiec Agg jest zawsze
# poprawny. Warunek na DISPLAY nie wystarcza: w WSL zmienna bywa ustawiona,
# mimo ze serwera X nie ma, i Qt wywala sie na "could not load plugin xcb".
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Grubosci warstw [nm] - do wyznaczenia ROI w przekrojach pionowych
TH_AU = 30.0
TH_TI = 5.0

# Foldery do przetworzenia: {"folder": <nazwa w results/>, "gap": <nm>}
SIMULATIONS = [
    {"folder": "Hybrid/HybridBar_gap_30nm_Lbar_2400nm_Ltip_200_W_240nm_AuTiSiO2_res300_lambda_8.47611",
    "gap": 30},
]

RESULTS_TXT = "results/Hybrid/DFT_EFE_summary_maps.txt"

PLANE_LABELS = {
    "xyplanar":    ("XY", "X [nm]", "Y [nm]"),
    "xyplanarTOP": ("XY", "X [nm]", "Y [nm]"),
    "xzplanar":    ("XZ", "X [nm]", "Z [nm]"),
    "yzplanar":    ("YZ", "Y [nm]", "Z [nm]"),
}

# ====== TU ZMIENIASZ PRZYBLIZENIE MAPY ======
# Polowa szerokosci kadru [nm] per rodzina plaszczyzn: (os pozioma, os pionowa).
# None dla calej plaszczyzny albo dla pojedynczej osi.
#
# UWAGA: to NIE jest to samo co roi_for_gap() - ROI sluzy wylacznie do liczenia
# statystyk mean_roi/max_roi w pliku txt i nie wplywa na to, co widac.
#
# Zapisana plaszczyzna dla HybridBar 2400x200x240 siega +-2662 nm w X, +-172 nm
# w Y i +-38 nm w Z, wiec np. (250, 170) pokazuje sama szczeline z otoczeniem.
VIEW_NM = {
    "XY": (250.0, 170.0),
    "XZ": (250.0, None),
    "YZ": (170.0, None),
}

# Skala kolorow: True -> vmax z widocznego wycinka (szczelina dobrze naswietlona),
# False -> vmax z calej plaszczyzny (kadry porownywalne miedzy soba)
COLOR_FROM_VIEW = True

# ====== STYL RYSUNKU ======
# Ten sam co w plot_resonance.py / plot_efe.py i co na starych mapach bowtie:
# czcionka szeryfowa (Times New Roman, z DejaVu Serif jako zapasem, bo Times
# nie zawsze jest zainstalowany w srodowisku conda), matematyka w stix.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
FS_TITLE = 19      # tytul
FS_LABEL = 19      # opisy osi
FS_TICKS = 16      # liczby na osiach i na colorbarze
FS_CBAR = 17       # opis colorbara
FIGSIZE = (8.5, 7.0)


def roi_for_gap(gap_nm):
    """ROI w nm: przerwa miedzy ramionami (XY) i przekroje przez Au+Ti."""
    return {
        "XY": {"center": (0.0, 0.0),       "width": gap_nm, "height": 10.0},
        "XZ": {"center": (0.0, -TH_TI / 2.0), "width": gap_nm, "height": TH_AU + TH_TI},
        "YZ": {"center": (0.0, -TH_TI / 2.0), "width": 10.0,  "height": TH_AU + TH_TI},
    }


def _plot_dft_map(enh, h_nm, v_nm, xlabel, ylabel, title, save_path, view=None):
    """
    Pojedyncza mapa wzmocnienia w skali log -> PNG.

    view : (polowa_h, polowa_v) w nm albo None. Kadr jest przycinany przez
    set_xlim/set_ylim, wiec dane pozostaja pelne - zmienia sie tylko to, co widac.
    """
    max_enh = float(np.max(enh))

    # zakres kolorow liczony z tego, co faktycznie widac - inaczej przy mocnym
    # przyblizeniu na szczeline caly kadr potrafi wyjsc jednolicie jasny albo czarny
    vmax = max_enh
    if view is not None and COLOR_FROM_VIEW:
        hh, vv = view
        hm = np.ones_like(h_nm, dtype=bool) if hh is None else (np.abs(h_nm) <= hh)
        vm = np.ones_like(v_nm, dtype=bool) if vv is None else (np.abs(v_nm) <= vv)
        sub = enh[np.ix_(hm, vm)]
        if sub.size:
            vmax = float(np.max(sub))

    extent = [h_nm[0], h_nm[-1], v_nm[0], v_nm[-1]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(
        enh.T,
        origin="lower",
        extent=extent,
        cmap="inferno",
        # skala log obejmujaca 4 dekady ponizej maksimum: slabe pole wzdluz
        # ramion (mediana ~48 przy szczelinie ~47000) jest wtedy widoczne,
        # zamiast wpadac pod prog czerni
        norm=LogNorm(vmin=max(vmax * 1e-4, 1e-3), vmax=vmax),
        aspect="auto",
    )
    if view is not None:
        hh, vv = view
        if hh is not None:
            ax.set_xlim(-hh, hh)
        if vv is not None:
            ax.set_ylim(-vv, vv)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=12)
    ax.tick_params(which="both", direction="in", color="white",
                   top=True, right=True, labelsize=FS_TICKS)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"|E|$^2$ / |E$_0$|$^2$", fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_TICKS)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def postprocess_maps(load_path, roi_nm=None):
    """
    Renderuje mapy PNG z dft_enhancement_*.h5 w `load_path` i liczy statystyki
    w ROI. Zwraca {plane: {"max":..., "mean_roi":..., "max_roi":...}}.
    """
    stats = {}
    for name, (roi_key, xlabel, ylabel) in PLANE_LABELS.items():
        h5_path = os.path.join(load_path, f"dft_enhancement_{name}_e2.h5")
        if not os.path.exists(h5_path):
            print(f"  [pomijam] brak {os.path.basename(h5_path)}")
            continue

        with h5py.File(h5_path, "r") as f:
            enh = f["enhancement"][...]
            h_nm = f["axis_h_um"][...] * 1e3
            v_nm = f["axis_v_um"][...] * 1e3
            wavelength_nm = float(f.attrs.get("wavelength_nm", 0.0))

        entry = {"max": float(np.max(enh))}

        if roi_nm is not None and roi_key in roi_nm:
            r = roi_nm[roi_key]
            cx, cy = r["center"]
            hmask = (h_nm >= cx - r["width"] / 2.0) & (h_nm <= cx + r["width"] / 2.0)
            vmask = (v_nm >= cy - r["height"] / 2.0) & (v_nm <= cy + r["height"] / 2.0)
            roi_data = enh[np.ix_(hmask, vmask)]
            if roi_data.size > 0:
                entry["mean_roi"] = float(np.mean(roi_data))
                entry["max_roi"] = float(np.max(roi_data))

        png = os.path.join(load_path, f"dft_enhancement_{name}_e2.png")
        _plot_dft_map(
            enh, h_nm, v_nm, xlabel, ylabel,
            title=(f"|E|$^2$ enhancement @ {wavelength_nm:.0f} nm "
                   f"(max = {entry['max']:.0f})"),
            save_path=png,
            view=VIEW_NM.get(roi_key),
        )
        print(f"  [ok] {name:12s} max={entry['max']:10.1f}"
              + (f"  meanROI={entry['mean_roi']:9.1f}" if "mean_roi" in entry else "")
              + f"  -> {os.path.basename(png)}")
        stats[name] = entry

    # diagnostyka Ex(t) z monitora punktowego
    dat_path = os.path.join(load_path, "field_vs_time_gap.dat")
    if os.path.exists(dat_path):
        t_e, ex_e, t_a, ex_a = [], [], [], []
        with open(dat_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    t_e.append(float(parts[0])); ex_e.append(float(parts[1]))
                if len(parts) >= 4 and parts[2] and parts[3]:
                    t_a.append(float(parts[2])); ex_a.append(float(parts[3]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_e, ex_e, color="gray", linewidth=1, label="empty")
        ax.plot(t_a, ex_a, color="darkred", linewidth=1, label="antenna")
        ax.set_xlabel("t [Meep units]", fontsize=13)
        ax.set_ylabel("Ex at gap centre", fontsize=13)
        ax.set_title("Ex(t) at monitor point", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(load_path, "field_vs_time_gap.png"), dpi=300)
        plt.close(fig)
        print("  [ok] field_vs_time_gap.png")

    return stats


def run(simulations=None, results_filename=RESULTS_TXT):
    """Przetwarza liste folderow i dopisuje statystyki do pliku txt."""
    simulations = simulations if simulations is not None else SIMULATIONS
    if not simulations:
        print("Nic do zrobienia: uzupelnij SIMULATIONS albo podaj foldery "
              "w linii polecen.")
        return 0

    os.makedirs(os.path.dirname(results_filename) or ".", exist_ok=True)
    new_file = not os.path.exists(results_filename)
    with open(results_filename, "a") as f:
        if new_file:
            f.write("Folder\tGap[nm]\t"
                    "meanROI_XY\tmaxROI_XY\tmax_XY\t"
                    "meanROI_XYTOP\tmaxROI_XYTOP\tmax_XYTOP\t"
                    "meanROI_XZ\tmaxROI_XZ\tmax_XZ\t"
                    "meanROI_YZ\tmaxROI_YZ\tmax_YZ\n")

    for sim in simulations:
        folder, gap_nm = sim["folder"], sim["gap"]
        load_path = folder if os.path.isdir(folder) else os.path.join("results", folder)
        if not os.path.isdir(load_path):
            print(f"Brak folderu {load_path}, pomijam...")
            continue

        print(f"\nPostprocessing map DFT: {os.path.basename(load_path)}")
        stats = postprocess_maps(load_path, roi_nm=roi_for_gap(gap_nm))
        if not stats:
            continue

        row = [os.path.basename(load_path), str(gap_nm)]
        for plane in ["xyplanar", "xyplanarTOP", "xzplanar", "yzplanar"]:
            s = stats.get(plane, {})
            row += [f"{s.get('mean_roi', 0.0):.3f}",
                    f"{s.get('max_roi', 0.0):.3f}",
                    f"{s.get('max', 0.0):.3f}"]
        with open(results_filename, "a") as f:
            f.write("\t".join(row) + "\n")

    print(f"\nWyniki dopisano do {results_filename}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # gap 30 nm
        run([{"folder": a, "gap": 30} for a in sys.argv[1:]])
    else:
        run()
