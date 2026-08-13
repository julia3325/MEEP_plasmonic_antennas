import sys, os, time
import meep as mp
from meep.materials import *
from utils.sys_utils import *
from utils.meep_utils import *
from utils.geometry_utils import make_cell
from utils.logger import save_and_show_config, append_time_to_file
from src.antenna_geometries import *
from src.config import SimulationConfig
from src.sources import *
from src.volumes import *
from visualization.plotter import *

xm = 1000
mp.Simulation.eps_averaging = False

# DFT convergence settings: run until DFT fields stop changing (source
# turn-off + resonance ring-down), with a hard cap on total run time
DFT_DECAY_TOL = 1e-6
DFT_MAX_RUN_TIME = 800.0  # [Meep time units]

# --- tip sharpness / gap validity ------------------------------------------
# ROOT CAUSE of the inconsistent rows in resonant_peaks_hybrid_summary1.txt
# (gap=30 with L_tip=250/300, W=140, and the L_tip=300/W=160 row):
#
#   corrected_gap(g, R, theta) = g - 2*(R/sin(theta/2) - R),  theta = tip angle
#
# is the gap to use in the SHARP geometry so that, after filleting with radius
# R, the effective gap comes out as g. For a sharp tip (narrow W, long L_tip)
# theta gets small, 1/sin(theta/2) blows up, and the correction eats the whole
# gap. Measured on the summary1 geometries (gap=30 nm, R=5 nm):
#
#   W=240 L_tip=150 -> 23.99 nm   (the design point, clean)
#   W=140 L_tip=150 -> 16.35 nm   \
#   W=240 L_tip=250 -> 16.89 nm    |  every anomalous row in the file
#   W=240 L_tip=300 -> 13.07 nm    |
#   W=160 L_tip=300 ->  1.19 nm   /   (a QUARTER of a pixel at resolution 200)
#
# The four broken rows are exactly the four smallest corrected gaps, and the
# cut is sharp: everything at 18.75 nm and above is self-consistent. At 1.19 nm
# the tips are effectively shorted on the grid, which is why that row reports
# FEF_center 10x BELOW FEF_mean.
#
# Every row in that file was produced AFTER 46ae533, i.e. with the auto-scaled
# tip_apex_patch, so the apex patch was placed correctly and is NOT the cause.
# What remains is the correction formula itself, and one thing it cannot know:
# R = 5 nm is a SINGLE pixel at resolution 200. A 5 nm fillet is simply not
# representable on a 5 nm grid, so the "effective gap becomes g after
# filleting" premise fails - and since delta ~ 1/sin(theta/2), the error is
# amplified exactly as the tip gets sharper. Hence a guard on both.
MIN_CORRECTED_GAP_FRAC = 0.6   # corrected gap must keep >=60% of the target gap
MIN_CORRECTED_GAP_PIXELS = 4   # ...and still span >=4 pixels
MIN_RADIUS_PIXELS = 2          # fillet radius must be resolved by the grid

# Tip fillet radius [nm] for the resonance SCANS. 0 = sharp tip: corected_gap
# collapses to gap, no apex patch, geometry is exactly what was requested.
# Set back to 5 only together with a resolution that resolves it (>=800, where
# R = 4 px) - at 200 a 5 nm fillet is a single pixel and is pure cost.
TIP_RADIUS_NM = 0.0

# --- gap probe geometry ----------------------------------------------------
# Separate latent issue found while chasing the above, not the cause of it:
# inner_gap used to be   max(gap - 4.0/config.resolution, 0.4*gap),
# so the volume FEF_mean averages over was a function of the RESOLUTION - at
# gap=30 nm it ran from 12 nm (res 200) to 25 nm (res 800). |E|^2 climbs
# steeply towards the metal walls, so the same antenna measured at two
# resolutions returns two different FEF_mean values. Every row in summary1 was
# run at resolution 200, so this did not affect that file - but it silently
# breaks any screening(200) vs showcase(400) comparison, and it means the mean
# does not actually converge. The volume is now a fixed fraction of the gap;
# resolution controls only how well that fixed volume is sampled.
INNER_GAP_FRACTION = 0.4     # inner box width = 0.4 * gap, resolution-free
MIN_INNER_PIXELS = 3         # warn below this many pixels across the inner box
# The gap-centre probe was a ZERO-SIZE dft_fields region: one raw Yee sample,
# sitting on BOTH mirror-symmetry planes, with eps_averaging off. It is the
# probe that reported FEF_max_px < FEF_center - structurally impossible, since
# the max is taken over a box that CONTAINS the centre point. Now a small box,
# reduced as mean |Ex|^2 like the inner-gap mean, so no single pixel dominates.
CENTER_PROBE_PIXELS = 2      # half-width of the centre box, in pixels

# --- peak picking ----------------------------------------------------------
# Median filter width [bins], applied ONLY to the spectrum used for peak
# PICKING - never to the reported values. At nfreq=400 over 5700-10300 nm the
# bins sit ~7 nm apart near 6 um, while a Q~20 resonance there is ~300 nm wide
# (~40 bins), so a 5-bin median cannot move a genuine peak. This is a cheap
# guard against single-bin artefacts, NOT the fix for the rows above - their
# spectra are smooth and single-peaked; the volume was the problem.
PEAK_PICK_MEDIAN_BINS = 5
# Flag a picked bin whose raw value stands this far above the median-filtered
# spectrum: that is a spike, not a resonance.
SPIKE_FLAG_RATIO = 1.5

def _get_mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD, MPI
    except ImportError:
        return None, None

def _init_results_file(path, header):
    """
    Prepare a summary file WITHOUT ever truncating it. Call on the master rank.
    Returns the path actually used - assign it back to `results_filename`.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            f.write(header)
        return path

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        existing = f.readline()

    if existing.strip() == header.strip():
        print(f"[i] dopisuje do istniejacego {path}")
        return path

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(path)
    new_path = f"{root}_{stamp}{ext or '.txt'}"
    print(f"[!] {path} ma inny naglowek (zmienil sie format kolumn) - "
          f"nie ruszam go, pisze do {new_path}")
    with open(new_path, "w") as f:
        f.write(header)
    return new_path


def _median_filter(y, width):
    """
    Odd-width running median with edge padding. numpy only - scipy is not
    guaranteed to be in the cluster environment.
    """
    y = np.asarray(y, dtype=float)
    w = int(width)
    if w < 3 or y.size < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, y.size if y.size % 2 else y.size - 1)
    h = w // 2
    pad = np.pad(y, h, mode="edge")
    try:
        win = np.lib.stride_tricks.sliding_window_view(pad, w)
        return np.median(win, axis=-1)
    except AttributeError:          # numpy < 1.20
        return np.array([np.median(pad[i:i + w]) for i in range(y.size)])



def _pick_resonance_idx(spectrum, ref=None, edge_guard=2, snr_frac=0.1,
                        smooth=PEAK_PICK_MEDIAN_BINS):
    """
    Pick the wavelength of MAXIMUM in-band enhancement.

    The scan window (5700-10300 nm) is the tuning range of the laser used
    in the lab, so the quantity of interest is the best enhancement
    ACHIEVABLE within that band - not the vacuum resonance position. Hence
    argmax over the window: if the spectrum peaks inside, that is the
    resonance; if it rises to an edge, the edge value is the best the laser
    can reach and is a valid (if sub-optimal) result.

    SNR mask (`ref`): the ant/empty ratio is meaningless where the empty
    reference field is near zero (division by noise). Passing the empty
    spectrum restricts the search to frequencies where the reference is at
    least `snr_frac` of its own maximum, killing the spurious band-edge
    spike that otherwise pins every geometry to the same bluest point. With
    a properly broad source this mask excludes nothing legitimate.

    `edge_guard` additionally trims a couple of outermost points. The
    returned `at_boundary` flag marks geometries whose best point sits at
    the window edge (true resonance likely outside the laser band).

    Spike rejection (`smooth`): argmax over the RAW spectrum lets a single bad
    bin win. That is what corrupted the (1600,150,140), (1600,250,240) and
    (1600,300,240) rows of summary1.txt - and because every FEF is then read at
    that bin, lambda_res and FEF_mean were wrong too, not only FEF_center. The
    peak is therefore picked on a median-filtered copy, then snapped back to the
    local RAW maximum so the filter itself cannot shift the answer. See
    PEAK_PICK_MEDIAN_BINS for why the width is safe.

    NOTE: the parabolic sub-bin refinement was removed on purpose - resolution
    in lambda now comes from nfreq alone (500 bins over 5700-10300 nm, i.e.
    ~5.1 nm near 5.7 um and ~16.5 nm near 10.3 um). Geometries whose resonances
    differ by less than one bin therefore report the SAME wavelength again.

    Returns
    -------
    (idx, at_boundary) : (int, bool)
    """
    n = len(spectrum)
    g = min(edge_guard, n // 2)
    lo, hi = g, n - g  # search range [lo, hi)

    spec = np.asarray(spectrum, dtype=float)
    picked_on = _median_filter(spec, smooth) if smooth else spec
    search = np.full(n, -np.inf)
    search[lo:hi] = picked_on[lo:hi]

    if ref is not None:
        ref = np.asarray(ref, dtype=float)
        thr = snr_frac * np.max(ref[lo:hi])
        low_snr = ref < thr
        # only apply the mask if it leaves something to choose from
        if np.any(np.isfinite(search) & ~low_snr):
            search[low_snr] = -np.inf

    idx = int(np.argmax(search))
    # The median filter can bias the argmax by ~1 bin even on a clean peak, so
    # snap back to the local RAW maximum within the filter half-width. Spike
    # rejection is kept (the spike is outside this neighbourhood), the bias is
    # not.
    if smooth and smooth >= 3:
        h = int(smooth) // 2
        a, b = max(lo, idx - h), min(hi, idx + h + 1)
        if b > a:
            idx = a + int(np.argmax(spec[a:b]))

    # flag if the chosen point is at (or right next to) either window edge
    at_boundary = (idx <= lo + 1) or (idx >= hi - 2)

    return idx, at_boundary

def check_gap_geometry(gap, width=None, tip_length=None, radius=0.0,
                       resolution=200):
    """
    Validate the gap geometry BEFORE spending cluster hours on it.

    Recomputes what build_geometry() will do and reports whether the result is
    still a physically meaningful, resolvable gap. See the note at the top of
    this module for why sharp tips degenerate.

    All lengths in nm.

    width / tip_length : the tapered tip, i.e. HybridBar(W, L_tip) or
        BowTie(W, L). Omit them for SplitBar, which is two plain bars with no
        tip and therefore no fillet correction - its gap is used verbatim and
        only the resolution check applies.

    Returns (effective_gap_nm, flags): `flags` is a list of short strings,
    empty when the geometry is sound.
    """
    px_nm = xm / float(resolution)

    # A tapered tip (HybridBar, BowTie) shrinks the gap through the fillet
    # correction; SplitBar is two plain bars with no tip, so its gap is used
    # verbatim by build_geometry() and only needs the resolution check.
    if tip_length and width and radius > 0:
        theta = 2.0 * np.arctan(width / (2.0 * tip_length))
        cg = gap - 2.0 * (radius / np.sin(theta / 2.0) - radius)
        shape = (f"W={width:.0f} L_tip={tip_length:.0f}, "
                 f"theta={np.degrees(theta):.1f} deg, ")
    else:
        cg = gap
        shape = ""

    flags = []
    if cg <= 0:
        flags.append(f"GAP_CLOSED(corrected={cg:.2f}nm)")
    else:
        if cg < MIN_CORRECTED_GAP_FRAC * gap:
            flags.append(f"GAP_EATEN({cg:.2f}/{gap:.0f}nm="
                         f"{100*cg/gap:.0f}%)")
        if cg / px_nm < MIN_CORRECTED_GAP_PIXELS:
            flags.append(f"GAP_UNDERRESOLVED({cg/px_nm:.1f}px)")
    # radius == 0 is a deliberate sharp tip, not an unresolved fillet - only
    # complain when a fillet was actually asked for and the grid cannot show it
    if radius > 0 and radius / px_nm < MIN_RADIUS_PIXELS:
        flags.append(f"FILLET_UNRESOLVED(R={radius/px_nm:.1f}px)")

    if flags and mp.am_master():
        print(f"[!] SUSPECT GEOMETRY gap={gap:.0f}: {shape}"
              f"effective_gap={cg:.2f} nm ({cg/px_nm:.1f} px) -> {' '.join(flags)}")
    return cg, flags


def _result_flags(fef_mean, fef_center, fef_max, best_idx, at_boundary,
                  geom_flags=()):
    """
    Self-consistency flags for one reported row.

    WHY: summary1.txt had four rows that looked like ordinary numbers and were
    plotted as physics for weeks. Nothing in the output said they were broken -
    it took comparing ratios across the whole file to notice. These checks make
    a bad row announce itself at the moment it is produced:

      MAX<CENTER    FEF_max_px is a maximum over a box that CONTAINS the centre
                    box, and both are divided by an almost identical empty-run
                    value, so max >= centre cannot be violated by physics. When
                    it is, a probe is reading somewhere it should not - e.g.
                    inside the metal, which is what the degenerate geometries did.
      AT_WINDOW_EDGE  the peak sits at the edge of the search window, so the
                    true resonance is probably outside it and the reported FEF
                    is an off-resonance value (all three bow-tie rows at
                    5712.79 nm were this).
      SPIKE         the winning bin stands far above the median-filtered
                    spectrum, i.e. a single-bin artefact rather than a resonance.

    Plus whatever check_gap_geometry already found about the geometry itself.
    """
    flags = list(geom_flags)
    if at_boundary:
        flags.append("AT_WINDOW_EDGE")
    if fef_max[best_idx] < fef_center[best_idx]:
        flags.append(f"MAX<CENTER({fef_max[best_idx]/fef_center[best_idx]:.3f})")
    smoothed = _median_filter(fef_center, PEAK_PICK_MEDIAN_BINS)
    if smoothed[best_idx] > 0 and fef_center[best_idx] / smoothed[best_idx] > SPIKE_FLAG_RATIO:
        flags.append("SPIKE")
    return flags


def _provenance(resolution, radius_nm, cg_nm, probe, flags):
    """
    Trailing '#' comment written next to every result row.

    WHY: a row records geometry and FEF but says nothing about HOW it was
    measured. Rows produced at different radius / probe volume are then
    indistinguishable, which is exactly how the summary1.txt confusion started
    - the only clue was a hand-typed `HybridBar_1` vs `_3` label that turned out
    to mean nothing. plot_resonance.parse_rows() cuts everything after '#', so
    this is invisible to the plots.
    """
    txt = (f"\t# res={resolution} R={radius_nm:g}nm "
           f"eff_gap={cg_nm:.2f}nm inner={probe['inner_gap_nm']:.1f}nm")
    if flags:
        txt += " FLAGS:" + ",".join(flags)
    return txt


def _gap_probe_sizes(gap, resolution):
    """
    Build the three gap DFT region sizes, all in Meep units.

    Returns (full, inner, centre, info). `gap` is already in Meep units.

    The inner box is a FIXED fraction of the gap (INNER_GAP_FRACTION) so that
    FEF_mean measures the same physical volume at every resolution - see the
    root-cause note at the top of this module. `info` carries the numbers that
    must be written next to every result row, so a row can never again be
    ambiguous about how it was measured.
    """
    px = 1.0 / float(resolution)
    inner = INNER_GAP_FRACTION * gap
    centre = CENTER_PROBE_PIXELS * px

    full_size = mp.Vector3(gap, 10 / xm, 10 / xm)
    inner_size = mp.Vector3(inner, 10 / xm, 10 / xm)
    centre_size = mp.Vector3(centre, centre, centre)

    n_px = inner / px
    info = {"resolution": resolution, "inner_gap_nm": inner * xm,
            "inner_px": n_px, "centre_px": CENTER_PROBE_PIXELS}
    if n_px < MIN_INNER_PIXELS and mp.am_master():
        print(f"[!] inner gap box spans only {n_px:.1f} pixels "
              f"({inner*xm:.1f} nm at resolution {resolution}) - FEF_mean will "
              f"be poorly sampled; raise the resolution rather than the box.")
    return full_size, inner_size, centre_size, info


def _dft_gap_spectra(sim, dft_box, dft_inner, dft_center, nfreq, comm, MPI):
    """
    Extract three Ex spectra from the gap DFT regions:

        max_amp[i]    - max |Ex| over the full gap box (includes pixels at
                        the metal walls -> grid-singular, resolution-dependent)
        mean_int[i]   - mean |Ex|^2 over the inner box (metal-adjacent pixels
                        excluded; fixed physical volume -> converges with
                        resolution)
        center_int[i] - mean |Ex|^2 over the small centre box

    center is an INTENSITY now, not an amplitude: the probe went from one raw
    Yee sample to a small box, so callers divide it directly by the empty
    reference instead of squaring first.

    All reductions are MPI-safe whether get_dft_array returns full or
    chunk-local arrays: max uses a MAX reduction, the two means use SUM of
    (sum, count) so rank multiplicity cancels.
    """
    max_amp = np.zeros(nfreq)
    mean_int = np.zeros(nfreq)
    center_int = np.zeros(nfreq)

    for i in range(nfreq):
        arr = sim.get_dft_array(dft_box, mp.Ex, i)
        vmax = float(np.max(np.abs(arr))) if (arr is not None and arr.size > 0) else 0.0

        arr_in = sim.get_dft_array(dft_inner, mp.Ex, i)
        if arr_in is not None and arr_in.size > 0:
            s = float(np.sum(np.abs(arr_in) ** 2))
            n = float(arr_in.size)
        else:
            s, n = 0.0, 0.0

        arr_c = sim.get_dft_array(dft_center, mp.Ex, i)
        vc = float(np.max(np.abs(arr_c))) if (arr_c is not None and arr_c.size > 0) else 0.0

        if comm is not None:
            vmax = comm.allreduce(vmax, op=MPI.MAX)
            s = comm.allreduce(s, op=MPI.SUM)
            n = comm.allreduce(n, op=MPI.SUM)
            vc = comm.allreduce(vc, op=MPI.MAX)

        max_amp[i] = vmax
        mean_int[i] = s / n if n > 0 else 0.0
        center_amp[i] = vc

    return max_amp, mean_int, center_amp

def check_antenna_geometry(kind="hybridbar", gap=30, W=240, L_bar=1800, L_tip=150,
                           length=1100, radius=5, resolution=800,
                           with_substrate=True, view_nm=None,
                           out_dir="results/geom_check"):
    """
    Parameters:

    kind : "hybridbar" | "bowtie"
    gap, W, L_bar, L_tip, length, radius : geometry in nm (length = bow-tie
        triangle length; L_bar/L_tip = hybrid bar/tip). 
    with_substrate : include the SiO2 block
    view_nm : full width of the square XY view [nm]; None -> auto around gap.
    out_dir : where the PNGs are written.

    Writes (per call): geom_<kind>_gap<g>_XY.png, _XYzoom.png, _XZ.png
    """

    Th_Au = 30 / xm
    Th_Ti = 5 / xm
    Th_Sub = 100 / xm
    g = gap / xm
    r = radius / xm

    if kind == "hybridbar":
        tag = f"hybrid_gap{gap}_Lbar{L_bar}_Ltip{L_tip}_W{W}"
        AuTop = HybridBar(gap=g, bar_length=L_bar/xm, tip_length=L_tip/xm,
                          width=W/xm, thickness=Th_Au, radius=r, material=Au, z_offset=0.0)
        TiBetween = HybridBar(gap=g, bar_length=L_bar/xm, tip_length=L_tip/xm,
                              width=W/xm, thickness=Th_Ti, radius=r, material=Ti,
                              z_offset=-(Th_Au + Th_Ti)/2.0)
        tip_reach_nm = L_tip
    elif kind == "bowtie":
        tag = f"bowtie_gap{gap}_L{length}_W{W}"
        AuTop = BowTie(gap=g, length=length/xm, width=W/xm, thickness=Th_Au,
                       radius=r, material=Au, z_offset=0.0)
        TiBetween = BowTie(gap=g, length=length/xm, width=W/xm, thickness=Th_Ti,
                           radius=r, material=Ti, z_offset=-(Th_Au + Th_Ti)/2.0)
        tip_reach_nm = length
    else:
        raise ValueError("kind must be 'hybridbar' or 'bowtie'")

    geometry = AuTop.build_geometry() + TiBetween.build_geometry()

    # view: square window centred on the gap. Default shows the whole tip
    # (hybrid) or ~800 nm around the apex (bow-tie is huge, only apex matters).
    if view_nm is None:
        view_nm = 2 * (tip_reach_nm + gap) + 200 if kind == "hybridbar" else 800
    view = view_nm / xm
    z_ext = (Th_Au + Th_Ti + Th_Sub) + 160/xm

    if with_substrate:
        # substrate spans the view so it forms a clean background
        substrate = mp.Block(size=mp.Vector3(view + 400/xm, view + 400/xm, Th_Sub),
                             center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)),
                             material=SiO2)
        geometry = geometry + [substrate]

    cell = mp.Vector3(view, view, z_ext)
    sim = mp.Simulation(cell_size=cell, boundary_layers=[], geometry=geometry,
                        resolution=resolution, dimensions=3)

    if mp.am_master():
        os.makedirs(out_dir, exist_ok=True)
        print(f"[geom-check] {tag}")
        print(f"  gap = {gap} nm   corrected_gap (Au) = {AuTop.corected_gap*xm:.2f} nm"
              f"   radius = {radius} nm")
        if radius > 0:
            patch_xc = AuTop.corected_gap * xm
            patch_len = (AuTop.corected_gap + 2/1000) * xm
            patch_h = max(1.2 * r, 4/1000) * xm
            print(f"  tip_apex_patch: x-centre = +-{patch_xc:.2f} nm,"
                  f" size = {patch_len:.2f} x {patch_h:.2f} nm"
                  f"   (patch inner edge at x = {patch_xc - patch_len/2:.2f} nm,"
                  f" gap wall at x = {gap/2:.2f} nm)")

    save_2D_plot(sim, mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(view, view, 0)),
                 save_name=f"geom_{tag}_XY.png", IMG_CLOSE=True, path_to_save=out_dir)
    zoom = min(view, (2 * (L_tip if kind == "hybridbar" else 120) + 2 * gap) / xm)
    save_2D_plot(sim, mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(zoom, zoom, 0)),
                 save_name=f"geom_{tag}_XYzoom.png", IMG_CLOSE=True, path_to_save=out_dir)
    save_2D_plot(sim, mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(view, 0, z_ext)),
                 save_name=f"geom_{tag}_XZ.png", IMG_CLOSE=True, path_to_save=out_dir)

    sim.reset_meep()
    if mp.am_master():
        print(f"  -> saved geom_{tag}_XY.png / _XYzoom.png / _XZ.png in {out_dir}/")
    return 0

def hybridbar_calculate_resonant_peaks():
    config = SimulationConfig()

    lambda_min_nm = 5700.0
    lambda_max_nm = 10300.0

    fmin = 1.0 / (lambda_max_nm / xm)
    fmax = 1.0 / (lambda_min_nm / xm)

    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    center_wavelength_nm = (1.0 / fcen) * xm  # około 7338.7 nm
    nfreq = 500
    config.resolution = 200
    config.lambda0 = center_wavelength_nm / xm
    config.frequency_width = 6.0 * df

    # Resolution convergence test: SAME geometry (gap=30, L_bar=2000,
    # L_tip=150, W=240) run at 200/250/300/350. With radius=0 the geometry is
    # mathematically identical at every resolution, so any shift in the
    # resonant wavelength is purely numerical convergence (not a change of the
    # simulated corner shape). Per-sweep "res" overrides config.resolution.
    sweeps = [
        {"name": "conv_res200", "gap": 30, "L_bar": 2000, "L_tip": 150, "W": 240, "res": 200},
        {"name": "conv_res250", "gap": 30, "L_bar": 2000, "L_tip": 150, "W": 240, "res": 250},
        {"name": "conv_res300", "gap": 30, "L_bar": 2000, "L_tip": 150, "W": 240, "res": 300},
        {"name": "conv_res350", "gap": 30, "L_bar": 2000, "L_tip": 150, "W": 240, "res": 350},
    ]

    results_filename = "results/resonant_peaks_hybrid_summary6.txt"
    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Geometria\tGap[nm]\tL_bar[nm]\tL_tip[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)

    for p in sweeps:
        mp.print_messages = False
        print_task(1, f"Szukanie rezonansu dla: {p['name']}")

        # per-sweep resolution override (falls back to the module default above)
        config.resolution = p.get("res", config.resolution)

        L_bar = p["L_bar"] / xm
        L_tip = p["L_tip"] / xm
        width = p["W"] / xm
        gap = p["gap"] / xm
        
        Th_Au = 30 / xm
        Th_Ti = 5 / xm
        Th_Sub = 100 / xm
        L_Sub = (p["gap"] + 2 * p["L_bar"] + p["L_tip"] + 400)/xm
        W_Sub = (p["W"] + 400)/xm
        # radius = 0 -> corected_gap == gap exactly and no apex patch is added,
        # i.e. the simulated gap is the gap you asked for. At resolution 200 the
        # 5 nm fillet is ONE pixel and buys no geometric fidelity, while its
        # corrected_gap already costs 20% of the gap - so screening runs use a
        # sharp tip and only the final showcase geometry is re-run rounded at
        # high resolution. Sharp corners are grid-singular, which is exactly
        # what FEF_mean (inner box, metal-adjacent pixels excluded) filters out.
        radius = TIP_RADIUS_NM / xm

        cg_nm, geom_flags = check_gap_geometry(
            p["gap"], p["W"], p["L_tip"], TIP_RADIUS_NM, config.resolution)

        AuTop = HybridBar(gap=gap, bar_length=L_bar, tip_length=L_tip, width=width, thickness=Th_Au, radius=radius, material=Au, z_offset=0.0)
        TiBetween = HybridBar(gap=gap, bar_length=L_bar, tip_length=L_tip, width=width, thickness=Th_Ti, radius=radius, material=Ti, z_offset=-(Th_Au + Th_Ti)/2.0)
        
        substrate = mp.Block(size=mp.Vector3(L_Sub, W_Sub, Th_Sub), center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)), material=SiO2)

        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

        config.pad = 200 / xm
        config.pml = 350 / xm
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,
            W_Sub + 2*config.pad + 2*config.pml,
            Th_Sub + AuTop.thickness + TiBetween.thickness + 2*config.pad + 2*config.pml
        ]
        
        cell = make_cell(config=config)
        
        config.src_size = [L_Sub, W_Sub, 0.0]
        config.src_center = [0.0, 0.0, config.cell_size[2]/2.0 - 1.15*config.pml]

        comm, MPI = _get_mpi_comm()

        # full box / inner box (fixed fraction of the gap, resolution-free) /
        # small centre box - see the notes at the top of this module
        dft_size, dft_size_inner, dft_size_c, probe = _gap_probe_sizes(
            gap, config.resolution)

        sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

        # run until the DFT fields converge (pulse + ring-down), instead of
        # a fixed time shorter than the source itself
        sim_empty.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
        if mp.am_master():
            print(f"EMPTY run finished at t = {sim_empty.meep_time():.1f}")

        empty_max, empty_mean, empty_center = _dft_gap_spectra(
            sim_empty, dft_empty, dft_empty_in, dft_empty_c, nfreq, comm, MPI)

        sim_empty.reset_meep()

        sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=geometry, sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_ant = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_ant_in = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
        if mp.am_master():
            print(f"ANTENNA run finished at t = {sim.meep_time():.1f}")

        ant_max, ant_mean, ant_center = _dft_gap_spectra(
            sim, dft_ant, dft_ant_in, dft_ant_c, nfreq, comm, MPI)

        sim.reset_meep()

        if mp.am_master():
            eps = 1e-16
            fef_max = ant_max**2 / (empty_max**2 + eps)          # hottest pixel (grid-singular)
            fef_mean = ant_mean / (empty_mean + eps)             # gap-averaged intensity
            fef_center = ant_center / (empty_center + eps) # gap centre point

            # peak POSITION from the gap-centre spectrum: it tracks the
            # dipolar gap resonance cleanly, unlike the gap-averaged one
            # whose rising short-wavelength background pushes argmax to the
            # window edge. Interior local-max search avoids boundary pinning.
            best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=np.sqrt(empty_center))
            best_freq = freqs[best_idx]
            best_wavelength_nm = (1.0 / best_freq) * xm

            # heights at the SAME refined position, not at the nearest bin -
            # bin-snapping under-reports exactly the sharpest (best) antennas
            v_mean = float(fef_mean[best_idx])
            v_center = float(fef_center[best_idx])
            v_max = float(fef_max[best_idx])

            flags = _result_flags(fef_mean, fef_center, fef_max,
                                  best_idx, at_boundary, geom_flags)

            warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
            print(f"--> ZNALEZIONO REZONANS: {best_wavelength_nm:.2f} nm "
                  f"(FEF_mean = {v_mean:.2f}, "
                  f"FEF_center = {v_center:.2f}, "
                  f"FEF_max_px = {v_max:.2f}){warn}")
            if flags:
                print(f"    [!] FLAGI: {' '.join(flags)}  <- wiersz podejrzany")

            import matplotlib.pyplot as plt

            wavelengths_nm = (1.0 / freqs) * xm

            plt.figure(figsize=(8, 5))
            plt.semilogy(wavelengths_nm, fef_mean, '-', color='darkred', linewidth=2, label='FEF mean (gap)')
            plt.semilogy(wavelengths_nm, fef_center, '-', color='darkblue', linewidth=1.5, label='FEF centre (peak pick)')
            plt.semilogy(wavelengths_nm, fef_max, '--', color='gray', linewidth=1.5, label='FEF max pixel')
            plt.plot(best_wavelength_nm, v_center, 'o', color='gold', markersize=8, markeredgecolor='black', label=f'Peak: {best_wavelength_nm:.1f} nm')

            plt.xlabel('Wavelength [nm]', fontsize=14)
            plt.ylabel('Field Enhancement Factor (FEF)', fontsize=14)
            plt.title(f'Resonance Spectrum: L_bar {p["L_bar"]} nm, L_tip {p["L_tip"]} nm, width {p["W"]} nm', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()

            plot_filename = os.path.join("results", f"spectrum_gap_{p['gap']}nm_Lbar_{p['L_bar']}nm_Ltip_{p['L_tip']}nm_W_{p['W']}nm_CHECK4.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"Zapisano wykres widma: {plot_filename}")

            with open(results_filename, "a") as f:
                # provenance columns: without them a row cannot be told apart
                # from a row measured with a different radius / resolution /
                # probe volume, which is exactly how summary1.txt went wrong.
                # plot_resonance.parse_rows strips everything after '#'.
                f.write(f"{p['name']}\t{p['gap']}\t{p['L_bar']}\t{p['L_tip']}\t{p['W']}\t"
                        f"{best_wavelength_nm:.2f}\t{v_mean:.2f}\t{v_center:.2f}\t{v_max:.2f}"
                        + _provenance(config.resolution, TIP_RADIUS_NM,
                                      cg_nm, probe, flags) + "\n")

    if mp.am_master():
        print_task(5, f" Wyniki zapisano w {results_filename}")
    return 0

def splitbar_calculate_resonant_peaks():
    """
    Resonant-wavelength scan for the SPLIT-BAR antenna, repeated for TWO substrates: SiO2 and Si.

    Same method as hybridbar_calculate_resonant_peaks: one broadband pulse +
    converged DFT (stop_when_dft_decayed), FEF = |E_ant|^2/|E_empty|^2 in the
    gap, peak position taken from the gap-centre spectrum. For every geometry
    the scan runs once per substrate, so you can read off the substrate shift
    directly (Si has a much higher index than SiO2 -> strong RED-shift).
    """
    config = SimulationConfig()

    lambda_min_nm = 5700.0
    lambda_max_nm = 10300.0

    fmin = 1.0 / (lambda_max_nm / xm)
    fmax = 1.0 / (lambda_min_nm / xm)

    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    center_wavelength_nm = (1.0 / fcen) * xm
    nfreq = 500
    config.resolution = 350   # lower to ~200-250 for screening
    config.lambda0 = center_wavelength_nm / xm
    config.frequency_width = 6.0 * df   # broad source; cancels in the ratio

    # Substrates to scan: (label, meep material). Si = crystalline silicon
    # (cSi). 
    substrates = [
        ("SiO2", SiO2),
        ("Si",   cSi),
    ]

    sweeps = [
        {"name": "SplitBar", "gap": 20, "L": 1800, "W": 240},
    ]

    results_filename = "results/resonant_peaks_splitbar_summary.txt"
    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Geometria\tSubstrate\tGap[nm]\tL[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)

    for sub_name, sub_material in substrates:
        for p in sweeps:
            mp.print_messages = False
            print_task(1, f"Szukanie rezonansu (split-bar, podloze {sub_name}): {p['name']} L={p['L']} W={p['W']}")

            L_arm = p["L"] / xm
            width = p["W"] / xm
            gap = p["gap"] / xm

            Th_Au = 30 / xm
            Th_Ti = 5 / xm
            Th_Sub = 100 / xm
            L_Sub = (p["gap"] + 2 * p["L"] + 400) / xm   # 2*L + gap + margin
            W_Sub = (p["W"] + 400) / xm
            # SplitBar has no tapered tip, so there is no fillet gap
            # correction here - the radius only rounds the bar corners and the
            # gap is used verbatim. check_gap_geometry is called without
            # width/tip_length, so it only verifies that the gap is resolved.
            radius = 5 / xm

            cg_nm, geom_flags = check_gap_geometry(
                p["gap"], resolution=config.resolution)

            AuTop = SplitBar(gap=gap, length=L_arm, width=width, thickness=Th_Au, radius=radius, material=Au, z_offset=0.0)
            TiBetween = SplitBar(gap=gap, length=L_arm, width=width, thickness=Th_Ti, radius=radius, material=Ti, z_offset=-(Th_Au + Th_Ti)/2.0)

            substrate = mp.Block(size=mp.Vector3(L_Sub, W_Sub, Th_Sub), center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)), material=sub_material)

            geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

            config.pad = 200 / xm
            config.pml = 350 / xm
            config.cell_size = [
                L_Sub + 2*config.pad + 2*config.pml,
                W_Sub + 2*config.pad + 2*config.pml,
                Th_Sub + AuTop.thickness + TiBetween.thickness + 2*config.pad + 2*config.pml
            ]

            cell = make_cell(config=config)

            config.src_size = [L_Sub, W_Sub, 0.0]
            config.src_center = [0.0, 0.0, config.cell_size[2]/2.0 - 1.15*config.pml]

            comm, MPI = _get_mpi_comm()

            dft_size, dft_size_inner, dft_size_c, probe = _gap_probe_sizes(
                gap, config.resolution)

            sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

            dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
            dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
            dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

            sim_empty.run(until_after_sources=mp.stop_when_dft_decayed(
                tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
            if mp.am_master():
                print(f"[{sub_name}] EMPTY run finished at t = {sim_empty.meep_time():.1f}")

            empty_max, empty_mean, empty_center = _dft_gap_spectra(
                sim_empty, dft_empty, dft_empty_in, dft_empty_c, nfreq, comm, MPI)

            sim_empty.reset_meep()

            sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=geometry, sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

            dft_ant = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
            dft_ant_in = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
            dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

            sim.run(until_after_sources=mp.stop_when_dft_decayed(
                tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
            if mp.am_master():
                print(f"[{sub_name}] ANTENNA run finished at t = {sim.meep_time():.1f}")

            ant_max, ant_mean, ant_center = _dft_gap_spectra(
                sim, dft_ant, dft_ant_in, dft_ant_c, nfreq, comm, MPI)

            sim.reset_meep()

            if mp.am_master():
                eps = 1e-16
                fef_max = ant_max**2 / (empty_max**2 + eps)
                fef_mean = ant_mean / (empty_mean + eps)
                fef_center = ant_center / (empty_center + eps)

                best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=np.sqrt(empty_center))
                best_freq = freqs[best_idx]
                best_wavelength_nm = (1.0 / best_freq) * xm
                flags = _result_flags(fef_mean, fef_center, fef_max,
                                      best_idx, at_boundary, geom_flags)
                warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
                print(f"--> [{sub_name}] REZONANS: {best_wavelength_nm:.2f} nm "
                      f"(FEF_mean = {fef_mean[best_idx]:.2f}, "
                      f"FEF_center = {fef_center[best_idx]:.2f}, "
                      f"FEF_max_px = {fef_max[best_idx]:.2f}){warn}")
                if flags:
                    print(f"    [!] FLAGI: {' '.join(flags)}  <- wiersz podejrzany")

                import matplotlib.pyplot as plt

                wavelengths_nm = (1.0 / freqs) * xm

                plt.figure(figsize=(8, 5))
                plt.semilogy(wavelengths_nm, fef_mean, '-', color='darkred', linewidth=2, label='FEF mean (gap)')
                plt.semilogy(wavelengths_nm, fef_center, '-', color='darkblue', linewidth=1.5, label='FEF centre (peak pick)')
                plt.semilogy(wavelengths_nm, fef_max, '--', color='gray', linewidth=1.5, label='FEF max pixel')
                plt.plot(best_wavelength_nm, fef_center[best_idx], 'o', color='gold', markersize=8, markeredgecolor='black', label=f'Peak: {best_wavelength_nm:.1f} nm')

                plt.xlabel('Wavelength [nm]', fontsize=14)
                plt.ylabel('Field Enhancement Factor (FEF)', fontsize=14)
                plt.title(f'Split-bar / {sub_name}: L {p["L"]} nm, W {p["W"]} nm, gap {p["gap"]} nm', fontsize=13)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                plt.tight_layout()

                plot_filename = os.path.join("results", f"spectrum_splitbar_{sub_name}_gap_{p['gap']}nm_L_{p['L']}nm_W_{p['W']}nm.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                print(f"Zapisano wykres widma: {plot_filename}")

                with open(results_filename, "a") as f:
                    f.write(f"{p['name']}\t{sub_name}\t{p['gap']}\t{p['L']}\t{p['W']}\t"
                            f"{best_wavelength_nm:.2f}\t{fef_mean[best_idx]:.2f}\t"
                            f"{fef_center[best_idx]:.2f}\t{fef_max[best_idx]:.2f}"
                            + _provenance(config.resolution, 5.0, cg_nm, probe, flags) + "\n")

    if mp.am_master():
        print_task(5, f" Wyniki zapisano w {results_filename}")
    return 0

def hybridbar_AuTiSiO2_opt():

    config = SimulationConfig()
    config.IMG_CLOSE = True

    tasks = [
        {"gap": 30, "L_bar": 2400, "L_tip": 200,  "W": 240, "Wavelength": 8476.11},
        # {"gap": 30, "L_bar": 1600, "L_tip": 150,  "W": 280, "Wavelength": 9784.18},
        # {"gap": 30, "L_bar": 1600, "L_tip": 200,  "W": 240, "Wavelength": 10300},
        # {"gap": 30, "L_bar": 1800, "L_tip": 150,  "W": 240, "Wavelength": 10300},
        # {"gap": 30, "L_bar": 1600, "L_tip": 100,  "W": 240, "Wavelength": 9421.25},
        #{"gap": 30, "L_bar": 1600, "L_tip": 200,  "W": 240, "Wavelength": 10300},
        #{"gap": 30, "L_bar": 1200, "L_tip": 150,  "W": 240, "Wavelength": 6340}, 
    ]

    for p in tasks:

        plt.close('all')

        config.resolution = 300
        config.sim_time = 12000 / xm      # unused by DFT maps (kept for other utils)
        config.sim_time_step = 50 / xm
        config.lambda0 = p["Wavelength"] / xm
        # moderate-bandwidth Gaussian pulse centred at the resonance; the
        # spectral envelope cancels in the DFT ant/empty ratio, so this only
        # sets the pulse duration (~10/fwidth), not the physics
        config.frequency_width = config.frequency
        gap = p["gap"]
        L_bar = p["L_bar"]/xm
        L_tip = p["L_tip"]/xm
        width = p["W"]/xm
        Th_Au = 30/xm
        Th_Ti = 5/xm
        Th_Sub = 100/xm
        L_Sub = (p["gap"] + 2 * p["L_bar"] + p["L_tip"] + 400)/xm
        W_Sub = (p["W"] + 400)/xm
        radius = 5 /xm

        SIM_NAME = f"HybridBar_gap_{gap}nm_Lbar_{p['L_bar']}nm_Ltip_{p['L_tip']}_W_{p['W']}nm_AuTiSiO2_res{config.resolution}_lambda_{config.lambda0}"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)

        # ====================================================

        AuTop = HybridBar(
            gap=gap/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Au, 
            material=Au, 
            z_offset=0.0,
            radius=radius
        )
        TiBetween = HybridBar(
            gap=gap/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Ti, 
            material=Ti, 
            z_offset=-(Th_Au + Th_Ti)/2.0,
            radius=radius
        )

        substrate = mp.Block(
                size=mp.Vector3(L_Sub, W_Sub, Th_Sub),
                center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)),
                material=SiO2
            )
        
        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

        config.pad = 200/xm
        config.pml = 500/xm
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,   # x
            W_Sub + 2*config.pad + 2*config.pml,   # y
            Th_Sub + AuTop.thickness + TiBetween.thickness+ 2*config.pad + 2*config.pml    # z
        ]
        cell = make_cell(config=config)

        config.src_size = [
            L_Sub,  # x
            W_Sub,  # y
            0.0 / xm    # z
        ]
        config.src_center = [
            0.0,    # x
            0.0,    # y
            config.cell_size[2]/2.0-1.15*config.pml  # z
        ]

        config.nfreq = 500
        # config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
        # config.z_transmission = -(config.cell_size[2]/2.0-1.15*config.pml)

        antenna_vols = VolumeSetROI(cell, antenna=AuTop)

        save_and_show_config(config, [AuTop, substrate])

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=geometry,
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        sim_empty = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=[],
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        
        print("Antenna bounding box:", np.array(AuTop.bounding_box())*1000, "\n")
        
        # =====================================================
        print_task(1, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_vis_{plane}.png"
            save_2D_plot(
                sim,
                antenna_vols.vis_volume[plane],
                save_name=Name2D,
                path_to_save=config.path_to_save,
                IMG_CLOSE=config.IMG_CLOSE
            )
        print_task(2, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_roi_{plane}.png"
            save_2D_plot(
                sim,
                antenna_vols.volume[plane],
                save_name=Name2D,
                path_to_save=config.path_to_save,
                IMG_CLOSE=config.IMG_CLOSE
            )
        # =====================================================
        print_task(3, "3D calculations - DFT enhancement maps at resonance.")
        # steady-state |E|^2 enhancement maps at lambda0, directly comparable
        # with the FEF values from hybridbar_calculate_resonant_peaks()
        max_enh = compute_dft_enhancement_maps(
            sim,
            sim_empty,
            antenna_vols,
            config,
            decay_tol=DFT_DECAY_TOL,
            max_run_time=DFT_MAX_RUN_TIME,
        )

        if mp.am_master() and max_enh:
            summary_path = os.path.join("results", "DFT_enhancement_summary.txt")
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a") as f:
                if write_header:
                    f.write("SIM_NAME\tWavelength[nm]\t" + "\t".join(max_enh.keys()) + "\n")
                f.write(SIM_NAME + f"\t{p['Wavelength']:.2f}\t"
                        + "\t".join(f"{v:.3f}" for v in max_enh.values()) + "\n")

        # NOTE: time-domain h5 dumps are no longer produced, so the
        # source-profile plots below would fail - disabled
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar-empty_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=0,
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_empty"
        # )
        # plot_signal_amplitude_vs_time_from_h5(
        #     "xyplanar_ex.h5",
        #     load_h5data_path=config.path_to_save,
        #     xzeros=0,
        #     time_step=config.sim_time_step,
        #     save_name=f"source_prof_antenna"
        # )

        sim.reset_meep()
        sim_empty.reset_meep()

        del sim
        del sim_empty

        import gc
        gc.collect()

        if mp.am_master():
            print("Pamięć zresetowana. Przechodzę do kolejnej anteny.")

    return 0

def bowtie_calculate_resonant_peaks():
    config = SimulationConfig()

    lambda_min_nm = 5700.0
    lambda_max_nm = 10300.0

    fmin = 1.0 / (lambda_max_nm / xm)
    fmax = 1.0 / (lambda_min_nm / xm)

    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    center_wavelength_nm = (1.0 / fcen) * xm  # około 7338.7 nm
    nfreq = 500
    config.resolution = 200
    config.lambda0 = center_wavelength_nm / xm
    config.frequency_width = 6.0 * df

    sweeps = [
        # {"name": "BowTie4", "gap": 30, "L": 500, "W": 300},
        # {"name": "BowTie4", "gap": 30, "L": 150, "W": 300},
        # {"name": "BowTie4", "gap": 30, "L": 300, "W": 300},
        # {"name": "BowTie4", "gap": 30, "L": 700, "W": 300},
        # {"name": "BowTie4", "gap": 30, "L": 900, "W": 300},
        # {"name": "BowTie4", "gap": 30, "L": 500, "W": 200},
        # {"name": "BowTie4", "gap": 30, "L": 500, "W": 400},
        # {"name": "BowTie4", "gap": 30, "L": 500, "W": 500},
        # {"name": "BowTie4", "gap": 30, "L": 500, "W": 700},
        # {"name": "BowTie4", "gap": 50, "L": 500, "W": 300},
        # {"name": "BowTie4", "gap": 70, "L": 500, "W": 300},
        # {"name": "BowTie4", "gap": 90, "L": 500, "W": 300},
        # {"name": "BowTie4", "gap": 10, "L": 500, "W": 300},
        {"name": "BowTie4", "gap": 30, "L": 1000, "W": 1000},

    ]

    results_filename = "results/resonant_peaks_bowtie_summary2.txt"
    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Geometria\tGap[nm]\tL[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)

    for params in sweeps:
        mp.print_messages = False 
        print_task(1, f"Szukanie rezonansu dla: {params['name']}")

        L_tri = params["L"] / xm
        width = params["W"] / xm
        gap = params["gap"] / xm
        Th_Au = 30 / xm
        Th_Ti = 5 / xm
        Th_Sub = 100 / xm
        L_Sub = (params["gap"] + 2 * params["L"] + 400)/xm
        W_Sub = (params["W"] + 400)/xm
        # see TIP_RADIUS_NM at the top of this module: bow-tie tips are much
        # sharper than the hybrid ones, so the fillet correction degenerates
        # even faster here (L=900/W=300 gives corrected_gap = -20.8 nm)
        radius = TIP_RADIUS_NM / xm

        cg_nm, geom_flags = check_gap_geometry(
            params["gap"], params["W"], params["L"], TIP_RADIUS_NM,
            config.resolution)

        AuTop = BowTie(gap=gap, length=L_tri, width=width, thickness=Th_Au, radius=radius, material=Au, z_offset=0.0)
        TiBetween = BowTie(gap=gap, length=L_tri, width=width, thickness=Th_Ti, radius=radius, material=Ti, z_offset=-(Th_Au + Th_Ti)/2.0)
        substrate = mp.Block(size=mp.Vector3(L_Sub, W_Sub, Th_Sub), center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)), material=SiO2)
        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

        config.pad = 200 / xm
        config.pml = 350 / xm
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,
            W_Sub + 2*config.pad + 2*config.pml,
            Th_Sub + AuTop.thickness + TiBetween.thickness + 2*config.pad + 2*config.pml
        ]
        
        cell = make_cell(config=config)
        
        config.src_size = [L_Sub, W_Sub, 0.0]
        config.src_center = [0.0, 0.0, config.cell_size[2]/2.0 - 1.15*config.pml]

        comm, MPI = _get_mpi_comm()

        # full box / inner box (fixed fraction of the gap, resolution-free) /
        # small centre box - see the notes at the top of this module
        dft_size, dft_size_inner, dft_size_c, probe = _gap_probe_sizes(
            gap, config.resolution)

        sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

        # run until the DFT fields converge (pulse + ring-down), instead of
        # a fixed time shorter than the source itself
        sim_empty.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
        if mp.am_master():
            print(f"EMPTY run finished at t = {sim_empty.meep_time():.1f}")

        empty_max, empty_mean, empty_center = _dft_gap_spectra(
            sim_empty, dft_empty, dft_empty_in, dft_empty_c, nfreq, comm, MPI)

        sim_empty.reset_meep()

        sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=geometry, sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_ant = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_ant_in = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_c)

        sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=DFT_DECAY_TOL, maximum_run_time=DFT_MAX_RUN_TIME))
        if mp.am_master():
            print(f"ANTENNA run finished at t = {sim.meep_time():.1f}")

        ant_max, ant_mean, ant_center = _dft_gap_spectra(
            sim, dft_ant, dft_ant_in, dft_ant_c, nfreq, comm, MPI)

        sim.reset_meep()

        if mp.am_master():
            eps = 1e-16
            fef_max = ant_max**2 / (empty_max**2 + eps)          # hottest pixel (grid-singular)
            fef_mean = ant_mean / (empty_mean + eps)             # gap-averaged intensity
            fef_center = ant_center / (empty_center + eps) # gap centre point

            # peak POSITION from the gap-centre spectrum: it tracks the
            # dipolar gap resonance cleanly, unlike the gap-averaged one
            # whose rising short-wavelength background pushes argmax to the
            # window edge. Interior local-max search avoids boundary pinning.
            best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=np.sqrt(empty_center))
            best_freq = freqs[best_idx]
            best_wavelength_nm = (1.0 / best_freq) * xm
            flags = _result_flags(fef_mean, fef_center, fef_max,
                                  best_idx, at_boundary, geom_flags)
            warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
            print(f"--> ZNALEZIONO REZONANS: {best_wavelength_nm:.2f} nm "
                  f"(FEF_mean = {fef_mean[best_idx]:.2f}, "
                  f"FEF_center = {fef_center[best_idx]:.2f}, "
                  f"FEF_max_px = {fef_max[best_idx]:.2f}){warn}")
            if flags:
                print(f"    [!] FLAGI: {' '.join(flags)}  <- wiersz podejrzany")

            import matplotlib.pyplot as plt

            wavelengths_nm = (1.0 / freqs) * xm

            plt.figure(figsize=(8, 5))
            plt.semilogy(wavelengths_nm, fef_mean, '-', color='darkred', linewidth=2, label='FEF mean (gap)')
            plt.semilogy(wavelengths_nm, fef_center, '-', color='darkblue', linewidth=1.5, label='FEF centre (peak pick)')
            plt.semilogy(wavelengths_nm, fef_max, '--', color='gray', linewidth=1.5, label='FEF max pixel')
            plt.plot(best_wavelength_nm, fef_center[best_idx], 'o', color='gold', markersize=8, markeredgecolor='black', label=f'Peak: {best_wavelength_nm:.1f} nm')

            plt.xlabel('Wavelength [nm]', fontsize=14)
            plt.ylabel('Field Enhancement Factor (FEF)', fontsize=14)
            plt.title(f'Resonance Spectrum: length {params["L"]} nm, width {params["W"]} nm', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()

            plot_filename = os.path.join("results", f"spectrum_BowTie_gap_{params['gap']}nm_L_{params['L']}nm_W_{params['W']}nm_CHECK4.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"Zapisano wykres widma: {plot_filename}")

            with open(results_filename, "a") as f:
                f.write(f"{params['name']}\t{params['gap']}\t{params['L']}\t{params['W']}\t"
                        f"{best_wavelength_nm:.2f}\t{fef_mean[best_idx]:.2f}\t"
                        f"{fef_center[best_idx]:.2f}\t{fef_max[best_idx]:.2f}"
                        + _provenance(config.resolution, TIP_RADIUS_NM,
                                      cg_nm, probe, flags) + "\n")

    if mp.am_master():
        print_task(5, f" Wyniki zapisano w {results_filename}")
    return 0

def bowtie_AuTiSiO2_opt():

    config = SimulationConfig()

    tasks = [
        # {"gap": 30, "L": 600, "W": 300, "Wavelength": 6912}
        # {"gap": 30, "L": 900, "W": 1000, "Wavelength": 4849},
        {"gap": 30, "L": 1000, "W": 1000, "Wavelength": 4939},  
        {"gap": 30, "L": 1100, "W": 800, "Wavelength": 6146.72},
        {"gap": 30, "L": 1100, "W": 900, "Wavelength": 6130.94},
        {"gap": 30, "L": 1100, "W": 1000, "Wavelength": 6146.72},  
    ]

    for p in tasks:

        plt.close('all')

        config.resolution = 350
        config.sim_time = 12000 / xm      # unused by DFT maps (kept for other utils)
        config.sim_time_step = 50 / xm
        config.lambda0 = p["Wavelength"] / xm
        # moderate-bandwidth Gaussian pulse centred at the resonance; the
        # spectral envelope cancels in the DFT ant/empty ratio, so this only
        # sets the pulse duration (~10/fwidth), not the physics
        config.frequency_width = config.frequency
        gap = p["gap"]
        L_tri = p["L"]/xm
        width = p["W"]/xm
        Th_Au = 30/xm
        Th_Ti = 5/xm
        Th_Sub = 100/xm
        # BUGFIX: previously referenced p["L_bar"]/p["L_tip"] (hybridbar
        # keys); bowtie total length = 2*L + gap
        L_Sub = (p["gap"] + 2 * p["L"] + 400)/xm
        W_Sub = (p["W"] + 400)/xm
        radius = 5 /xm

        SIM_NAME = f"BowTie_gap_{gap}nm_L_{p['L']}nm_W_{p['W']}nm_AuTiSiO2_res{config.resolution}_lambda_{config.lambda0}"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)

        # =====================================================
        AuTop = BowTie(
            gap=gap/xm, 
            length = L_tri,
            width=width, 
            thickness=Th_Au, 
            radius=radius,
            material=Au, 
            z_offset=0.0
        )
        TiBetween = BowTie(
            gap=gap/xm, 
            length = L_tri, 
            width=width, 
            thickness=Th_Ti, 
            radius = radius,
            material=Ti, 
            z_offset=-(Th_Au + Th_Ti)/2.0
        )

        substrate = mp.Block(
                size=mp.Vector3(L_Sub, W_Sub, Th_Sub),
                center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)),
                material=SiO2
            )
        
        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

        config.pad = 200/xm
        config.pml = 500/xm
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,   # x
            W_Sub + 2*config.pad + 2*config.pml,   # y
            Th_Sub+AuTop.thickness + TiBetween.thickness+ 2*config.pad + 2*config.pml    # z
        ]
        cell = make_cell(config=config)

        config.src_size = [
            L_Sub,  # x
            W_Sub,  # y
            0.0 / xm    # z
        ]
        config.src_center = [
            0.0,    # x
            0.0,    # y
            config.cell_size[2]/2.0-1.15*config.pml  # z
        ]

        config.nfreq = 500
        # config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
        # config.z_transmission = -(config.cell_size[2]/2.0-1.15*config.pml)

        antenna_vols = VolumeSetROI(cell, antenna=AuTop)

        save_and_show_config(config, [AuTop, substrate])

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=geometry,
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        sim_empty = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=[],
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        
        print("Antenna bounding box:", np.array(AuTop.bounding_box())*1000, "\n")
        
        # =====================================================
        print_task(1, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_vis_{plane}.png"
            save_2D_plot(
                sim,
                antenna_vols.vis_volume[plane],
                save_name=Name2D,
                path_to_save=config.path_to_save,
                IMG_CLOSE=config.IMG_CLOSE
            )
        print_task(2, "2D projections.")
        for plane in ["XY", "XZ", "YZ"]:
            Name2D = f"antenna_roi_{plane}.png"
            save_2D_plot(
                sim,
                antenna_vols.volume[plane],
                save_name=Name2D,
                path_to_save=config.path_to_save,
                IMG_CLOSE=config.IMG_CLOSE
            )
        # =====================================================
        print_task(3, "3D calculations - DFT enhancement maps at resonance.")
        # steady-state |E|^2 enhancement maps at lambda0, directly comparable
        # with the FEF values from bowtie_calculate_resonant_peaks()
        max_enh = compute_dft_enhancement_maps(
            sim,
            sim_empty,
            antenna_vols,
            config,
            decay_tol=DFT_DECAY_TOL,
            max_run_time=DFT_MAX_RUN_TIME,
        )

        if mp.am_master() and max_enh:
            summary_path = os.path.join("results", "DFT_enhancement_summary.txt")
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a") as f:
                if write_header:
                    f.write("SIM_NAME\tWavelength[nm]\t" + "\t".join(max_enh.keys()) + "\n")
                f.write(SIM_NAME + f"\t{p['Wavelength']:.2f}\t"
                        + "\t".join(f"{v:.3f}" for v in max_enh.values()) + "\n")

        sim.reset_meep()
        sim_empty.reset_meep()

        del sim
        del sim_empty
    
        import gc
        gc.collect()
        
        if mp.am_master():
            print("Pamięć zresetowana. Przechodzę do kolejnej anteny.")
        
    return 0

def postprocess_dft_efe():
    """
    Lokalny postprocessing wynikow z compute_dft_enhancement_maps. Renderuje mapy PNG i zapisuje srednie/maksymalne
    wzmocnienie w przerwie do pliku txt.
    """
    simulations = [
        # {"folder": "HybridBar_gap_30nm_Lbar_2400nm_Ltip_200_W_240nm_AuTiSiO2_res300_lambda_8.47611", "gap": 30},
    ]

    Th_Au = 30.0  # nm
    Th_Ti = 5.0   # nm

    results_filename = "results/Hybrid/DFT_EFE_summary_2400x200x240.txt"

    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Folder\tGap[nm]\t"
                "meanROI_XY\tmaxROI_XY\tmax_XY\t"
                "meanROI_XYTOP\tmaxROI_XYTOP\tmax_XYTOP\t"
                "meanROI_XZ\tmaxROI_XZ\tmax_XZ\t"
                "meanROI_YZ\tmaxROI_YZ\tmax_YZ\n")

    for sim in simulations:
        folder = sim["folder"]
        gap_nm = sim["gap"]

        load_path = os.path.join("results", folder)
        if not os.path.exists(load_path):
            print(f"Brak folderu {folder}, pomijam...")
            continue

        print_task(1, f"Postprocessing map DFT dla: {folder}")

        # ROI w nm: przerwa miedzy ramionami (XY) oraz przekroje przez
        # warstwy Au+Ti (XZ, YZ); z=0 to srodek warstwy Au
        roi_nm = {
            "XY": {"center": (0.0, 0.0), "width": gap_nm, "height": 10.0},
            "XZ": {"center": (0.0, -Th_Ti/2.0), "width": gap_nm, "height": Th_Au + Th_Ti},
            "YZ": {"center": (0.0, -Th_Ti/2.0), "width": 10.0, "height": Th_Au + Th_Ti},
        }

        stats = postprocess_dft_enhancement_maps(load_path, roi_nm=roi_nm)

        if mp.am_master() and stats:
            row = [folder, str(gap_nm)]
            for plane in ["xyplanar", "xyplanarTOP", "xzplanar", "yzplanar"]:
                s = stats.get(plane, {})
                row.append(f"{s.get('mean_roi', 0.0):.3f}")
                row.append(f"{s.get('max_roi', 0.0):.3f}")
                row.append(f"{s.get('max', 0.0):.3f}")
            with open(results_filename, "a") as f:
                f.write("\t".join(row) + "\n")

    if mp.am_master():
        print_task(5, f"Wyniki zapisano w {results_filename}")
    return 0

def postprocess_hybrid_efe():
    config = SimulationConfig()
    
    # Zmieniłem klucze w słowniku z "L" i "T" na "L_bar" i "L_tip" dla spójności
    simulations = [
        {"folder": "HybridBar_gap_30nm_Lbar_1600nm_Ltip_150_W_240nm_AuTiSiO2_res350_lambda_9.57313", "gap": 30, "L_bar": 1600, "L_tip": 150, "W": 240},
        {"folder": "HybridBar_gap_30nm_Lbar_1400nm_Ltip_150_W_240nm_AuTiSiO2_res350_lambda_6.90951", "gap": 30, "L_bar": 1400, "L_tip": 150, "W": 240},
        {"folder": "HybridBar_gap_30nm_Lbar_1200nm_Ltip_150_W_240nm_AuTiSiO2_res350_lambda_6.34031", "gap": 30, "L_bar": 1200, "L_tip": 150, "W": 240},
    ]

    results_filename = "results/EFE_summary_hybrid.txt"
    
    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Folder\tGap[nm]\tL_bar[nm]\tL_tip[nm]\tW[nm]\tEFE_XY\tEFE_XZ\tEFE_YZ\n")

    for sim in simulations:
        folder = sim["folder"]
        gap_nm = sim["gap"]
        L_bar_nm = sim["L_bar"]
        L_tip_nm = sim["L_tip"]
        width_nm = sim["W"]
        
        print_task(1, f"Przetwarzanie post-processing dla: {folder}")

        config.path_to_save = os.path.join("results", folder)
        config.animations_folder_path = os.path.join(config.path_to_save, "animations")
        
        if not os.path.exists(config.path_to_save):
            print(f"Brak folderu {folder}, pomijam...")
            continue

        config.IMG_CLOSE = True 

        L_bar = L_bar_nm / xm
        L_tip = L_tip_nm / xm
        width = width_nm / xm
        Th_Au = 30 / xm
        Th_Ti = 5 / xm
        Th_Sub = 100 / xm
        
        L_Sub = (gap_nm + 2 * L_bar_nm + L_tip_nm + 400) / xm
        W_Sub = (width_nm + 400) / xm
        radius = 5 / xm

        AuTop = HybridBar(
            gap=gap_nm/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Au, 
            material=Au, 
            z_offset=0.0,
            radius=radius
        )

        TiBetween = HybridBar(
            gap=gap_nm/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Ti, 
            material=Ti, 
            z_offset=-(Th_Au + Th_Ti)/2.0,
            radius=radius
        )
        
        config.pad = 200 / xm
        config.pml = 500 / xm
        
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,
            W_Sub + 2*config.pad + 2*config.pml,
            Th_Sub + AuTop.thickness + TiBetween.thickness + 2*config.pad + 2*config.pml
        ]
        
        cell = make_cell(config=config)

        antenna_vols = VolumeSetROI(cell, antenna=AuTop)
        
        roi_w = gap_nm
        roi_h = 5
        
        draw_params = {
            "XY": {"x_zoom": 0.05, "y_zoom": 0.2, 
                   "roi": {"center": (0, 0), "width": roi_w, "height": roi_h}}, # Poprawiono środek z (-1, 0) na (0, 0) - centrum szczeliny
            "XZ": {"x_zoom": 0.05, "y_zoom": 0.8, 
                   "roi": {"center": (0, -1e3*TiBetween.thickness/2.0), "width": roi_w, "height": (AuTop.thickness + TiBetween.thickness) * 1e3}},
            "YZ": {"x_zoom": 0.25, "y_zoom": 0.8, 
                   "roi": {"center": (0, -1e3*TiBetween.thickness/2.0), "width": roi_h, "height": (AuTop.thickness + TiBetween.thickness) * 1e3}},
        }

        max_enh = animate_enhancement_fields(config=config, volumes=antenna_vols, draw_params=draw_params, animate=False)
        
        if mp.am_master() and max_enh:
            efe_xy = max_enh.get("XY", 0)
            efe_xz = max_enh.get("XZ", 0)
            efe_yz = max_enh.get("YZ", 0)
            
            with open(results_filename, "a") as f:
                f.write(f"{folder}\t{gap_nm}\t{L_bar_nm}\t{L_tip_nm}\t{width_nm}\t{efe_xy:.3f}\t{efe_xz:.3f}\t{efe_yz:.3f}\n")

    return 0

def postprocess_bowties_efe():
    config = SimulationConfig()
    
    simulations = [
        # {"folder": "BowTie_gap_10nm_gap_10nm_AuTiSiO2_res300",     "gap": 10, "L": 500, "W": 300},
        # {"folder": "BowTie_gap_30nm_lenghts_300nm_AuTiSiO2_res300", "gap": 30, "L": 300, "W": 300},
        # {"folder": "BowTie_gap_30nm_lenghts_400nm_AuTiSiO2_res300", "gap": 30, "L": 400, "W": 300},
        # {"folder": "BowTie_gap_30nm_widths_300nm_AuTiSiO2_res300",  "gap": 30, "L": 500, "W": 300},
        # {"folder": "BowTie_gap_30nm_widths_400nm_AuTiSiO2_res300",  "gap": 30, "L": 500, "W": 400},
        # {"folder": "BowTie_gap_30nm_widths_500nm_AuTiSiO2_res300",  "gap": 30, "L": 500, "W": 500},
        # {"folder": "BowTie_gap_50nm_gaps_50nm_AuTiSiO2_res300",     "gap": 50, "L": 500, "W": 300},
        # {"folder": "BowTie_gap_70nm_gaps_70nm_AuTiSiO2_res300",     "gap": 70, "L": 500, "W": 300},
        {"folder": "BowTie_gap_30nm_L_1000nm_W_800nm_AuTiSiO2_res400_lambda_4.849", "gap": 30, "L": 1000, "W": 800},
    ]

    results_filename = "results/EFE_summary.txt"
    
    if mp.am_master():
        results_filename = _init_results_file(
            results_filename,
            "Folder\tGap[nm]\tL[nm]\tW[nm]\tEFE_XY\tEFE_XZ\tEFE_YZ\n")

    for sim in simulations:
        folder = sim["folder"]
        gap = sim["gap"]
        L = sim["L"]
        W = sim["W"]
        
        print_task(1, f"Przetwarzanie post-processing dla: {folder}")

        config.path_to_save = os.path.join("results", folder)
        config.animations_folder_path = os.path.join(config.path_to_save, "animations")
        
        if not os.path.exists(config.path_to_save):
            print(f"Brak folderu {folder}, pomijam...")
            continue

        config.IMG_CLOSE = True 

        L_tri = L / xm
        width = W / xm
        Th_Au = 30 / xm
        Th_Ti = 5 / xm
        Th_Sub = 100 / xm
        L_Sub = 1500 / xm
        W_Sub = 1000 / xm
        radius = 5 / xm

        AuTop = BowTie(gap=gap/xm, length=L_tri, width=width, thickness=Th_Au, radius=radius, material=Au, z_offset=0.0)
        TiBetween = BowTie(gap=gap/xm, length=L_tri, width=width, thickness=Th_Ti, radius=radius, material=Ti, z_offset=-(Th_Au + Th_Ti)/2.0)
        
        config.pad = 200 / xm
        config.pml = 500 / xm
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,
            W_Sub + 2*config.pad + 2*config.pml,
            Th_Sub + AuTop.thickness + 2*config.pad + 2*config.pml
        ]
        cell = make_cell(config=config)
        antenna_vols = VolumeSetROI(cell, antenna=AuTop)

        roi_w = gap
        roi_h = 10
        
        draw_params = {
            "XY": {"x_zoom": 0.4, "y_zoom": 0.4, 
                   "roi": {"center": (0, 0), "width": roi_w, "height": roi_h}},
            "XZ": {"x_zoom": 0.5, "y_zoom": 0.9, 
                   "roi": {"center": (0, -1e3*TiBetween.thickness/2.0), "width": roi_w, "height": (AuTop.thickness + TiBetween.thickness) * 1e3}},
            "YZ": {"x_zoom": 0.5, "y_zoom": 0.9, 
                   "roi": {"center": (0, -1e3*TiBetween.thickness/2.0), "width": roi_h, "height": (AuTop.thickness + TiBetween.thickness) * 1e3}},
        }

        max_enh = animate_enhancement_fields(config=config, volumes=antenna_vols, draw_params=draw_params, animate=False)
        
        if mp.am_master() and max_enh:
            efe_xy = max_enh.get("XY", 0)
            efe_xz = max_enh.get("XZ", 0)
            efe_yz = max_enh.get("YZ", 0)
            
            with open(results_filename, "a") as f:
                f.write(f"{folder}\t{gap}\t{L}\t{W}\t{efe_xy:.3f}\t{efe_xz:.3f}\t{efe_yz:.3f}\n")

    return 0

def bowtie_AuTiSiO2_opt_test():

    config = SimulationConfig()

    config.resolution = 400
    config.sim_time = 12000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 6000 / xm
    config.frequency_width = 1.0
    gap = 10

    plt.close('all')

    SIM_NAME = f"test_BowTie_gap_{gap}nm_AuTiSiO2_res_{config.resolution}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)

    L_tri = 100/xm
    width = 100/xm
    Th_Au = 30/xm
    Th_Ti = 5/xm
    Th_Sub = 100/xm
    L_Sub = 800/xm
    W_Sub = 800/xm
    radius = 5 /xm

    # =====================================================
    AuTop = BowTie(
        gap=gap/xm, 
        length = L_tri,
        width=width, 
        thickness=Th_Au, 
        radius=radius,
        material=Au, 
        z_offset=0.0
    )
    TiBetween = BowTie(
        gap=gap/xm, 
        length = L_tri, 
        width=width, 
        thickness=Th_Ti, 
        radius = radius,
        material=Ti, 
        z_offset=-(Th_Au + Th_Ti)/2.0
    )
    substrate = mp.Block(
            size=mp.Vector3(L_Sub, W_Sub, Th_Sub),
            center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)),
            material=SiO2
        )
    
    geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

    config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        L_Sub + 2*config.pad + 2*config.pml,   # x
        W_Sub + 2*config.pad + 2*config.pml,   # y
        Th_Sub+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        L_Sub,  # x
        W_Sub,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    # config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    # config.z_transmission = -(config.cell_size[2]/2.0-1.15*config.pml)

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)

    save_and_show_config(config, [AuTop, substrate])

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    
    print("Antenna bounding box:", np.array(AuTop.bounding_box())*1000, "\n")
    
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    # print_task(3, "3D calculations.")
    # compute_fields(
    #     sim, 
    #     sim_empty, 
    #     antenna_vols, 
    #     config, 
    #     fluxes = False,
    #     scattering = False,
    #     dft_gap_spectrum = False,
    #     harminv = False,
    #     scattering_antenna=AuTop
    # )
    
    # =====================================================
    print_task(4, "Postprocesing - raw animations for X.")
    animate_raw_fields(config=config, mode="BOTH", component="X")
    # =====================================================
    print_task(4, "Postprocesing - raw animations for Y.")
    animate_raw_fields(config=config, mode="BOTH", component="Y")
    # =====================================================
    print_task(4, "Postprocesing - raw animations for Z.")
    animate_raw_fields(config=config, mode="BOTH", component="Z")
    # =====================================================

    roi_w = gap
    roi_h = 15

    # Postprocessing - gap zoom
    draw_params = {
            "XY": {"x_zoom": 0.5,
                "y_zoom": 0.5,
                "roi": {
                        "center": (0, 0),
                        "width": roi_w,
                        "height": roi_h,
                    },
            },
            "XZ": {"x_zoom": 1,
                "y_zoom": 1,
                "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": roi_w,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
            "YZ": {"x_zoom": 1,
                "y_zoom": 1,
                "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": roi_h,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
    }

    print_task(5, "Postprocesing - animations and plots.")
    animate_enhancement_fields(config=config, volumes=antenna_vols, draw_params=draw_params, animate=False)
    # =====================================================
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar-empty_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=0,
        time_step=config.sim_time_step,
        save_name=f"source_prof_empty"
    )
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=0,
        time_step=config.sim_time_step,
        save_name=f"source_prof_antenna"
    )
        
    return 0

def experiment_hybrid_bar_AuTiSiO2():
    config = SimulationConfig()
    config.resolution = 400
    config.IMG_CLOSE = True

    base = {"gap": 10, "L": 1600, "W": 240, "T": 150}
    
    sweeps = [
            ("gap" , [10, 20, 30, 50]),
            ("length", [1400, 1600, 1800]),
            ("width", [200, 240, 280]),
            ("tip_length", [100, 150, 200])
    ]

    tasks = []
    seen_configs = []

    for param_name, values in sweeps:
        for val in values:
            p = base.copy()
            if param_name == "gap": p["gap"] = val
            elif param_name == "length": p["L"] = val
            elif param_name == "width": p["W"] = val 
            elif param_name == "tip_length": p["T"] = val 
            if p not in seen_configs:
                seen_configs.append(p)
                tasks.append((param_name, val, p))

    for p_name, p_val, params in tasks:
        plt.close('all')

        SIM_NAME = f"HybridBar_{p_name}_{p_val}nm_AuTiSiO2_res_{config.resolution}"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)

        L_bar = params["L"]/xm
        L_tip = params["T"]/xm
        width = params["W"]/xm
        Th_Au = 30/xm
        Th_Ti = 5/xm
        Th_Sub = 70/xm

        AuTop = HybridBar(
            gap=params["gap"]/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Au, 
            material=Au, 
            z_offset=0.0
        )

        TiBetween = HybridBar(
            gap=params["gap"]/xm, 
            bar_length=L_bar, 
            tip_length=L_tip,
            width=width, 
            thickness=Th_Ti, 
            material=Ti, 
            z_offset=-(Th_Au + Th_Ti)/2.0
        )
        substrate = mp.Block(
            size=mp.Vector3(4000/xm, 400/xm, Th_Sub),
            center=mp.Vector3(0, 0, -(Th_Au/2.0 + Th_Ti + Th_Sub/2.0)),
            material=SiO2
        )

        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + [substrate]

        config.pad = 100/xm
        config.pml = 100/xm
        total_len = (L_bar + L_tip) * 2 + (params["gap"]/xm)
        
        config.cell_size = [
            total_len + 2*config.pad + 2*config.pml,
            width + 2*config.pad + 2*config.pml,
            (Th_Au + Th_Ti + Th_Sub) + 2*config.pad + 2*config.pml
        ]
        cell = make_cell(config=config)

        config.src_center = [0, 0, config.cell_size[2]/2.0 - 1.1*config.pml]
        config.src_size = [total_len, width, 0]

        antenna_vols = VolumeSet(cell, antenna=AuTop, top_z=AuTop.thickness)

        save_and_show_config(config, [AuTop, TiBetween, substrate])

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=geometry,
            sources=make_source(config),
            resolution=config.resolution,
            symmetries=config.symmetries,
            dimensions=3
        )

        sim_empty = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=[],
            sources=make_source(config),
            resolution=config.resolution,
            symmetries=config.symmetries,
            dimensions=3
        )

        print_task(1, "Saving 2D Geometry Projections.")
        for plane in ["XY", "XZ"]:
            save_2D_plot(sim, antenna_vols.vis_volume[plane], 
                        save_name=f"geom_{plane}.png", path_to_save=config.path_to_save,
                        IMG_CLOSE=config.IMG_CLOSE)

        print_task(3, "Running 3D Field Calculations.")
        compute_fields(sim, sim_empty, antenna_vols, config, scattering = False, scattering_antenna=AuTop)

        roi_w = params["gap"]
        roi_h = 15

        # Postprocessing - gap zoom
        draw_params = {
                "XY": {"x_zoom": 0.10,
                    "y_zoom": 0.6,
                    "roi": {
                            "center": (0, 0),
                            "width": roi_w,
                            "height": roi_h,
                        },
                },
                "XZ": {"x_zoom": 0.1,
                    "y_zoom": 0.2,
                    "roi": {
                            "center": (0, -1e3*TiBetween.thickness/2.0),
                            "width": roi_w,
                            "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                        },
                },
                "YZ": {"x_zoom": 0.4,
                    "y_zoom": 0.2,
                    "roi": {
                            "center": (0, -1e3*TiBetween.thickness/2.0),
                            "width": roi_h,
                            "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                        },
                },
            }
        
        print_task(5, "Generating Enhancement Maps.")
        animate_enhancement_fields(config=config, draw_params=draw_params)

    return 0

def bowtie_substrate_experiment(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 18000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 660 / xm
    config.frequency_width = 1.0
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"NS_smallsrc_bigdet_F_BSE_Au_on_{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTieEquilateral(
        gap=gap/xm,
        length=86.6/xm, # <- to have about 100 nm in width
        thickness=30/xm,
        radius=5/xm,
        material=Au,
        z_offset=0.0
    )
    substrate = Bar(
        length=800/xm,
        width=800/xm,
        thickness=100/xm,
        material=X_material,
        z_offset=-(30/2.0+100/2.0)/xm,
        radius=12/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()
    geometry_empty = substrate.build_geometry()

    config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        substrate.length + 2*config.pad + 2*config.pml,   # x
        substrate.width + 2*config.pad + 2*config.pml,   # y
        substrate.thickness+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        substrate.length,  # x
        substrate.width,  # y
        0.0 / xm    # z
    ]

    #!!!
    config.src_is_integrated = True
    
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -config.cell_size[2]/2.0+config.pml+15/xm

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    # =====================================================
    save_and_show_config(config, [AuTop, substrate])
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_vis_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_roi_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(
        sim,
        sim_empty,
        antenna_vols,
        config,
        fluxes=True,
        # fluxes_X_size=substrate.length/2.0,
        # fluxes_Y_size=substrate.width/2.0,
        fluxes_X_size=config.cell_size[0],
        fluxes_Y_size=config.cell_size[1],
        scattering=True,
        dft_gap_spectrum=True,
        harminv=True,
        scattering_antenna=AuTop
    )
    return 0

def bowtie_big_substrate_experiment(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 18000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 660 / xm
    config.frequency_width = 1.0
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"NGeo_smallsrc_bigdet_F_BSE_Au_on_{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTieEquilateral(
        gap=gap/xm,
        length=86.6/xm, # <- to have about 100 nm in width
        thickness=30/xm,
        radius=5/xm,
        material=Au,
        z_offset=0.0
    )
    substrate = Bar(
        length=1000/xm,
        width=1000/xm,
        thickness=520/xm,
        material=X_material,
        z_offset=-(30/2.0+520/2.0)/xm,
        radius=0/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()
    geometry_empty = substrate.build_geometry()

    config.pad = 0
    # config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        substrate.length,   # x
        substrate.width,   # y
        substrate.thickness+AuTop.thickness + 1.5*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        AuTop.length*3,  # x
        AuTop.length*3,  # y
        0.0 / xm    # z
    ]

    #!!!
    config.src_is_integrated = True
    
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.05*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -config.cell_size[2]/2.0+config.pml+15/xm

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry_empty,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    # =====================================================
    save_and_show_config(config, [AuTop, substrate])
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_vis_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_roi_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # # =====================================================
    # print_task(3, "3D calculations.")
    # compute_fields(
    #     sim,
    #     sim_empty,
    #     antenna_vols,
    #     config,
    #     fluxes=True,
    #     # fluxes_X_size=substrate.length/2.0,
    #     # fluxes_Y_size=substrate.width/2.0,
    #     fluxes_X_size=config.cell_size[0],
    #     fluxes_Y_size=config.cell_size[1],
    #     scattering=True,
    #     dft_gap_spectrum=True,
    #     harminv=True,
    #     scattering_antenna=AuTop
    # )
    return 0

def bowtie_substrate_experiment_LT(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 25000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 660 / xm
    config.frequency_width = 1.0
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"BSELT_Au{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTieEquilateral(
        gap=gap/xm,
        length=86.6/xm, # <- to have about 100 nm in width
        thickness=30/xm,
        radius=5/xm,
        material=Au,
        z_offset=0.0
    )
    substrate = Bar(
        length=800/xm,
        width=800/xm,
        thickness=100/xm,
        material=X_material,
        z_offset=-(30/2.0+100/2.0)/xm,
        radius=12/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()

    config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        substrate.length + 2*config.pad + 2*config.pml,   # x
        substrate.width + 2*config.pad + 2*config.pml,   # y
        substrate.thickness+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        substrate.length,  # x
        substrate.width,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -config.cell_size[2]/2.0+config.pml+15/xm

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    # =====================================================
    save_and_show_config(config, [AuTop, substrate])
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_vis_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_roi_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(
        sim,
        sim_empty,
        antenna_vols,
        config,
        fluxes=True,
        scattering=True,
        dft_gap_spectrum=True,
        harminv=True,
        scattering_antenna=AuTop
    )
    return 0

def bowtie_substrate_experiment_MIR(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 40000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 1700 / xm
    config.frequency_width = 0.4
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"BSEMIR_Au{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTie(
        gap=gap/xm,
        length=86.6/xm*2.5, # <- to have about 100 nm in width
        width=100/xm,
        thickness=30/xm,
        radius=5/xm,
        material=Au,
        z_offset=0.0
    )
    substrate = Bar(
        length=800/xm,
        width=800/xm,
        thickness=100/xm,
        material=X_material,
        z_offset=-(30/2.0+100/2.0)/xm,
        radius=12/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()

    config.pad = 80/xm
    config.pml = 700/xm
    config.cell_size = [
        substrate.length + 2*config.pad + 2*config.pml,   # x
        substrate.width + 2*config.pad + 2*config.pml,   # y
        substrate.thickness+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        substrate.length,  # x
        substrate.width,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -config.cell_size[2]/2.0+config.pml+15/xm

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    # =====================================================
    save_and_show_config(config, [AuTop, substrate])
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_vis_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_roi_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(
        sim,
        sim_empty,
        antenna_vols,
        config,
        fluxes=True,
        scattering=True,
        dft_gap_spectrum=True,
        harminv=True,
        scattering_antenna=AuTop
    )
    return 0

def bowtie_substrate_ONLY_experiment(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 18000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 660 / xm
    config.frequency_width = 1.0
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"BSOE_AIR{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTieEquilateral(
        gap=gap/xm,
        length=86.6/xm, # <- to have about 100 nm in width
        thickness=30/xm,
        radius=5/xm,
        material=mp.air,
        z_offset=0.0
    )
    substrate = Bar(
        length=800/xm,
        width=800/xm,
        thickness=100/xm,
        material=X_material,
        z_offset=-(30/2.0+100/2.0)/xm,
        radius=12/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()

    config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        substrate.length + 2*config.pad + 2*config.pml,   # x
        substrate.width + 2*config.pad + 2*config.pml,   # y
        substrate.thickness+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        substrate.length,  # x
        substrate.width,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -config.cell_size[2]/2.0+config.pml+15/xm

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    # =====================================================
    save_and_show_config(config, [AuTop, substrate])
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_vis_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"antenna_roi_{plane}.png"
        save_2D_plot(
            sim,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(1, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_vis_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.vis_volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE,
            config=config
        )
    print_task(2, "2D projections.")
    for plane in ["XY", "XZ", "YZ"]:
        Name2D = f"empty_roi_{plane}.png"
        save_2D_plot(
            sim_empty,
            antenna_vols.volume[plane],
            save_name=Name2D,
            path_to_save=config.path_to_save,
            IMG_CLOSE=config.IMG_CLOSE
        )
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(
        sim,
        sim_empty,
        antenna_vols,
        config,
        fluxes=True,
        scattering=True,
        dft_gap_spectrum=True,
        harminv=True,
        scattering_antenna=AuTop
    )
    return 0

def after_hpc_redraw(material_name):
    # =====================================================
    config = SimulationConfig()

    config.resolution = 500
    config.sim_time = 18000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 660 / xm
    config.frequency_width = 1.0
    gap = 6

    X_material = get_materials_dict(material_name)
    X_material_name = material_name
    
    SIM_NAME = f"BSE_Au{X_material_name}_wavleng_{config.lambda0}_gap_{gap}"
    config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
    # =====================================================
    AuTop = BowTieEquilateral(
        gap=gap/xm,
        length=86.6/xm, # <- to have about 100 nm in width
        thickness=30/xm,
        radius=5/xm,
        material=Au,
        z_offset=0.0
    )
    substrate = Bar(
        length=800/xm,
        width=800/xm,
        thickness=100/xm,
        material=X_material,
        z_offset=-(30/2.0+100/2.0)/xm,
        radius=12/xm,
    )

    geometry = AuTop.build_geometry() + substrate.build_geometry()

    config.pad = 80/xm
    config.pml = 350/xm
    config.cell_size = [
        substrate.length + 2*config.pad + 2*config.pml,   # x
        substrate.width + 2*config.pad + 2*config.pml,   # y
        substrate.thickness+AuTop.thickness + 2*config.pad + 2*config.pml    # z
    ]
    cell = make_cell(config=config)

    config.src_size = [
        substrate.length,  # x
        substrate.width,  # y
        0.0 / xm    # z
    ]
    config.src_center = [
        0.0,    # x
        0.0,    # y
        config.cell_size[2]/2.0-1.15*config.pml  # z
    ]

    config.nfreq = 500
    config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
    config.z_transmission = -(config.cell_size[2]/2.0-1.15*config.pml)

    antenna_vols = VolumeSetROI(cell, antenna=AuTop)

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=geometry,
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )
    sim_empty = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(config.pml)],
        geometry=[],
        sources=make_source(config),
        resolution = config.resolution,
        k_point = mp.Vector3(),
        symmetries=config.symmetries,
        dimensions=3
        )

    # =====================================================
    print_task(4, "Postprocesing - raw animations for X.")
    animate_raw_fields(config=config, mode="BOTH", component="X")
    # =====================================================
    print_task(4, "Postprocesing - raw animations for Y.")
    animate_raw_fields(config=config, mode="BOTH", component="Y")
    # =====================================================
    print_task(4, "Postprocesing - raw animations for Z.")
    animate_raw_fields(config=config, mode="BOTH", component="Z")
    # =====================================================
    print_task(3, "3D calculations.")
    compute_fields(
        sim,
        sim_empty,
        antenna_vols,
        config,
        fluxes=False,
        scattering=False,
        dft_gap_spectrum=False,
        harminv=False,
        scattering_antenna=AuTop,
        mode="ENH_ONLY"
    )
    # =====================================================
    draw_params = {
        "XY": {"x_zoom": 1,
                "y_zoom": 1,
                "roi": {
                    "center": (0, 0),
                    "width": AuTop.gap * 1.05 * 1e3,
                    "height": AuTop.radius * 2.1 * 1e3,
                },
        },
        "XZ": {"x_zoom": 1,
                "y_zoom": 1,
                "roi": {
                    "center": (0, 0),
                    "width": AuTop.gap * 1.05 * 1e3,
                    "height": AuTop.thickness * 1e3,
                },
        },
        "YZ": {"x_zoom": 0.25,
                "y_zoom": 1,
                "roi": {
                    "center": (0, 0),
                    "width": AuTop.radius * 2.1 * 1e3,
                    "height": AuTop.thickness * 1e3,
                },
        },
    }
    print_task(5, "Postprocesing - animations and plots.")
    animate_enhancement_fields(config=config, volumes=antenna_vols, draw_params=draw_params, animate=True)
    # =====================================================
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar-empty_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=0,
        time_step=config.sim_time_step,
        save_name=f"source_prof_empty"
    )
    plot_signal_amplitude_vs_time_from_h5(
        "xyplanar_ex.h5",
        load_h5data_path=config.path_to_save,
        xzeros=0,
        time_step=config.sim_time_step,
        save_name=f"source_prof_antenna"
    ) 
    return 0

def split_bar_AuTiX():
    # =====================================================
    config = SimulationConfig()

    config.resolution = 400
    config.sim_time = 25000 / xm
    config.sim_time_step = 50 / xm
    config.lambda0 = 1200 / xm
    config.frequency_width = 0.6

    gap = 30

    X_materials = [Cu, Be, Cr, Pt, W] #SiO2, mp.air, Au,
    X_material_names = ["Cu", "Be", "Cr", "Pt", "W"] #"SiO2","Air", "Au",
    for X_material, X_material_name in zip(X_materials, X_material_names):
        SIM_NAME = f"split_bar_antenna_AuTi{X_material_name}_res{config.resolution}"
        config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
        # =====================================================
        AuTop = SplitBar(
            gap=gap/xm,
            length=300/xm,
            width=50/xm,
            thickness=30/xm,
            material=Au,
            z_offset=0.0/xm,
            radius=0/xm,
        )
        TiBetween = SplitBar(
            gap=gap/xm,
            length=300/xm,
            width=50/xm,
            thickness=5/xm,
            material=Ti,
            z_offset=-(30+5)/2.0/xm,
            radius=0/xm,
        )
        substrate = Bar(
            length=1000/xm,
            width=200/xm,
            thickness=70/xm,
            material=X_material,
            z_offset=-(30/2.0+5+70/2.0)/xm,
            radius=12/xm,
        )

        geometry = AuTop.build_geometry() + TiBetween.build_geometry() + substrate.build_geometry()

        config.pad = 100/xm
        config.pml = 350/xm
        config.cell_size = [
            substrate.length + 2*config.pad + 2*config.pml,   # x
            substrate.width + 2*config.pad + 2*config.pml,   # y
            substrate.thickness+AuTop.thickness+TiBetween.thickness + 2*config.pad + 2*config.pml    # z
        ]
        cell = make_cell(config=config)

        config.src_size = [
            substrate.length,  # x
            substrate.width,  # y
            0.0 / xm    # z
        ]
        config.src_center = [
            0.0,    # x
            0.0,    # y
            config.cell_size[2]/2.0-1.15*config.pml  # z
        ]

        config.nfreq = 500
        config.z_reflection = config.cell_size[2]/2.0-1.20*config.pml
        config.z_transmission = -(config.cell_size[2]/2.0-1.15*config.pml)


        antenna_vols = VolumeSet(cell, antenna=AuTop, top_z=AuTop.thickness, extra_vols_in_gap=False)

        # save_and_show_config(config, [AuTop, TiBetween, substrate])

        sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=geometry,
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        sim_empty = mp.Simulation(
            cell_size=cell,
            boundary_layers=[mp.PML(config.pml)],
            geometry=[],
            sources=make_source(config),
            resolution = config.resolution,
            k_point = mp.Vector3(),
            symmetries=config.symmetries,
            dimensions=3
            )
        # # =====================================================
        # print_task(1, "2D projections.")
        # for plane in ["XY", "XZ", "YZ"]:
        #     Name2D = f"antenna_{plane}.png"
        #     save_2D_plot(
        #         sim,
        #         antenna_vols.vis_volume[plane],
        #         save_name=Name2D,
        #         path_to_save=config.path_to_save,
        #         IMG_CLOSE=config.IMG_CLOSE
        #     )
        # =====================================================
        print_task(3, "3D calculations.")
        compute_fields(sim, sim_empty, antenna_vols, config)
        # =====================================================
        print_task(4, "Postprocesing - raw animations.")
        animate_raw_fields(config=config, mode="BOTH")
        # =====================================================
        draw_params = {
            "XY": {"x_zoom": 0.10,
                    "y_zoom": 0.3,
                    "roi": {
                        "center": (0, 0),
                        "width": AuTop.gap * 1e3,
                        "height": AuTop.width * 1e3,
                    },
            },
            "XZ": {"x_zoom": 0.1,
                    "y_zoom": 0.2,
                    "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": AuTop.gap * 1e3,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
            "YZ": {"x_zoom": 0.4,
                    "y_zoom": 0.2,
                    "roi": {
                        "center": (0, -1e3*TiBetween.thickness/2.0),
                        "width": AuTop.width * 1e3,
                        "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                    },
            },
        }
        print_task(5, "Postprocesing - animations and plots.")
        animate_enhancement_fields(config=config, draw_params=draw_params)
        # =====================================================
        plot_signal_amplitude_vs_time_from_h5(
            "xyplanar-empty_ex.h5",
            load_h5data_path=config.path_to_save,
            xzeros=int(100),
            time_step=config.sim_time_step,
            save_name=f"source_prof_empty"
        )
        plot_signal_amplitude_vs_time_from_h5(
            "xyplanar_ex.h5",
            load_h5data_path=config.path_to_save,
            xzeros=int(100),
            time_step=config.sim_time_step,
            save_name=f"source_prof_antenna"
        ) 
    return 0

def split_bar_AuTiSiO2():
    # =====================================================
    config = SimulationConfig()
    config.resolution = 500
    config.sim_time = 8000 / xm
    config.sim_time_step = 100 / xm
    config.lambda0 = 8100 / xm
    config.frequency_width = 1

    for gap in [10]:
        for T in [20, 25, 30, 35, 40]: #, 40
            SIM_NAME = f"T_{T}_split_bar_antenna_gap_{gap}nm_AuTiSiO2_test"
            config.path_to_save, config.animations_folder_path = create_directory(SIM_NAME)
            # =====================================================
            AuTop = SplitBar(
                gap=gap/xm,
                length=1800/xm,
                width=240/xm,
                thickness=T/xm,
                material=Au,
                z_offset=0.0/xm,
                radius=0/xm,
            )
            TiBetween = SplitBar(
                gap=gap/xm,
                length=1800/xm,
                width=240/xm,
                thickness=5/xm,
                material=Ti,
                # material=Pd,
                z_offset=-(T+5)/2.0/xm,
                radius=0/xm,
            )
    # 
    #         # !! PADS !! #########
    #         AuTopPAD = SplitBar(
    #             gap=(gap+1700*2)/xm,
    #             length=100/xm,
    #             width=40/xm,
    #             thickness=30/xm,
    #             material=Au,
    #             z_offset=0.0/xm,
    #             radius=0/xm,
    #             center=(0.0, (20+240/2.0)/xm)
    #         )
    #         TiBetweenPAD = SplitBar(
    #             gap=(gap+1700*2)/xm,
    #             length=100/xm,
    #             width=40/xm,
    #             thickness=5/xm,
    #             material=Ti,
    #             # material=Pd,
    #             z_offset=-(30+5)/2.0/xm,
    #             radius=0/xm,
    #             center=(0.0, (20+240/2.0)/xm)
    #         )
            #########################
            
            substrate = Bar(
                length=4000/xm,
                width=320/xm,
                thickness=70/xm,
                material=SiO2,
                z_offset=-(T/2.0+5+70/2.0)/xm,
                radius=12/xm,
            )

            # geometry = AuTop.build_geometry() + TiBetween.build_geometry() + AuTopPAD.build_geometry() + TiBetweenPAD.build_geometry() + substrate.build_geometry()
            geometry = AuTop.build_geometry() + TiBetween.build_geometry() + substrate.build_geometry()

            config.pad = 100/xm
            config.pml = 100/xm
            config.cell_size = [
                substrate.length + 2*config.pad + 2*config.pml,   # x
                substrate.width + 2*config.pad + 2*config.pml,   # y
                substrate.thickness+AuTop.thickness+TiBetween.thickness + 2*config.pad + 2*config.pml    # z
            ]
            cell = make_cell(config=config)

            config.src_size = [
                substrate.length,  # x
                substrate.width,  # y
                0.0 / xm    # z
            ]
            config.src_center = [
                0.0,    # x
                0.0,    # y
                config.cell_size[2]/2.0-1.15*config.pml  # z
            ]

            antenna_vols = VolumeSet(cell, antenna=AuTop, top_z=AuTop.thickness)


            sim = mp.Simulation(
                cell_size=cell,
                boundary_layers=[mp.PML(config.pml)],
                geometry=geometry,
                sources=make_source(config),
                resolution = config.resolution,
                k_point = mp.Vector3(),
                symmetries=config.symmetries,
                dimensions=3
                )
            sim_empty = mp.Simulation(
                cell_size=cell,
                boundary_layers=[mp.PML(config.pml)],
                geometry=[],
                sources=make_source(config),
                resolution = config.resolution,
                k_point = mp.Vector3(),
                symmetries=config.symmetries,
                dimensions=3
                )
                
            # # =====================================================                
            # save_and_show_config(config, [AuTop, TiBetween, substrate])
            # # =====================================================
            # print_task(1, "2D projections.")
            # for plane in ["XY", "XZ", "YZ"]:
            #     Name2D = f"antenna_{plane}.png"
            #     save_2D_plot(
            #         sim,
            #         antenna_vols.vis_volume[plane],
            #         save_name=Name2D,
            #         path_to_save=config.path_to_save,
            #         IMG_CLOSE=config.IMG_CLOSE
            #     )
            # # =====================================================
            # print_task(3, "3D calculations.")
            # compute_fields(
            #     sim,
            #     sim_empty,
            #     antenna_vols,
            #     config,
            #     fluxes=False,
            #     scattering=False,
            # )
            # # =====================================================
            # print_task(4, "Postprocesing - raw animations.")
            # animate_raw_fields(config=config, mode="BOTH")
            # # =====================================================
            draw_params = {
                "XY": {"x_zoom": 0.045,
                       "y_zoom": 0.6,
                       "roi": {
                            "center": (0, 0),
                            "width": AuTop.gap * 1e3,
                            "height": AuTop.width * 1e3,
                        },
                },
                "XZ": {"x_zoom": 0.045,
                       "y_zoom": 0.2,
                       "roi": {
                            "center": (0, -1e3*TiBetween.thickness/2.0),
                            "width": AuTop.gap * 1e3,
                            "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                        },
                },
                "YZ": {"x_zoom": 0.4,
                       "y_zoom": 0.2,
                       "roi": {
                            "center": (0, -1e3*TiBetween.thickness/2.0),
                            "width": AuTop.width * 1e3,
                            "height": (AuTop.thickness + TiBetween.thickness) * 1e3,
                        },
                },
            }
            print_task(5, "Postprocesing - animations and plots.")
            animate_enhancement_fields(config=config, volumes=antenna_vols, draw_params=draw_params, animate=False)
            # # =====================================================
            # plot_signal_amplitude_vs_time_from_h5(
            #     "xyplanar-empty_ex.h5",
            #     load_h5data_path=config.path_to_save,
            #     xzeros=int(100),
            #     time_step=config.sim_time_step,
            #     save_name=f"source_prof_empty"
            # )
            # plot_signal_amplitude_vs_time_from_h5(
            #     "xyplanar_ex.h5",
            #     load_h5data_path=config.path_to_save,
            #     xzeros=int(100),
            #     time_step=config.sim_time_step,
            #     save_name=f"source_prof_antenna"
            # )
                
    return 0
