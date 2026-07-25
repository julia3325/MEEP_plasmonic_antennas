import sys, os
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

def _get_mpi_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD, MPI
    except ImportError:
        return None, None

def _pick_resonance_idx(spectrum, ref=None, edge_guard=2, snr_frac=0.1):
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

    Returns
    -------
    (idx, at_boundary) : (int, bool)
    """
    n = len(spectrum)
    g = min(edge_guard, n // 2)
    lo, hi = g, n - g  # search range [lo, hi)

    search = np.full(n, -np.inf)
    search[lo:hi] = np.asarray(spectrum[lo:hi], dtype=float)

    if ref is not None:
        ref = np.asarray(ref, dtype=float)
        thr = snr_frac * np.max(ref[lo:hi])
        low_snr = ref < thr
        # only apply the mask if it leaves something to choose from
        if np.any(np.isfinite(search) & ~low_snr):
            search[low_snr] = -np.inf

    idx = int(np.argmax(search))
    # flag if the chosen point is at (or right next to) either window edge
    at_boundary = (idx <= lo + 1) or (idx >= hi - 2)
    return idx, at_boundary

def _dft_gap_spectra(sim, dft_box, dft_inner, dft_center, nfreq, comm, MPI):
    """
    Extract three Ex spectra from the gap DFT regions:

        max_amp[i]    - max |Ex| over the full gap box (includes pixels at
                        the metal walls -> grid-singular, resolution-dependent)
        mean_int[i]   - mean |Ex|^2 over the inner box (metal-adjacent pixels
                        excluded -> converges with resolution)
        center_amp[i] - |Ex| at the gap centre point

    All reductions are MPI-safe whether get_dft_array returns full or
    chunk-local arrays: max/center use MAX reduction, mean uses SUM of
    (sum, count) so rank multiplicity cancels.
    """
    max_amp = np.zeros(nfreq)
    mean_int = np.zeros(nfreq)
    center_amp = np.zeros(nfreq)

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

def hybridbar_calculate_resonant_peaks():
    config = SimulationConfig()

    lambda_min_nm = 5700.0
    lambda_max_nm = 10300.0

    fmin = 1.0 / (lambda_max_nm / xm)
    fmax = 1.0 / (lambda_min_nm / xm)

    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    center_wavelength_nm = (1.0 / fcen) * xm  # około 7338.7 nm
    nfreq = 200

    config.resolution = 350
    config.lambda0 = center_wavelength_nm / xm
    # fwidth MUST be broad enough that the empty (reference) field has good
    # SNR across the WHOLE scanned band, otherwise ant/empty blows up at the
    # band edges (division by a near-zero reference) and the peak pins to the
    # bluest point for every geometry. 6*df reproduces the old working source
    # (old code used df*lambda0 ~ 7*df). The envelope cancels in the ratio.
    config.frequency_width = 6.0 * df

    sweeps = [
        # {"name": "HybridBar_1", "gap": 30, "L_bar": 1600, "L_tip": 150, "W": 240},
        # {"name": "HybridBar_2", "gap": 30, "L_bar": 1800, "L_tip": 150, "W": 240},
        # {"name": "HybridBar_3", "gap": 30, "L_bar": 1400, "L_tip": 150, "W": 240},
        {"name": "HybridBar_3", "gap": 30, "L_bar": 1400, "L_tip": 150, "W": 280},
        {"name": "HybridBar_3", "gap": 30, "L_bar": 1600, "L_tip": 150, "W": 280},
        {"name": "HybridBar_3", "gap": 30, "L_bar": 1600, "L_tip": 200, "W": 240},
        {"name": "HybridBar_3", "gap": 30, "L_bar": 1600, "L_tip": 100, "W": 240},
        
    ]

    results_filename = "results/resonant_peaks_summary.txt"
    if not os.path.exists("results") and mp.am_master():
        os.makedirs("results")
        
    if mp.am_master():
        with open(results_filename, "w") as f:
            f.write("Geometria\tGap[nm]\tL_bar[nm]\tL_tip[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

    freqs = np.linspace(fcen - df/2.0, fcen + df/2.0, nfreq)

    for p in sweeps:
        mp.print_messages = False 
        print_task(1, f"Szukanie rezonansu dla: {p['name']}")

        L_bar = p["L_bar"] / xm
        L_tip = p["L_tip"] / xm
        width = p["W"] / xm
        gap = p["gap"] / xm
        
        Th_Au = 30 / xm
        Th_Ti = 5 / xm
        Th_Sub = 100 / xm
        L_Sub = (p["gap"] + 2 * p["L_bar"] + p["L_tip"] + 400)/xm
        W_Sub = (p["W"] + 400)/xm
        radius = 5 / xm

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

        # full gap box (touches the metal walls at x = +-gap/2)
        dft_size = mp.Vector3(gap, 10/xm, 10/xm)
        # inner box: exclude ~2 pixels next to each metal wall, where the
        # discretized corner fields are singular and do not converge
        inner_gap = max(gap - 4.0/config.resolution, 0.4*gap)
        dft_size_inner = mp.Vector3(inner_gap, 10/xm, 10/xm)

        sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
        dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
            fef_center = ant_center**2 / (empty_center**2 + eps) # gap centre point

            # peak POSITION from the gap-centre spectrum: it tracks the
            # dipolar gap resonance cleanly, unlike the gap-averaged one
            # whose rising short-wavelength background pushes argmax to the
            # window edge. Interior local-max search avoids boundary pinning.
            best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=empty_center)
            best_freq = freqs[best_idx]
            best_wavelength_nm = (1.0 / best_freq) * xm
            warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
            print(f"--> ZNALEZIONO REZONANS: {best_wavelength_nm:.2f} nm "
                  f"(FEF_mean = {fef_mean[best_idx]:.2f}, "
                  f"FEF_center = {fef_center[best_idx]:.2f}, "
                  f"FEF_max_px = {fef_max[best_idx]:.2f}){warn}")

            import matplotlib.pyplot as plt

            wavelengths_nm = (1.0 / freqs) * xm

            plt.figure(figsize=(8, 5))
            plt.semilogy(wavelengths_nm, fef_mean, '-', color='darkred', linewidth=2, label='FEF mean (gap)')
            plt.semilogy(wavelengths_nm, fef_center, '-', color='darkblue', linewidth=1.5, label='FEF centre (peak pick)')
            plt.semilogy(wavelengths_nm, fef_max, '--', color='gray', linewidth=1.5, label='FEF max pixel')
            plt.plot(best_wavelength_nm, fef_center[best_idx], 'o', color='gold', markersize=8, markeredgecolor='black', label=f'Peak: {best_wavelength_nm:.1f} nm')

            plt.xlabel('Wavelength [nm]', fontsize=14)
            plt.ylabel('Field Enhancement Factor (FEF)', fontsize=14)
            plt.title(f'Resonance Spectrum: L_bar {p["L_bar"]} nm, L_tip {p["L_tip"]} nm, width {p["W"]} nm', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()

            plot_filename = os.path.join("results", f"spectrum_gap_{p['gap']}nm_Lbar_{p['L_bar']}nm_Ltip_{p['L_tip']}nm_W_{p['W']}nm.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"Zapisano wykres widma: {plot_filename}")

            with open(results_filename, "a") as f:
                f.write(f"{p['name']}\t{p['gap']}\t{p['L_bar']}\t{p['L_tip']}\t{p['W']}\t{best_wavelength_nm:.2f}\t{fef_mean[best_idx]:.2f}\t{fef_center[best_idx]:.2f}\t{fef_max[best_idx]:.2f}\n")

    if mp.am_master():
        print_task(5, f" Wyniki zapisano w {results_filename}")
    return 0

def splitbar_calculate_resonant_peaks():
    """
    Resonant-wavelength scan for the SPLIT-BAR antenna (two plain rectangular
    bars, NO triangular tips), repeated for TWO substrates: SiO2 and Si.

    Same method as hybridbar_calculate_resonant_peaks: one broadband pulse +
    converged DFT (stop_when_dft_decayed), FEF = |E_ant|^2/|E_empty|^2 in the
    gap, peak position taken from the gap-centre spectrum. For every geometry
    the scan runs once per substrate, so you can read off the substrate shift
    directly (Si has a much higher index than SiO2 -> strong RED-shift).

    sweeps dict keys: name, gap, L, W
        L = length of ONE bar arm; the full dipole span is 2*L + gap.
        (No L_tip - a split bar has no tips.)
    """
    config = SimulationConfig()

    lambda_min_nm = 5700.0
    lambda_max_nm = 10300.0

    fmin = 1.0 / (lambda_max_nm / xm)
    fmax = 1.0 / (lambda_min_nm / xm)

    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    center_wavelength_nm = (1.0 / fcen) * xm
    nfreq = 200

    config.resolution = 350   # lower to ~200-250 for screening
    config.lambda0 = center_wavelength_nm / xm
    config.frequency_width = 6.0 * df   # broad source; cancels in the ratio

    # Substrates to scan: (label, meep material). Si = crystalline silicon
    # (cSi). NOTE: the paper's p++ Si is doped (extra free-carrier loss) and
    # sits under a 300 nm SiO2 spacer - to reproduce it exactly, stack Si with
    # a thin SiO2 layer here. Also, the antenna near-field penetrates deeper
    # than the 100 nm Th_Sub below; for a full high-index-substrate effect
    # increase Th_Sub (a thin Si layer underestimates the red-shift).
    substrates = [
        ("SiO2", SiO2),
        ("Si",   cSi),
    ]

    sweeps = [
        {"name": "SplitBar", "gap": 20, "L": 1800, "W": 240},
        # {"name": "SplitBar", "gap": 20, "L": 890,  "W": 240},  # ~1.8 um TOTAL dipole (paper L1)
    ]

    results_filename = "results/resonant_peaks_splitbar_summary.txt"
    if not os.path.exists("results") and mp.am_master():
        os.makedirs("results")

    if mp.am_master():
        with open(results_filename, "w") as f:
            f.write("Geometria\tSubstrate\tGap[nm]\tL[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

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
            radius = 5 / xm

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

            dft_size = mp.Vector3(gap, 10/xm, 10/xm)
            inner_gap = max(gap - 4.0/config.resolution, 0.4*gap)
            dft_size_inner = mp.Vector3(inner_gap, 10/xm, 10/xm)

            sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

            dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
            dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
            dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
            dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
                fef_center = ant_center**2 / (empty_center**2 + eps)

                best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=empty_center)
                best_freq = freqs[best_idx]
                best_wavelength_nm = (1.0 / best_freq) * xm
                warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
                print(f"--> [{sub_name}] REZONANS: {best_wavelength_nm:.2f} nm "
                      f"(FEF_mean = {fef_mean[best_idx]:.2f}, "
                      f"FEF_center = {fef_center[best_idx]:.2f}, "
                      f"FEF_max_px = {fef_max[best_idx]:.2f}){warn}")

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
                    f.write(f"{p['name']}\t{sub_name}\t{p['gap']}\t{p['L']}\t{p['W']}\t{best_wavelength_nm:.2f}\t{fef_mean[best_idx]:.2f}\t{fef_center[best_idx]:.2f}\t{fef_max[best_idx]:.2f}\n")

    if mp.am_master():
        print_task(5, f" Wyniki zapisano w {results_filename}")
    return 0

def hybridbar_AuTiSiO2_opt():

    config = SimulationConfig()
    config.IMG_CLOSE = True

    tasks = [
        {"gap": 30, "L_bar": 1400, "L_tip": 150,  "W": 280, "Wavelength": 6928.36},
        {"gap": 30, "L_bar": 1600, "L_tip": 150,  "W": 280, "Wavelength": 9784.18},
        {"gap": 30, "L_bar": 1600, "L_tip": 200,  "W": 240, "Wavelength": 10300},
        {"gap": 30, "L_bar": 1800, "L_tip": 150,  "W": 240, "Wavelength": 10300},
        {"gap": 30, "L_bar": 1600, "L_tip": 100,  "W": 240, "Wavelength": 9421.25},
        #{"gap": 30, "L_bar": 1600, "L_tip": 200,  "W": 240, "Wavelength": 10300},
        #{"gap": 30, "L_bar": 1200, "L_tip": 150,  "W": 240, "Wavelength": 6340}, 
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
    nfreq = 200

    config.resolution = 350
    config.lambda0 = center_wavelength_nm / xm
    # fwidth MUST be broad enough that the empty (reference) field has good
    # SNR across the WHOLE scanned band, otherwise ant/empty blows up at the
    # band edges (division by a near-zero reference) and the peak pins to the
    # bluest point for every geometry. 6*df reproduces the old working source
    # (old code used df*lambda0 ~ 7*df). The envelope cancels in the ratio.
    config.frequency_width = 6.0 * df

    sweeps = [
        # {"name": "BowTie_L300", "gap": 30, "L": 300, "W": 300},
        # {"name": "BowTie_L400", "gap": 30, "L": 400, "W": 300},
        # {"name": "BowTie_L500", "gap": 30, "L": 500, "W": 300},
        # {"name": "BowTie_L300", "gap": 30, "L": 150, "W": 300},
        # {"name": "BowTie_W400", "gap": 30, "L": 500, "W": 400},
        # {"name": "BowTie_W500", "gap": 30, "L": 500, "W": 500},
        # {"name": "BowTie_L1000_W800", "gap": 30, "L": 1000, "W": 800},
        # {"name": "BowTie_L1000_W1000", "gap": 30, "L": 1000, "W": 1000},
        # {"name": "BowTie_L900_W800", "gap": 30, "L": 900, "W": 800},
        {"name": "BowTie_L1100_W800", "gap": 30, "L": 1100, "W": 800},
        {"name": "BowTie_L1100_W900", "gap": 30, "L": 1100, "W": 900},
        {"name": "BowTie_L1100_W900", "gap": 30, "L": 1100, "W": 1000},
    ]

    results_filename = "results/resonant_peaks_summary.txt"
    if not os.path.exists("results") and mp.am_master():
        os.makedirs("results")
        
    if mp.am_master():
        with open(results_filename, "w") as f:
            f.write("Geometria\tGap[nm]\tL[nm]\tW[nm]\tResonant_Wavelength[nm]\tFEF_mean_gap\tFEF_center\tFEF_max_px\n")

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
        # BUGFIX: previously referenced p["L_bar"] etc. (leftover from the
        # hybridbar loop variable); bowtie total length = 2*L + gap
        L_Sub = (params["gap"] + 2 * params["L"] + 400)/xm
        W_Sub = (params["W"] + 400)/xm
        radius = 5 / xm

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

        # full gap box (touches the metal tips at x = +-gap/2)
        dft_size = mp.Vector3(gap, 10/xm, 10/xm)
        # inner box: exclude ~2 pixels next to each metal tip, where the
        # discretized corner fields are singular and do not converge
        inner_gap = max(gap - 4.0/config.resolution, 0.4*gap)
        dft_size_inner = mp.Vector3(inner_gap, 10/xm, 10/xm)

        sim_empty = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(config.pml)], geometry=[], sources=make_source(config), resolution=config.resolution, symmetries=config.symmetries, dimensions=3)

        dft_empty = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size)
        dft_empty_in = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=dft_size_inner)
        dft_empty_c = sim_empty.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
        dft_ant_c = sim.add_dft_fields([mp.Ex], fcen, df, nfreq, center=mp.Vector3(0, 0, 0), size=mp.Vector3())

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
            fef_center = ant_center**2 / (empty_center**2 + eps) # gap centre point

            # peak POSITION from the gap-centre spectrum: it tracks the
            # dipolar gap resonance cleanly, unlike the gap-averaged one
            # whose rising short-wavelength background pushes argmax to the
            # window edge. Interior local-max search avoids boundary pinning.
            best_idx, at_boundary = _pick_resonance_idx(fef_center, ref=empty_center)
            best_freq = freqs[best_idx]
            best_wavelength_nm = (1.0 / best_freq) * xm
            warn = "  [!] brak piku w oknie - rezonans prawdopodobnie POZA zakresem" if at_boundary else ""
            print(f"--> ZNALEZIONO REZONANS: {best_wavelength_nm:.2f} nm "
                  f"(FEF_mean = {fef_mean[best_idx]:.2f}, "
                  f"FEF_center = {fef_center[best_idx]:.2f}, "
                  f"FEF_max_px = {fef_max[best_idx]:.2f}){warn}")

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

            plot_filename = os.path.join("results", f"spectrum_gap_{params['gap']}nm_L_{params['L']}nm_W_{params['W']}nm.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"Zapisano wykres widma: {plot_filename}")

            with open(results_filename, "a") as f:
                f.write(f"{params['name']}\t{params['gap']}\t{params['L']}\t{params['W']}\t{best_wavelength_nm:.2f}\t{fef_mean[best_idx]:.2f}\t{fef_center[best_idx]:.2f}\t{fef_max[best_idx]:.2f}\n")

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
    LOKALNY postprocessing wynikow z compute_dft_enhancement_maps (na Aresie
    zapisywane sa tylko male pliki dft_enhancement_*.h5 + field_vs_time_gap.dat,
    bez PNG). Ta funkcja renderuje mapy PNG i zapisuje srednie/maksymalne
    wzmocnienie w przerwie do pliku txt.
    """
    simulations = [
        # {"folder": "HybridBar_gap_30nm_Lbar_1600nm_Ltip_150_W_240nm_AuTiSiO2_res350_lambda_9.57313", "gap": 30},
    ]

    Th_Au = 30.0  # nm
    Th_Ti = 5.0   # nm

    results_filename = "results/DFT_EFE_summary.txt"

    if mp.am_master():
        with open(results_filename, "w") as f:
            f.write("Folder\tGap[nm]\t"
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
        with open(results_filename, "w") as f:
            f.write("Folder\tGap[nm]\tL_bar[nm]\tL_tip[nm]\tW[nm]\tEFE_XY\tEFE_XZ\tEFE_YZ\n")

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
        
        # WZÓR MUSI BYĆ IDENTYCZNY JAK W SYMULACJI GŁÓWNEJ
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
        
        # PARAMETRY SIATKI MUSZĄ BYĆ IDENTYCZNE (było 100, zmienione na poprawne z aresa)
        config.pad = 200 / xm
        config.pml = 500 / xm
        
        config.cell_size = [
            L_Sub + 2*config.pad + 2*config.pml,
            W_Sub + 2*config.pad + 2*config.pml,
            Th_Sub + AuTop.thickness + TiBetween.thickness + 2*config.pad + 2*config.pml
        ]
        
        cell = make_cell(config=config)
        
        # Zmieniono VolumeSet na VolumeSetROI, by pasowało do klasy
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
        with open(results_filename, "w") as f:
            f.write("Folder\tGap[nm]\tL[nm]\tW[nm]\tEFE_XY\tEFE_XZ\tEFE_YZ\n")

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
