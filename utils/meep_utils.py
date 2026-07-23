import meep as mp
from meep.materials import *
import numpy as np
import os
from utils.logger import append_time_to_file

from visualization.plotter import *
# !!!!!!!!! ---> from main.src.simulation import * # CANT IMPORT DUE TO CIRCULAR DEPENDENCY

def get_materials_dict(material_name=None):
    materials = {
        # dielectrics / semiconductors
        "air": mp.air,
        "cSi": cSi,
        "aSi": aSi,
        "SiO2": SiO2,
        "ITO": ITO,
        "Al2O3": Al2O3,
        "GaAs": GaAs,
        "AlAs": AlAs,
        "AlN": AlN,
        "BK7": BK7,
        "FQ": fused_quartz,
        "Si3N4": Si3N4,
        "Ge": Ge,
        "InP": InP,
        "GaN": GaN,
        "CdTe": CdTe,
        "LiNbO3": LiNbO3,
        "BaB2O4": BaB2O4,
        "CaWO4": CaWO4,
        "CaCO3": CaCO3,
        "Y2O3": Y2O3,
        "YAG": YAG,
        "PMMA": PMMA,
        # metals
        "Ag": Ag,
        "Au": Au,
        "Cu": Cu,
        "Al": Al,
        "Be": Be,
        "Cr": Cr,
        "Ni": Ni,
        "Pd": Pd,
        "Pt": Pt,
        "Ti": Ti,
        "W": W,
    }
    if material_name not in materials:
        raise ValueError(f"Unknown material: {material_name}")
    if material_name is None:
        return materials["air"]

    return materials[material_name]

def collect_fields_with_output(
    sim,
    volumes,          # dict: {name: mp.Volume}
    delta_t,
    until,
    start_time=0.0,
    path=None,
    calc_E_fields=True,
    calc_H_fields=False,
    calc_Dpwr=False,
    extra_run_functions=None,
):
    """
    Collect selected field data for multiple volumes in ONE sim.run().

    Field components to be recorded are controlled by boolean flags.

    FIXED BEHAVIOR
    --------------
    * Data are written as HDF5 files using mp.to_appended(...)
    * All outputs follow the naming scheme:
          <prefix>_<field>.h5
      where <prefix> comes from the `volumes` dict.
    * Output directory is controlled at the Simulation level via
      sim.use_output_directory().

    PARAMETERS
    ----------
    sim : mp.Simulation
        Initialized Meep simulation object.

    volumes : dict[str, mp.Volume]
        Mapping of output prefixes to Meep volumes, e.g.:
            {
                "planar": planar_vol,
                "volume": volume_vol
            }

    delta_t : float
        Time interval between successive field outputs.

    until : float
        End time of the simulation.

    start_time : float, optional
        Time after which field recording starts.
        If 0.0, recording starts immediately.

    path : str or None, optional
        Directory where all HDF5 output files are written.
        If None, default Meep behavior is used.

    calc_E_fields : bool, optional
        If True, record electric field components:
            Ex, Ey, Ez

    calc_H_fields : bool, optional
        If True, record magnetic field components:
            Hx, Hy, Hz

    calc_Dpwr : bool, optional
        If True, record electric field energy density (Dpwr).

    extra_run_functions : callable or list of callables, optional
        Additional Meep step functions passed directly to sim.run().
    
        These can be used to extend the simulation with custom runtime
        behaviors such as:
    
            * mp.after_sources(mp.Harminv(...)) — modal (Harminv) analysis
            * mp.at_every(...) — custom field sampling
            * mp.at_end(...) — post-run callbacks
    
        The provided function(s) are appended to the internally constructed
        run actions and executed within the same sim.run() call.
    
        If None (default), no additional step functions are applied.

    NOTES
    -----
    * At least one of (calc_E_fields, calc_H_fields, calc_Dpwr)
      should be True, otherwise no data will be collected.
    * Default configuration records only E-field components.
    """
    # --------------------------------------------------
    # Disable default filename prefix (clean filenames)
    # --------------------------------------------------
    sim.filename_prefix = ""

    # --------------------------------------------------
    # Set output directory (Simulation-level, per docs)
    # --------------------------------------------------
    if path is not None:
        sim.use_output_directory(path)

    run_actions = []

    for prefix, volume in volumes.items():

        actions = []

        # -------------------------
        # Electric field components
        # -------------------------
        if calc_E_fields:
            actions.extend([
                mp.to_appended(f"{prefix}_ex", mp.at_every(delta_t, mp.output_efield_x)),
                mp.to_appended(f"{prefix}_ey", mp.at_every(delta_t, mp.output_efield_y)),
                mp.to_appended(f"{prefix}_ez", mp.at_every(delta_t, mp.output_efield_z)),
            ])

        # -------------------------
        # Magnetic field components
        # -------------------------
        if calc_H_fields:
            actions.extend([
                mp.to_appended(f"{prefix}_hx", mp.at_every(delta_t, mp.output_hfield_x)),
                mp.to_appended(f"{prefix}_hy", mp.at_every(delta_t, mp.output_hfield_y)),
                mp.to_appended(f"{prefix}_hz", mp.at_every(delta_t, mp.output_hfield_z)),
            ])

        # -------------------------
        # Energy density
        # -------------------------
        if calc_Dpwr:
            actions.append(
                mp.to_appended(f"{prefix}_dpwr", mp.at_every(delta_t, mp.output_dpwr))
            )

        # -------------------------
        # Skip volumes with no actions
        # -------------------------
        if not actions:
            continue

        # -------------------------
        # Volume action
        # -------------------------
        if start_time > 0:
            vol_action = mp.in_volume(
                volume,
                mp.after_time(start_time, *actions)
            )
        else:
            vol_action = mp.in_volume(volume, *actions)

        run_actions.append(vol_action)

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------
    if not run_actions:
        raise ValueError(
            "No field outputs selected. "
            "Set at least one of calc_E_fields, calc_H_fields, calc_Dpwr to True."
        )
        
    # --------------------------------------------------
    # Include extra funcs
    # --------------------------------------------------
    if extra_run_functions is not None:
        if isinstance(extra_run_functions, list):
            run_actions.extend(extra_run_functions)
        else:
            run_actions.append(extra_run_functions)

    # --------------------------------------------------
    # Single sim.run()
    # --------------------------------------------------
    sim.run(*run_actions, until=until)
    return sim

def enhancement_divided_by_maxes_arr(
    h5_target,
    h5_reference,
    path=None,
    save_to=None,
    dataset_target=None,
    dataset_reference=None,
    z_index=None,
    xzeros=0,
    yzeros=None,
    eps=1e-12,
    out_dataset_name="enhancement",
):
    """
    Compute time-dependent enhancement normalized by the time-maximum
    of a reference field.

    MATHEMATICAL DEFINITION
    -----------------------
    For each spatial point (x, y) and time t:

        enhancement[x, y, t] = A[x, y, t] / max_t(B[x, y, t])

    where:
        A[x,y,t] = sum_i (A_i[x,y,t]^2)
        B[x,y,t] = sum_i (B_i[x,y,t]^2)

    The summation index i runs over the provided field components
    (e.g. Ex, Ey, Ez). Any subset of components is allowed, but
    h5_target and h5_reference MUST have identical structure.

    FIXED DATA ASSUMPTIONS
    ---------------------
    * All datasets are stored as:
          data[x, y, time]
    * No axis reordering or transposition is performed.
    * Any Z slicing is applied BEFORE computations.

    PARAMETERS
    ----------
    h5_target : str or list[str]
        HDF5 filename(s) of the target field components.
        Can be:
            - single string (one field component)
            - list/tuple of strings (multiple components)

    h5_reference : str or list[str]
        HDF5 filename(s) of the reference field components.
        MUST mirror the structure of h5_target exactly
        (same type and same list length).

    dataset_target : str or None, optional
        Dataset name inside the target HDF5 files.
        If None, the first dataset in each file is used.

    dataset_reference : str or None, optional
        Dataset name inside the reference HDF5 files.
        If None, the first dataset in each file is used.

    z_index : int or None, optional
        Index of the Z slice to extract if the dataset has shape:
            data[x, y, z, time]
        If such a Z axis exists and z_index is None, an error is raised.

    xzeros : int, optional
        Number of grid points to overwrite at left/right boundaries
        of the reference field (PML cleanup).

    yzeros : int or None, optional
        Number of grid points to overwrite at bottom/top boundaries.
        If None, defaults to xzeros.

    eps : float, optional
        Small regularization constant added to the denominator to
        avoid division by zero.

    path : str or None, optional
        Directory in which all input HDF5 files are searched AND
        where the output file is written.
        If None, filenames are interpreted as given.

    save_to : str or None, optional
        Name of the output HDF5 file (written inside `path`).
        If None, no file is saved.

    out_dataset_name : str, optional
        Dataset name under which the enhancement array is stored
        in the output HDF5 file.

    RETURNS
    -------
    enhancement : np.ndarray
        Array of shape (x, y, time) containing the enhancement field.

    B_max : np.ndarray
        Array of shape (x, y) containing max_t(B[x,y,t]).
    """
    
    # ---------------------------
    # Helpers
    # ---------------------------
    def _open_all(h5_input, dataset_name):
        if isinstance(h5_input, (list, tuple)):
            files = []
            dsets = []
            for h5f in h5_input:
                f = h5py.File(os.path.join(path, h5f) if path else h5f, "r")
                name = dataset_name or list(f.keys())[0]
                d = f[name]
                files.append(f)
                dsets.append(d)
            return files, dsets
        else:
            f = h5py.File(os.path.join(path, h5_input) if path else h5_input, "r")
            name = dataset_name or list(f.keys())[0]
            return [f], [f[name]]

    def _get_frame_sum_sq(dsets, t):
        acc = None
        for d in dsets:
            if d.ndim == 4:
                if z_index is None:
                    raise ValueError("Z axis detected but z_index not provided")
                frame = d[:, :, z_index, t]
            else:
                frame = d[:, :, t]

            if acc is None:
                acc = frame**2
            else:
                acc += frame**2
        return acc

    # ---------------------------
    # Open files
    # ---------------------------
    fA, dA = _open_all(h5_target, dataset_target)
    fB, dB = _open_all(h5_reference, dataset_reference)

    # ---------------------------
    # Shape
    # ---------------------------
    for da, db in zip(dA, dB):
        if da.shape != db.shape:
            raise RuntimeError(
                "Enhancement ERROR: target and reference datasets have different shapes.\n"
                f"Target: {da.shape}, Reference: {db.shape}"
            )

    sample = dA[0]
    if sample.ndim == 4:
        Nx, Ny, Nz, Nt = sample.shape
    else:
        Nx, Ny, Nt = sample.shape

    if yzeros is None:
        yzeros = xzeros

    xzeros = max(0, min(xzeros, Nx // 2))
    yzeros = max(0, min(yzeros, Ny // 2))

    # ---------------------------
    # PASS 1: compute B_max
    # ---------------------------
    B_max = np.zeros((Nx, Ny), dtype=float)

    for t in range(Nt):
        B_frame = _get_frame_sum_sq(dB, t)

        # PML cleanup (on-the-fly)
        if xzeros > 0 or yzeros > 0:
            B_frame[:xzeros, :] = 1.0
            B_frame[-xzeros:, :] = 1.0
            B_frame[:, :yzeros] = 1.0
            B_frame[:, -yzeros:] = 1.0

        np.maximum(B_max, B_frame, out=B_max)

    # ---------------------------
    # PASS 2: compute enhancement
    # ---------------------------
    enhancement = np.empty((Nx, Ny, Nt), dtype=float)

    denom = B_max + eps

    for t in range(Nt):
        A_frame = _get_frame_sum_sq(dA, t)
        enhancement[:, :, t] = A_frame / denom

    # ---------------------------
    # Save
    # ---------------------------
    if save_to is not None:
        save_path = os.path.join(path, save_to) if path else save_to
        with h5py.File(save_path, "w") as f:
            f.create_dataset(out_dataset_name, data=enhancement)
            f.create_dataset("reference_max", data=B_max)

    # ---------------------------
    # Cleanup
    # ---------------------------
    for f in fA + fB:
        f.close()
    gc.collect()
    
    return enhancement, B_max

def analyze_roi_from_h5_physical(
    h5_filename,
    roi,
    load_h5data_path=None,
    dataset_name=None,

    # --- physical axis definition ---
    x_phys_range=None,   # (xmin, xmax)
    y_phys_range=None,   # (ymin, ymax)

    # --- PML / border crop ---
    xzeros=0,
    yzeros=None,
):
    """
    Analyze mean field value inside a physical ROI over time
    using the FULL simulation domain (no zoom).

    Parameters
    ----------
    h5_filename : str
        Name of HDF5 file with data[x,y,time].

    roi : dict
        ROI definition:
        {
            "type": "rectangle",
            "center": (x, y),
            "width": w,
            "height": h,
        }

    Returns
    -------
    frame_mean : ndarray (Nt, 2)
        [[frame_index, mean_value], ...]

    frame_max : ndarray (2,)
        [frame_index_of_max, max_mean_value]
    """
    # ---------------------------
    # Sanity
    # ---------------------------
    if x_phys_range is None or y_phys_range is None:
        raise ValueError("Provide x_phys_range and y_phys_range")

    if roi["type"] != "rectangle":
        raise NotImplementedError("Only rectangular ROI supported")

    h5_path = (
        os.path.join(load_h5data_path, h5_filename)
        if load_h5data_path is not None
        else h5_filename
    )

    # ---------------------------
    # OPEN FILE (lazy!)
    # ---------------------------
    f = h5py.File(h5_path, "r")

    if dataset_name is None:
        dataset_name = list(f.keys())[0]

    dset = f[dataset_name]

    if dset.ndim != 3:
        raise ValueError(f"Expected data[x,y,time], got {dset.shape}")

    Nx0, Ny0, Nt = dset.shape

    if yzeros is None:
        yzeros = xzeros

    # ---------------------------
    # CROPPING
    # ---------------------------
    xzeros = min(xzeros, Nx0 // 2)
    yzeros = min(yzeros, Ny0 // 2)

    xs = slice(xzeros, Nx0 - xzeros)
    ys = slice(yzeros, Ny0 - yzeros)

    Nx = Nx0 - 2 * xzeros
    Ny = Ny0 - 2 * yzeros

    # ---------------------------
    # PHYSICAL AXES
    # ---------------------------
    x_min0, x_max0 = x_phys_range
    y_min0, y_max0 = y_phys_range

    x_phys = np.linspace(x_min0, x_max0, Nx)
    y_phys = np.linspace(y_min0, y_max0, Ny)

    # ---------------------------
    # ROI mask
    # ---------------------------
    roi_mask = roi_mask_from_rectangle(
        x_phys,
        y_phys,
        center=roi["center"],
        width=roi["width"],
        height=roi["height"],
    )

    if roi_mask.shape != (Nx, Ny):
        raise ValueError(
            f"ROI mask shape {roi_mask.shape} != {(Nx, Ny)}"
        )

    # ---------------------------
    # PRECOMPUTE indices
    # ---------------------------
    roi_idx = np.where(roi_mask)

    # ---------------------------
    # LOOP over time (STREAMING)
    # ---------------------------
    mean_vals = np.empty(Nt, dtype=float)

    for t in range(Nt):
        frame = dset[xs, ys, t]
        mean_vals[t] = frame[roi_idx].mean()

    # ---------------------------
    # RESULTS
    # ---------------------------
    frames = np.arange(Nt)
    frame_mean = np.column_stack((frames, mean_vals))

    t_max = int(np.argmax(mean_vals))
    frame_max = np.array([t_max, mean_vals[t_max]])

    # ---------------------------
    # CLEANUP
    # ---------------------------
    f.close()
    gc.collect()

    return frame_mean, frame_max

def compute_fields(
    sim_antenna,
    sim_empty,
    volumes,
    config,
    mode="BOTH",
    calc_E=True,
    calc_H=False,
    calc_DPWR=False,
    fluxes=True,
    fluxes_X_size=None,
    fluxes_Y_size=None,
    scattering=True,
    dft_gap_spectrum=False,
    scattering_antenna=None,
    harminv=False,
):
    """
    Run field simulations and compute enhancement maps.

    Parameters
    ----------
    mode : str
        "WITH_ANTENNA", "EMPTY", "BOTH" or "ENH_ONLY"

    calc_E : bool
        Whether to calculate E-field enhancement.

    calc_H : bool
        Whether to calculate H-field enhancement.

    calc_DPWR : bool
        Whether to calculate power density fields.
    """

    valid_modes = ["WITH_ANTENNA", "EMPTY", "BOTH", "ENH_ONLY"]

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    # ============================================================
    # Plane configuration
    # ============================================================

    planes = {
        "xyplanar": volumes.volume["XY"],
        "xyplanarTOP": volumes.volume["XY_TOP"],
        "xzplanar": volumes.volume["XZ"],
        "yzplanar": volumes.volume["YZ"],
    }

    if volumes.extra:
        planes.update({
            "xyplanar_1": volumes.volume["XY_1"],
            "xyplanar_2": volumes.volume["XY_2"],
            "xyplanar_3": volumes.volume["XY_3"],
            "xyplanar_4": volumes.volume["XY_4"],
            "xyplanar_5": volumes.volume["XY_5"],
            "xzplanar_1": volumes.volume["XZ_1"],
            "xzplanar_2": volumes.volume["XZ_2"],
            "xzplanar_3": volumes.volume["XZ_3"],
            "xzplanar_4": volumes.volume["XZ_4"],
            "xzplanar_5": volumes.volume["XZ_5"],
            "yzplanar_1": volumes.volume["YZ_1"],
            "yzplanar_2": volumes.volume["YZ_2"],
            "yzplanar_3": volumes.volume["YZ_3"],
            "yzplanar_4": volumes.volume["YZ_4"],
            "yzplanar_5": volumes.volume["YZ_5"],
        })

    fcen = config.frequency
    df = config.frequency_width
    nfreq = config.nfreq
    # ============================================================
    # FLUX MONITORS
    # ============================================================
    if fluxes:
        if fluxes_X_size is None:
            fluxes_X_size = config.src_size[0]
        if fluxes_Y_size is None:
            fluxes_Y_size = config.src_size[1]
    
        refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, config.z_reflection),
            size=mp.Vector3(
                fluxes_X_size,
                fluxes_Y_size,
                0)
        )

        tran_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, config.z_transmission),
            size=mp.Vector3(
                fluxes_X_size,
                fluxes_Y_size,
                0)
        )

        refl_empty = sim_empty.add_flux(fcen, df, nfreq, refl_fr)
        tran_empty = sim_empty.add_flux(fcen, df, nfreq, tran_fr)
        refl = sim_antenna.add_flux(fcen, df, nfreq, refl_fr)
        tran = sim_antenna.add_flux(fcen, df, nfreq, tran_fr)
    # ============================================================
    # SCATTERING MONITORS
    # ============================================================
    if scattering:
        if scattering_antenna is None:
            raise ValueError("scattering_antenna must be provided for scattering spectrum")
        # scattering box
        Lx, Ly, Lz = make_scattering_box(
            antenna=scattering_antenna,
            config=config, padding_perc=10,
            extra_padding_nm=(0, 0, 0))

        cx, cy = scattering_antenna.center
        cz = scattering_antenna.z_offset

        scatt_regions = [
            # --- X planes ---
            mp.FluxRegion(
                center=mp.Vector3(cx - Lx/2, cy, cz),
                size=mp.Vector3(0, Ly, Lz)
            ),
            mp.FluxRegion(
                center=mp.Vector3(cx + Lx/2, cy, cz),
                size=mp.Vector3(0, Ly, Lz)
            ),

            # --- Y planes ---
            mp.FluxRegion(
                center=mp.Vector3(cx, cy - Ly/2, cz),
                size=mp.Vector3(Lx, 0, Lz)
            ),
            mp.FluxRegion(
                center=mp.Vector3(cx, cy + Ly/2, cz),
                size=mp.Vector3(Lx, 0, Lz)
            ),

            # --- Z planes ---
            mp.FluxRegion(
                center=mp.Vector3(cx, cy, cz - Lz/2),
                size=mp.Vector3(Lx, Ly, 0)
            ),
            mp.FluxRegion(
                center=mp.Vector3(cx, cy, cz + Lz/2),
                size=mp.Vector3(Lx, Ly, 0)
            ),
        ]

        scatt_empty = [sim_empty.add_flux(fcen, df, nfreq, r) for r in scatt_regions]
        scatt = [sim_antenna.add_flux(fcen, df, nfreq, r) for r in scatt_regions]
    # ============================================================
    # GAP DFT MONITORS
    # ============================================================
    if dft_gap_spectrum:
        if dft_gap_spectrum and mode != "BOTH":
            raise ValueError("DFT gap spectrum requires mode='BOTH'")
        if scattering_antenna is None:
            raise ValueError("scattering_antenna must be provided for DFT gap spectrum")
            
        cx, cy = scattering_antenna.center
        cz = scattering_antenna.z_offset
        t = scattering_antenna.thickness

        dz = 1 / config.resolution

        z_min = cz - t / 2
        z_max = cz + t / 2

        Nz = int(np.round(t / dz)) + 1
        z_points = z_min + np.arange(Nz) * dz

        gap_dft_empty = []
        gap_dft_antenna = []

        for z in z_points:
            pt = mp.Vector3(cx, cy, z)
            # EMPTY
            gap_dft_empty.append(
                sim_empty.add_dft_fields(
                    [mp.Ex, mp.Ey, mp.Ez],
                    fcen,
                    df,
                    nfreq,
                    where=mp.Volume(center=pt, size=mp.Vector3(0, 0, 0))
                )
            )
            # ANTENNA
            gap_dft_antenna.append(
                sim_antenna.add_dft_fields(
                    [mp.Ex, mp.Ey, mp.Ez],
                    fcen,
                    df,
                    nfreq,
                    where=mp.Volume(center=pt, size=mp.Vector3(0, 0, 0))
                )
            )

    # ============================================================
    # HARMONIC INVERSION IN GAP
    # ============================================================
    if harminv:
        if scattering_antenna is None:
            raise ValueError("scattering_antenna must be provided for Harminv")

        harminv_t0 = 8 # for debug !! after -> to config
    
        cx, cy = scattering_antenna.center
        cz = scattering_antenna.z_offset
        t = scattering_antenna.thickness
    
        dz = 1 / config.resolution
    
        # --- in gap ---
        gap_points = [
            mp.Vector3(cx, cy, cz - t/2 + 2*dz),  # bottom
            mp.Vector3(cx, cy, cz),             # center
            mp.Vector3(cx, cy, cz + t/2 - 2*dz),  # top
        ]
    
        # --- corner of antena ---
        # To have the same distance from the corner to the point as from the main corner to the gap center,
        # we need to take the angle into account
        corner_deg = np.arctan(scattering_antenna.width / 2 / scattering_antenna.length)
        corner_cx = cx + scattering_antenna.gap/2 + scattering_antenna.length + (scattering_antenna.gap/2.0)*np.cos(corner_deg) 
        corner_cy = cy + scattering_antenna.width/2 + (scattering_antenna.gap/2.0)*np.sin(corner_deg)
        tip_points = [
            mp.Vector3(corner_cx,corner_cy, cz),
            mp.Vector3(corner_cx,corner_cy, cz + 2*dz),
            mp.Vector3(corner_cx,corner_cy, cz - 2*dz),
        ]
    
        # --- arm ---
        arm_cx = cx + scattering_antenna.length + scattering_antenna.gap
        arm_points = [
            mp.Vector3(arm_cx, cy, cz),
            mp.Vector3(arm_cx, cy, cz + 2*dz),
            mp.Vector3(arm_cx, cy, cz - 2*dz),
        ]
    
        # --- far above ---
        far_point = mp.Vector3(cx, cy, cz + 2*t)
    
        harminv_points = (
            gap_points +
            tip_points +
            arm_points +
            [far_point]
        )
    
        harminv_objects = []
    
        for pt in harminv_points:
            hi_fcen = fcen
            hi_df = df
            h = mp.Harminv(mp.Ex, pt, hi_fcen, hi_df, mxbands=100) # !! mp.Ex for debug -> after should be config.component
            harminv_objects.append((pt, h))
    # ============================================================
    # EMPTY STRUCTURE
    # ============================================================

    if mode in ["EMPTY", "BOTH"]:
        if mp.am_master():
            print("Running simulation WITHOUT antenna")
            append_time_to_file(config, prefix="Running simulation WITHOUT antenna: ")

        empty_planes = {f"{k}-empty": v for k, v in planes.items()}

        sim_empty = collect_fields_with_output(
            sim_empty,
            volumes=empty_planes,
            delta_t=config.sim_time_step,
            until=config.sim_time,
            start_time=0,
            path=config.path_to_save,
            calc_E_fields=calc_E,
            calc_H_fields=calc_H,
            calc_Dpwr=calc_DPWR,
        )
        if fluxes and mode == "BOTH":
            incident_flux = mp.get_fluxes(tran_empty)
            refl_data = sim_empty.get_flux_data(refl_empty)
            sim_antenna.load_minus_flux_data(refl, refl_data)
        if scattering and mode == "BOTH":
            scatt_data = [sim_empty.get_flux_data(f) for f in scatt_empty]
            scatt_flux_faces_empty = [np.asarray(mp.get_fluxes(f)) for f in scatt_empty]
            for f, d in zip(scatt, scatt_data):
                sim_antenna.load_minus_flux_data(f, d)
            incident_flux_top = np.asarray(mp.get_fluxes(scatt_empty[5]))
            intensity = incident_flux_top / (Lx * Ly)
        if dft_gap_spectrum and mode == "BOTH":
            gap_data_empty = {
                "Ex": [],
                "Ey": [],
                "Ez": [],
            }
        
            for dft_e in gap_dft_empty:
        
                Ex_e = np.array([
                    sim_empty.get_dft_array(dft_e, mp.Ex, i)
                    for i in range(nfreq)
                ])
                Ey_e = np.array([
                    sim_empty.get_dft_array(dft_e, mp.Ey, i)
                    for i in range(nfreq)
                ])
                Ez_e = np.array([
                    sim_empty.get_dft_array(dft_e, mp.Ez, i)
                    for i in range(nfreq)
                ])
        
                gap_data_empty["Ex"].append(np.abs(Ex_e)**2)
                gap_data_empty["Ey"].append(np.abs(Ey_e)**2)
                gap_data_empty["Ez"].append(np.abs(Ez_e)**2)
        
            # numpy
            for comp in gap_data_empty:
                gap_data_empty[comp] = np.array(gap_data_empty[comp])

        if mp.am_master():
            print("Done.")
        sim_empty.reset_meep()

    # ============================================================
    # WITH ANTENNA
    # ============================================================


    if mode in ["WITH_ANTENNA", "BOTH"]:
        if mp.am_master():
            print("Running simulation WITH antenna")
            append_time_to_file(config, prefix="Running simulation WITH antenna: ")

        if harminv:
            extra_run_functions = [
                mp.after_time(harminv_t0, h) for _, h in harminv_objects
            ]
        else:
            extra_run_functions = None
        
        sim_antenna = collect_fields_with_output(
            sim_antenna,
            volumes=planes,
            delta_t=config.sim_time_step,
            until=config.sim_time,
            start_time=0,
            path=config.path_to_save,
            calc_E_fields=calc_E,
            calc_H_fields=calc_H,
            calc_Dpwr=calc_DPWR,
            extra_run_functions=extra_run_functions,
        )
        if fluxes and mode == "BOTH":
            refl_flux = mp.get_fluxes(refl)
            tran_flux = mp.get_fluxes(tran)
            flux_freqs = mp.get_flux_freqs(tran)
        if scattering and mode == "BOTH":
            scatt_flux_faces = [np.asarray(mp.get_fluxes(f)) for f in scatt]

            x1, x2, y1, y2, z1, z2 = scatt_flux_faces

            scatt_flux_total = (
                x1 - x2 +
                y1 - y2 +
                z1 - z2
            )
            scatt_cross_section = scatt_flux_total / intensity # <- from empty
            flux_freqs_scatt = mp.get_flux_freqs(scatt_empty[5])  # z2
        if dft_gap_spectrum and mode == "BOTH":
            gap_data = {
                "Ex": {"empty": gap_data_empty["Ex"], "antenna": []},
                "Ey": {"empty": gap_data_empty["Ey"], "antenna": []},
                "Ez": {"empty": gap_data_empty["Ez"], "antenna": []},
            }

            for dft_a in gap_dft_antenna:

                Ex_a = np.array([
                    sim_antenna.get_dft_array(dft_a, mp.Ex, i)
                    for i in range(nfreq)
                ])
                Ey_a = np.array([
                    sim_antenna.get_dft_array(dft_a, mp.Ey, i)
                    for i in range(nfreq)
                ])
                Ez_a = np.array([
                    sim_antenna.get_dft_array(dft_a, mp.Ez, i)
                    for i in range(nfreq)
                ])

                gap_data["Ex"]["antenna"].append(np.abs(Ex_a)**2)
                gap_data["Ey"]["antenna"].append(np.abs(Ey_a)**2)
                gap_data["Ez"]["antenna"].append(np.abs(Ez_a)**2)

            # numpy
            for comp in gap_data:
                gap_data[comp]["antenna"] = np.array(gap_data[comp]["antenna"])

            gap_data["E2"] = {}

            gap_data["E2"]["antenna"] = (
                gap_data["Ex"]["antenna"] +
                gap_data["Ey"]["antenna"] +
                gap_data["Ez"]["antenna"]
            )

            gap_data["E2"]["empty"] = (
                gap_data["Ex"]["empty"] +
                gap_data["Ey"]["empty"] +
                gap_data["Ez"]["empty"]
            )

            eps = 1e-20

            for comp in gap_data:
                gap_data[comp]["enh"] = (
                    gap_data[comp]["antenna"] /
                    (gap_data[comp]["empty"] + eps)
                )
        if mp.am_master():
            print("Done.")
        sim_antenna.reset_meep()

    # ============================================================
    # TRAN AND REFL CALCULATION
    # ============================================================
    if fluxes and mode == "BOTH":
        if mp.am_master():
            print("Calculating fluxes")
        compute_T_R_A(
            incident_flux,
            tran_flux, refl_flux,
            flux_freqs,
            config.path_to_save)
        if mp.am_master():
            print("Done.")

    # ============================================================
    # SCATT CALCULATION
    # ============================================================        
    if scattering and mode == "BOTH":
        if mp.am_master():
            print("Calculating scattering")
        compute_scattering(
            scatt_cross_section,
            intensity,
            flux_freqs,
            scatt_flux_faces,
            scatt_flux_faces_empty,
            save_path=config.path_to_save,
        )
        if mp.am_master():
            print("Done.")

    # ============================================================
    # GAP DFT DATA COLLECTION
    # ============================================================        
    if dft_gap_spectrum and mode == "BOTH":
        if mp.am_master():
            print("Calculating gap DFT spectrum")
        freqs = np.linspace(fcen - df/2, fcen + df/2, nfreq)
        compute_gap_spectrum(
            gap_data,
            z_points,
            freqs,
            save_path=config.path_to_save,
        )
        if mp.am_master():
            print("Done.")

    # ============================================================
    # HARMINV CALCULATION
    # ============================================================
    if harminv and mode in ["WITH_ANTENNA", "BOTH"]:
        if mp.am_master():
            print("Calculating Harminv modes")
        compute_harminv(
            harminv_objects,
            save_path=config.path_to_save,
        )
        if mp.am_master():
            print("Done.")

    # ============================================================
    # ENHANCEMENT CALCULATION
    # ============================================================

    if (mode == "BOTH" or mode == "ENH_ONLY") and mp.am_master():
        if mp.am_master():
            print("Computing enhancement maps")
            append_time_to_file(config, prefix="Computing enhancement maps: ")
        
        enhancement_planes = [
            "xyplanar",
            "xyplanarTOP",
            "xzplanar",
            "yzplanar",
        ]

        if volumes.extra:
            enhancement_planes.extend([
                "xyplanar_1",
                "xyplanar_2",
                "xyplanar_3",
                "xyplanar_4",
                "xyplanar_5",
                "xzplanar_1",
                "xzplanar_2",
                "xzplanar_3",
                "xzplanar_4",
                "xzplanar_5",
                "yzplanar_1",
                "yzplanar_2",
                "yzplanar_3",
                "yzplanar_4",
                "yzplanar_5",
            ])

        # ---------- E FIELD ENHANCEMENT ----------
        if calc_E:
            for base_name in enhancement_planes:

                enhancement_divided_by_maxes_arr(
                    [f"{base_name}_ex.h5", f"{base_name}_ey.h5", f"{base_name}_ez.h5"],
                    [f"{base_name}-empty_ex.h5", f"{base_name}-empty_ey.h5", f"{base_name}-empty_ez.h5"],
                    save_to=f"enhancement_{base_name}_e2.h5",
                    path=config.path_to_save,
                    out_dataset_name="enhancement",
                )

        # ---------- H FIELD ENHANCEMENT ----------
        if calc_H:
            for base_name in enhancement_planes:

                enhancement_divided_by_maxes_arr(
                    [f"{base_name}_hx.h5", f"{base_name}_hy.h5", f"{base_name}_hz.h5"],
                    [f"{base_name}-empty_hx.h5", f"{base_name}-empty_hy.h5", f"{base_name}-empty_hz.h5"],
                    save_to=f"enhancement_{base_name}_h2.h5",
                    path=config.path_to_save,
                    out_dataset_name="enhancement",
                )
    return 0

def compute_dft_enhancement_maps(
    sim_antenna,
    sim_empty,
    volumes,
    config,
    decay_tol=1e-6,
    max_run_time=None,
    components=(mp.Ex, mp.Ey, mp.Ez),
    monitor_pt=mp.Vector3(0, 0, 0),
    monitor_dt=None,
    make_plots=False,
):
    """
    Compute steady-state (CW) intensity enhancement maps at a single
    frequency f0 = config.frequency using DFT fields:

        enhancement[r] = sum_i |E_i^ant(r, f0)|^2 / sum_i |E_i^empty(r, f0)|^2

    This is the same quantity as the FEF spectra from the resonance-scan
    functions (ratio of DFT amplitudes), so the map peak is directly
    comparable with the FEF value at the resonance wavelength. Unlike the
    time-domain snapshot maps (animate_enhancement_fields), the result does
    not depend on the source pulse shape - the spectral envelope cancels in
    the ratio.

    Both simulations are run with until_after_sources=stop_when_dft_decayed,
    i.e. until the DFT fields converge (source turn-off + resonance
    ring-down), capped by max_run_time [Meep time units].

    A lightweight point monitor records Ex(t) at monitor_pt (default: gap
    centre) in both runs - this replaces the old amplitude-vs-time plots
    from the full-plane h5 dumps and doubles as a ring-down/convergence
    diagnostic.

    Outputs (master process only), written to config.path_to_save:
        dft_enhancement_<plane>_e2.h5   (enhancement, ant_e2, empty_e2, axes)
        field_vs_time_gap.dat           (Ex(t) at monitor_pt, both runs)
    and, only when make_plots=True (PNG rendering does not work on the
    cluster - keep False there and render locally with
    postprocess_dft_enhancement_maps):
        dft_enhancement_<plane>_e2.png  (log-scale map)
        field_vs_time_gap.png

    Returns
    -------
    dict {plane_name: max_enhancement} on master, {} on other processes.
    """
    freq = config.frequency
    if monitor_dt is None:
        monitor_dt = getattr(config, "sim_time_step", 0.05)

    plane_map = {
        "xyplanar": "XY",
        "xyplanarTOP": "XY_TOP",
        "xzplanar": "XZ",
        "yzplanar": "YZ",
    }

    dfts_empty = {}
    dfts_antenna = {}
    for name, key in plane_map.items():
        vol = volumes.volume[key]
        dfts_empty[name] = sim_empty.add_dft_fields(list(components), [freq], where=vol)
        dfts_antenna[name] = sim_antenna.add_dft_fields(list(components), [freq], where=vol)

    # ============================================================
    # EMPTY RUN
    # ============================================================
    if mp.am_master():
        print("Running EMPTY simulation (DFT enhancement maps)")
        append_time_to_file(config, prefix="DFT maps, EMPTY run: ")

    # point monitor Ex(t) - replacement for the old amplitude-vs-time plots
    t_empty, ex_empty = [], []

    def _record_empty(sim):
        t_empty.append(sim.meep_time())
        ex_empty.append(np.real(sim.get_field_point(mp.Ex, monitor_pt)))

    # NOTE: stop_when_dft_decayed returns a stateful closure -> build a
    # fresh one for each run
    sim_empty.run(
        mp.at_every(monitor_dt, _record_empty),
        until_after_sources=mp.stop_when_dft_decayed(
            tol=decay_tol, maximum_run_time=max_run_time
        ))

    if mp.am_master():
        print(f"EMPTY run finished at t = {sim_empty.meep_time():.1f}")

    empty_e2 = {}
    for name in plane_map:
        acc = None
        for c in components:
            arr = sim_empty.get_dft_array(dfts_empty[name], c, 0)
            a2 = np.abs(arr) ** 2
            acc = a2 if acc is None else acc + a2
        empty_e2[name] = acc

    # ============================================================
    # ANTENNA RUN
    # ============================================================
    if mp.am_master():
        print("Running simulation WITH antenna (DFT enhancement maps)")
        append_time_to_file(config, prefix="DFT maps, ANTENNA run: ")

    t_ant, ex_ant = [], []

    def _record_ant(sim):
        t_ant.append(sim.meep_time())
        ex_ant.append(np.real(sim.get_field_point(mp.Ex, monitor_pt)))

    sim_antenna.run(
        mp.at_every(monitor_dt, _record_ant),
        until_after_sources=mp.stop_when_dft_decayed(
            tol=decay_tol, maximum_run_time=max_run_time
        ))

    if mp.am_master():
        print(f"ANTENNA run finished at t = {sim_antenna.meep_time():.1f}")

    antenna_e2 = {}
    coords = {}
    for name in plane_map:
        acc = None
        for c in components:
            arr = sim_antenna.get_dft_array(dfts_antenna[name], c, 0)
            a2 = np.abs(arr) ** 2
            acc = a2 if acc is None else acc + a2
        antenna_e2[name] = acc
        # physical coordinates matching the returned array shape
        # (robust also w.r.t. symmetry-reduced storage)
        coords[name] = sim_antenna.get_array_metadata(
            dft_cell=dfts_antenna[name]
        )[:3]

    # ============================================================
    # ENHANCEMENT MAPS (master only)
    # ============================================================
    results = {}
    if mp.am_master():
        eps = 1e-20

        # ---------- Ex(t) at the monitor point (both runs) ----------
        dat_path = os.path.join(config.path_to_save, "field_vs_time_gap.dat")
        with open(dat_path, "w") as f:
            f.write("# t_empty\tEx_empty\tt_antenna\tEx_antenna\n")
            n_rows = max(len(t_empty), len(t_ant))
            for i in range(n_rows):
                te = f"{t_empty[i]:.4f}\t{ex_empty[i]:.6e}" if i < len(t_empty) else "\t"
                ta = f"{t_ant[i]:.4f}\t{ex_ant[i]:.6e}" if i < len(t_ant) else "\t"
                f.write(te + "\t" + ta + "\n")

        for name, key in plane_map.items():
            enh = antenna_e2[name] / (empty_e2[name] + eps)
            max_enh = float(np.max(enh))
            results[name] = max_enh

            xs, ys, zs = [np.atleast_1d(np.asarray(c)) for c in coords[name]]
            if key.startswith("XY"):
                h_ax, v_ax = xs, ys
            elif key == "XZ":
                h_ax, v_ax = xs, zs
            else:  # YZ
                h_ax, v_ax = ys, zs

            idx = np.unravel_index(np.argmax(enh), enh.shape)
            print(
                f"[{name}] max |E|^2 enhancement = {max_enh:.1f} "
                f"at ({h_ax[idx[0]]*1e3:.1f}, {v_ax[idx[1]]*1e3:.1f}) nm"
            )

            h5_path = os.path.join(
                config.path_to_save, f"dft_enhancement_{name}_e2.h5"
            )
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("enhancement", data=enh)
                f.create_dataset("ant_e2", data=antenna_e2[name])
                f.create_dataset("empty_e2", data=empty_e2[name])
                f.create_dataset("axis_h_um", data=h_ax)
                f.create_dataset("axis_v_um", data=v_ax)
                f.attrs["frequency"] = freq
                f.attrs["wavelength_nm"] = 1e3 / freq
                f.attrs["plane"] = key

        # PNG rendering does not work on the cluster -> keep make_plots=False
        # there and render locally with postprocess_dft_enhancement_maps()
        if make_plots:
            postprocess_dft_enhancement_maps(config.path_to_save)

    return results

def _plot_dft_map(enh, h_nm, v_nm, xlabel, ylabel, title, save_path):
    """Render a single log-scale DFT enhancement map to PNG."""
    from matplotlib.colors import LogNorm

    max_enh = float(np.max(enh))
    extent = [h_nm[0], h_nm[-1], v_nm[0], v_nm[-1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        enh.T,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(vmin=max(max_enh * 1e-4, 1e-3), vmax=max_enh),
        aspect="auto",
    )
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="|E|$^2$ / |E$_0$|$^2$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def postprocess_dft_enhancement_maps(load_path, roi_nm=None):
    """
    LOCAL postprocessing of the dft_enhancement_<plane>_e2.h5 files produced
    on the cluster by compute_dft_enhancement_maps (where PNG rendering is
    disabled). Renders the log-scale PNG maps next to the h5 files and,
    additionally, the Ex(t) diagnostic plot from field_vs_time_gap.dat.

    Parameters
    ----------
    load_path : str
        Folder with dft_enhancement_*.h5 (a single simulation's results).

    roi_nm : dict or None
        Optional ROI (in nm) for gap statistics, per plane family, e.g.:
            {
                "XY": {"center": (0, 0),  "width": 30, "height": 10},
                "XZ": {"center": (0, -2.5), "width": 30, "height": 35},
                "YZ": {"center": (0, -2.5), "width": 10, "height": 35},
            }
        For planes with a matching ROI the mean and max enhancement inside
        the ROI are computed.

    Returns
    -------
    dict {plane_name: {"max": ..., "mean_roi": ..., "max_roi": ...}}
        ("mean_roi"/"max_roi" present only when a ROI was applied)
    """
    plane_labels = {
        "xyplanar":    ("XY", "X [nm]", "Y [nm]"),
        "xyplanarTOP": ("XY", "X [nm]", "Y [nm]"),
        "xzplanar":    ("XZ", "X [nm]", "Z [nm]"),
        "yzplanar":    ("YZ", "Y [nm]", "Z [nm]"),
    }

    stats = {}
    for name, (roi_key, xlabel, ylabel) in plane_labels.items():
        h5_path = os.path.join(load_path, f"dft_enhancement_{name}_e2.h5")
        if not os.path.exists(h5_path):
            print(f"[postprocess_dft] missing {h5_path}, skipping")
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

        _plot_dft_map(
            enh, h_nm, v_nm, xlabel, ylabel,
            title=(f"|E|$^2$ enhancement @ {wavelength_nm:.0f} nm "
                   f"(max = {entry['max']:.0f})"),
            save_path=os.path.join(load_path, f"dft_enhancement_{name}_e2.png"),
        )
        stats[name] = entry

    # Ex(t) diagnostic plot from the point-monitor data
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

    return stats

def get_phys_ranges(bounds, plane):
    if plane == "XY":
        return [bounds["xmin"], bounds["xmax"]], [bounds["ymin"], bounds["ymax"]]

    elif plane == "XZ":
        return [bounds["xmin"], bounds["xmax"]], [bounds["zmin"], bounds["zmax"]]

    elif plane == "YZ":
        return [bounds["ymin"], bounds["ymax"]], [bounds["zmin"], bounds["zmax"]]

    else:
        raise ValueError(f"Unknown plane: {plane}")

def animate_enhancement_fields(config, volumes, draw_params, field='E', animate=True):
    """
    - Animate field enhancement for XY / XZ / YZ planes
    - Plot max-frame field maps with structure + ROI
    - Collect mean |E|^2 enchancement in gap vs time for each plane
    - Plot all mean curves on a single axes using multi_line_plotter_same_axes
    """
    if mp.am_master():
        valid_field = ["E", "H"]
        
        if field not in valid_field:
            raise ValueError(f"field must be one of {valid_field}")
        
        field=field.lower()

        # ============================================================
        # Bounds of planes configuration
        # ============================================================
        b_xy = volumes.bounds["XY"]
        b_xy_top = volumes.bounds["XY_TOP"]
        b_xz = volumes.bounds["XZ"]
        b_yz = volumes.bounds["YZ"]
        
        xy_x, xy_y = get_phys_ranges(b_xy, "XY")
        xy_top_x, xy_top_y = get_phys_ranges(b_xy_top, "XY")
        xz_x, xz_y = get_phys_ranges(b_xz, "XZ")
        yz_x, yz_y = get_phys_ranges(b_yz, "YZ")
        
        xy_x = [v * 1e3 for v in xy_x]
        xy_y = [v * 1e3 for v in xy_y]
        
        xy_top_x = [v * 1e3 for v in xy_top_x]
        xy_top_y = [v * 1e3 for v in xy_top_y]
        
        xz_x = [v * 1e3 for v in xz_x]
        xz_y = [v * 1e3 for v in xz_y]
        
        yz_x = [v * 1e3 for v in yz_x]
        yz_y = [v * 1e3 for v in yz_y]
        # ============================================================
        # Plane configuration
        # ============================================================
        planes = {
            "XY": {
                "filename": f"enhancement_xyplanar_{field}2.h5",
                "save_anim": f"enh_xy_{field}2.mp4",
                "x_phys_range": xy_x,
                "y_phys_range": xy_y,
                "x_zoom": draw_params["XY"]["x_zoom"],
                "y_zoom": draw_params["XY"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Y [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XY"]["roi"]["center"],
                    "width": draw_params["XY"]["roi"]["width"],
                    "height": draw_params["XY"]["roi"]["height"],
                },
            },

            "XYTop": {
                "filename": f"enhancement_xyplanarTOP_{field}2.h5",
                "save_anim": f"enh_xy_TOP_{field}2.mp4",
                "x_phys_range": xy_top_x,
                "y_phys_range": xy_top_y,
                "x_zoom": draw_params["XY"]["x_zoom"],
                "y_zoom": draw_params["XY"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Y [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XY"]["roi"]["center"],
                    "width": draw_params["XY"]["roi"]["width"],
                    "height": draw_params["XY"]["roi"]["height"],
                },
            },

            "XZ": {
                "filename": f"enhancement_xzplanar_{field}2.h5",
                "save_anim": f"enh_xz_{field}2.mp4",
                "x_phys_range": xz_x,
                "y_phys_range": xz_y,
                "x_zoom": draw_params["XZ"]["x_zoom"],
                "y_zoom": draw_params["XZ"]["y_zoom"],
                "xlabel": "X [nm]",
                "ylabel": "Z [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["XZ"]["roi"]["center"],
                    "width": draw_params["XZ"]["roi"]["width"],
                    "height": draw_params["XZ"]["roi"]["height"],
                },
            },

            "YZ": {
                "filename": f"enhancement_yzplanar_{field}2.h5",
                "save_anim": f"enh_yz_{field}2.mp4",
                "x_phys_range": yz_x,
                "y_phys_range": yz_y,
                "x_zoom": draw_params["YZ"]["x_zoom"],
                "y_zoom": draw_params["YZ"]["y_zoom"],
                "xlabel": "Y [nm]",
                "ylabel": "Z [nm]",
                "roi": {
                    "type": "rectangle",
                    "center": draw_params["YZ"]["roi"]["center"],
                    "width": draw_params["YZ"]["roi"]["width"],
                    "height": draw_params["YZ"]["roi"]["height"],
                },
            },
        }

        # ============================================================
        # Containers for line plots
        # ============================================================

        line_xdata = []
        line_ydata = []
        line_labels = []

        # ============================================================
        # Main loop over planes
        # ============================================================

        for plane, cfg in planes.items():
            print(f"Processing {plane} plane")

            if animate:
                # ---------- Animation ----------
                animate_field_from_h5_physical(
                    h5_filename=cfg["filename"],
                    load_h5data_path=config.path_to_save,
                    save_name=cfg["save_anim"],
                    save_path=config.animations_folder_path,
                    interval=50,
                    cmap="inferno",
                    transpose_xy=True,
                    IMG_CLOSE=config.IMG_CLOSE,
                    x_phys_range=cfg["x_phys_range"],
                    y_phys_range=cfg["y_phys_range"],
                    x_zoom=cfg["x_zoom"],
                    y_zoom=cfg["y_zoom"],
                    mask_left=0,
                    mask_right=0,
                    mask_bottom=0,
                    mask_top=0,
                    title=f"Field enhancement |E|²/|E0|² ({plane})",
                    xlabel=cfg["xlabel"],
                    ylabel=cfg["ylabel"],
                )

            # ---------- ROI analysis ----------
            frame_mean, frame_max = analyze_roi_from_h5_physical(
                h5_filename=cfg["filename"],
                load_h5data_path=config.path_to_save,
                roi=cfg["roi"],
                x_phys_range=cfg["x_phys_range"],
                y_phys_range=cfg["y_phys_range"],
            )

            print(f"Max mean enhancement in ROI for {plane}: {frame_max[1]:.2f} at frame {frame_max[0]}")

            # ---------- Max-frame plot ----------
            plot_field_frame_from_h5_physical(
                frame_index=int(frame_max[0]),
                h5_filename=cfg["filename"],
                load_h5data_path=config.path_to_save,
                cmap="inferno",
                transpose_xy=True,
                IMG_CLOSE=config.IMG_CLOSE,
                x_phys_range=cfg["x_phys_range"],
                y_phys_range=cfg["y_phys_range"],
                x_zoom=cfg["x_zoom"],
                y_zoom=cfg["y_zoom"],
                mask_left=0,
                mask_right=0,
                mask_bottom=0,
                mask_top=0,
                roi=cfg["roi"],
                title=f"Field enhancement |E|²/|E0|² ({plane})",
                xlabel=cfg["xlabel"],
                ylabel=cfg["ylabel"],
                save_path=config.animations_folder_path,
                save_name=f"MAP_{plane}.png",
            )

            # ---------- Collect data for joint line plot ----------
            line_xdata.append(frame_mean[:, 0])
            line_ydata.append(frame_mean[:, 1])
            line_labels.append(f"{plane}")

        # ============================================================
        # Joint line plot
        # ============================================================
        colors = cm2c(cm_inferno, 14)
        multi_line_plotter_same_axes(
            xdata_list=line_xdata,
            ydata_list=line_ydata,
            labels=line_labels,
            colors=[colors[0], colors[5], colors[7], colors[9]],
            linestyles=["-", "--", "-.", ":"],
            grid=True,
            xlabel="Time step",
            ylabel="|E|²/|E0|²",
            title="Mean |E|²/|E0|² in gap vs time",
            legend=True,
            save_path=config.animations_folder_path,
            save_name="MEAN_ENHANCEMENT_ALL_PLANES.png",
            IMG_CLOSE=config.IMG_CLOSE,
        )
    return 0

def compute_T_R_A(
    incident_flux,
    tran_flux,
    refl_flux,
    flux_freqs,
    save_path=None,
    save_name="spectra_TRA.txt"
):
    """
    Compute reflection (R), transmission (T), absorption (A)
    and wavelength from Meep flux monitors.

    Parameters
    ----------
    incident_flux : list or array
        Flux through transmission monitor in empty simulation.

    tran_flux : list or array
        Flux through transmission monitor with structure.

    refl_flux : list or array
        Flux through reflection monitor with structure.

    flux_freqs : list or array
        Frequencies returned by mp.get_flux_freqs().

    save_path : str or None
        Directory where results will be saved.

    save_name : str
        Name of the output file.

    Returns
    -------
    wavelength, R, T, A : numpy arrays
    """
    if not mp.am_master():
        return

    incident_flux = np.array(incident_flux)
    tran_flux = np.array(tran_flux)
    refl_flux = np.array(refl_flux)
    flux_freqs = np.array(flux_freqs)

    # -----------------------------------------
    # wavelength
    # -----------------------------------------
    wavelength = 1.0 / flux_freqs

    # -----------------------------------------
    # R T A
    # -----------------------------------------
    R = -refl_flux / incident_flux
    T = tran_flux / incident_flux
    A = 1.0 - R - T

    # -----------------------------------------
    # CREATE TRA FOLDER
    # -----------------------------------------
    if save_path is not None:
        tra_dir = os.path.join(save_path, "TRA")
        os.makedirs(tra_dir, exist_ok=True)
    else:
        tra_dir = None

    # -----------------------------------------
    # SAVE DATA
    # -----------------------------------------
    if tra_dir is not None:

        data = np.column_stack((wavelength, R, T, A))
        header = "lambda  R  T  A"

        np.savetxt(
            os.path.join(tra_dir, save_name),
            data,
            header=header
        )

    # -----------------------------------------
    # PLOT
    # -----------------------------------------
    multi_line_plotter_same_axes(
        xdata_list=[wavelength, wavelength, wavelength],
        ydata_list=[R, T, A],
        labels=["R", "T", "A"],
        colors=["blue", "red", "green"],
        linestyles=["-", "--", "-."],
        xlabel="Wavelength [μm]",
        ylabel="Fraction",
        title="Reflection, Transmission, Absorption Spectra",
        legend=True,
        save_path=tra_dir,
        save_name="spectra_T_R_A.png",
    )

    return wavelength, R, T, A

def make_scattering_box(
    antenna,
    config,
    padding_perc=1,
    extra_padding_nm=(0, 0, 0),  # (dx, dy, dz) in nm
):
    Lx, Ly, Lz = antenna.bounding_box()

    # -----------------------------------------
    # minimal padding from grid
    # -----------------------------------------
    lower_limit = (1 / config.resolution) * 2
    check = padding_perc / 100 * min(Lx, Ly, Lz)

    if check < lower_limit:
        print(f"Warning: padding_perc {padding_perc:.2f}% too small\n")
        padding_perc = lower_limit / min(Lx, Ly, Lz) * 100
        print(f"Increased to {padding_perc:.2f}% for sufficient grid padding.\n")

    Lx = Lx * (1 + padding_perc / 100)
    Ly = Ly * (1 + padding_perc / 100)
    Lz = Lz * (1 + padding_perc / 100)

    # -----------------------------------------
    # anizotropy correction
    # -----------------------------------------
    dx, dy, dz = extra_padding_nm

    xm = 1000  # nm to μm

    Lx += dx / xm
    Ly += dy / xm
    Lz += dz / xm

    Lx = np.ceil(Lx * xm) / xm
    Ly = np.ceil(Ly * xm) / xm
    Lz = np.ceil(Lz * xm) / xm

    print(
        f"Scattering box dimensions:\n"
        f"  Lx = {Lx*1000:.2f} nm\n"
        f"  Ly = {Ly*1000:.2f} nm\n"
        f"  Lz = {Lz*1000:.2f} nm\n"
        f"(padding={padding_perc:.2f}%, extra={extra_padding_nm} nm)"
    )

    return Lx, Ly, Lz

def compute_scattering(
    scatt_cross_section,
    intensity,
    flux_freqs,
    scatt_flux_faces,
    scatt_flux_faces_empty,
    save_path=None,
    save_name="spectra_scattering.txt",
    save_faces_name="scattering_faces.txt",
    save_faces_empty_name="scattering_faces_empty.txt",
):
    """
    Compute and save scattering results from Meep flux monitors.

    Parameters
    ----------
    scatt_cross_section : array-like
        Scattering cross-section (sigma_scatt).

    intensity : array-like
        Incident intensity (W/area), frequency dependent.

    flux_freqs : array-like
        Frequencies from mp.get_flux_freqs().

    scatt_flux_faces : list of arrays
        Flux through each face [x1, x2, y1, y2, z1, z2].

    scatt_flux_faces_empty : list of arrays
        Flux through each face in empty cell [x1, x2, y1, y2, z1, z2].

    save_path : str or None
        Directory where results will be saved.

    save_name : str
        Name of scattering spectrum file.

    save_faces_name : str
        Name of face flux output file.

    save_faces_empty_name : str
        Name of empty face flux output file.

    Returns
    -------
    wavelength, scatt_cross_section : numpy arrays
    """

    if not mp.am_master():
        return

    # -----------------------------------------
    # Convert to numpy
    # -----------------------------------------
    scatt_cross_section = np.array(scatt_cross_section)
    intensity = np.array(intensity)
    flux_freqs = np.array(flux_freqs)

    scatt_flux_faces = [np.array(f) for f in scatt_flux_faces]
    x1, x2, y1, y2, z1, z2 = scatt_flux_faces
    ex1, ex2, ey1, ey2, ez1, ez2 = scatt_flux_faces_empty

    # -----------------------------------------
    # wavelength
    # -----------------------------------------
    wavelength = 1.0 / flux_freqs

    # -----------------------------------------
    # CREATE SCATTERING FOLDER
    # -----------------------------------------
    if save_path is not None:
        scattering_dir = os.path.join(save_path, "scattering")
        os.makedirs(scattering_dir, exist_ok=True)
    else:
        scattering_dir = None

    # -----------------------------------------
    # FILE 1: scattering spectrum
    # -----------------------------------------
    if scattering_dir is not None:

        data_main = np.column_stack(
            (wavelength, scatt_cross_section, intensity)
        )

        header_main = (
            "# lambda(um)  sigma_scatt  intensity(W/um^2)\n"
            "# sigma_scatt normalized by local incident intensity"
        )

        np.savetxt(
            os.path.join(scattering_dir, save_name),
            data_main,
            header=header_main
        )

    # -----------------------------------------
    # FILE 2: flux per face
    # -----------------------------------------
    if scattering_dir is not None:

        data_faces = np.column_stack(
            (flux_freqs, x1, x2, y1, y2, z1, z2)
        )

        header_faces = "# freq  x1  x2  y1  y2  z1  z2"

        np.savetxt(
            os.path.join(scattering_dir, save_faces_name),
            data_faces,
            header=header_faces
        )

    # -----------------------------------------
    # FILE 3: z split
    # -----------------------------------------
    if scattering_dir is not None:

        data_z = np.column_stack(
            (flux_freqs, z1, z2)
        )

        header_z = "# freq  z_down(z1)  z_up(z2)"

        np.savetxt(
            os.path.join(scattering_dir, "scattering_z_split.txt"),
            data_z,
            header=header_z
        )

    # -----------------------------------------
    # FILE 4: empty faces
    # -----------------------------------------
    if scattering_dir is not None:

        data_faces_empty = np.column_stack(
            (flux_freqs, ex1, ex2, ey1, ey2, ez1, ez2)
        )

        header_faces = "# freq  x1  x2  y1  y2  z1  z2"

        np.savetxt(
            os.path.join(scattering_dir, save_faces_empty_name),
            data_faces_empty,
            header=header_faces
        )

    # -----------------------------------------
    # PLOTS
    # -----------------------------------------
    line_plotter(
        wavelength,
        scatt_cross_section,
        xlabel="Wavelength [μm]",
        ylabel="Scattering cross-section",
        title="Scattering Spectrum",
        save_path=scattering_dir,
        save_name="spectra_scattering.png",
    )

    multi_line_plotter_same_axes(
        xdata_list=[wavelength]*6,
        ydata_list=[x1, x2, y1, y2, z1, z2],
        labels=["x1", "x2", "y1", "y2", "z1", "z2"],
        colors=["#149dff", "#14517c", "#ff7700", "#914300", "#5ec75e", "#205220"],
        linestyles=["-", "-.", "-", "-.", "-", "-."],
        xlabel="Wavelength [μm]",
        ylabel="Scattering",
        title="Scattering Spectrum",
        legend=True,
        save_path=scattering_dir,
        save_name="spectra_scattering_each_face.png",
    )

    multi_line_plotter_same_axes(
        xdata_list=[wavelength]*6,
        ydata_list=[ex1, ex2, ey1, ey2, ez1, ez2],
        labels=["x1", "x2", "y1", "y2", "z1", "z2"],
        colors=["#149dff", "#14517c", "#ff7700", "#914300", "#5ec75e", "#205220"],
        linestyles=["-", "-.", "-", "-.", "-", "-."],
        xlabel="Wavelength [μm]",
        ylabel="Scattering",
        title="Scattering Spectrum for empty cell",
        legend=True,
        save_path=scattering_dir,
        save_name="spectra_scattering_each_face_empty.png",
    )

    return wavelength, scatt_cross_section

def compute_gap_spectrum(
    gap_data,
    z_points,
    freqs,
    save_path=None,
):
    if not mp.am_master():
        return

    freqs = np.array(freqs)
    wavelength = 1.0 / freqs
    z_points = np.array(z_points)

    gap_dir = os.path.join(save_path, "gap_spec")
    os.makedirs(gap_dir, exist_ok=True)

    # =========================================
    # SAVE FILES PER POINT
    # =========================================
    for zi, z in enumerate(z_points):

        z_str = f"{z:.6f}".replace(".", "p")

        for comp in ["Ex", "Ey", "Ez", "E2"]:

            empty = gap_data[comp]["empty"][zi]
            antenna = gap_data[comp]["antenna"][zi]
            enh = gap_data[comp]["enh"][zi]

            data = np.column_stack((
                wavelength,
                empty,
                antenna,
                enh
            ))

            header = "wavelength  empty(|E|^2)  antenna(|E|^2)  enhancement"

            fname = os.path.join(
                gap_dir,
                f"{comp}_z_{z_str}.txt"
            )

            np.savetxt(fname, data, header=header)

    # =========================================
    # PLOTS
    # =========================================
    for comp in ["E2", "Ex", "Ey", "Ez"]:
        plot_gap_component(
            component_name=comp,
            gap_data=gap_data,
            z_points=z_points,
            wavelength=wavelength,
            save_path=gap_dir,
        )

def plot_gap_component(
    component_name,
    gap_data,
    z_points,
    wavelength,
    save_path,
):

    comp_empty = gap_data[component_name]["empty"]
    comp_ant = gap_data[component_name]["antenna"]
    comp_enh = gap_data[component_name]["enh"]

    # =========================================
    # GENERATE COLORS FROM COLORMAP
    # =========================================
    cmap = plt.get_cmap("inferno")
    n = len(z_points)
    if n == 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (n - 1)) for i in range(n)]

    def make_plot(data, label_suffix, filename):

        xdata_list = [wavelength for _ in range(n)]
        ydata_list = [data[i] for i in range(n)]
        labels = [f"z={z:.3f}" for z in z_points]

        multi_line_plotter_same_axes(
            xdata_list=xdata_list,
            ydata_list=ydata_list,
            labels=labels,
            colors=colors,
            xlabel="Wavelength [μm]",
            ylabel=f"{component_name} {label_suffix}",
            title=f"{component_name} {label_suffix} spectrum along gap",
            legend=False,
            save_path=save_path,
            save_name=filename,
        )

    # --- EMPTY ---
    make_plot(
        comp_empty,
        "|E|² (empty)",
        f"{component_name}_empty_all_z.png"
    )

    # --- ANTENNA ---
    make_plot(
        comp_ant,
        "|E|² (antenna)",
        f"{component_name}_antenna_all_z.png"
    )

    # --- ENHANCEMENT ---
    make_plot(
        comp_enh,
        "enhancement",
        f"{component_name}_enh_all_z.png"
    )

def compute_harminv(
    harminv_objects,
    save_path=None,
    save_name="harminv_modes.txt",
):
    """
    Process and save Harminv results.

    Creates a dedicated folder 'harminv' and stores:
        - raw data file
        - plots: amplitude, Q, error vs frequency

    Parameters
    ----------
    harminv_objects : list of tuples
        [(point, harminv_instance), ...]

    save_path : str or None
        Output directory.

    save_name : str
        Output filename.
    """

    if not mp.am_master():
        return

    # -----------------------------------------
    # create folder
    # -----------------------------------------
    harminv_dir = os.path.join(save_path, "harminv")
    os.makedirs(harminv_dir, exist_ok=True)

    # -----------------------------------------
    # collect data
    # -----------------------------------------
    all_data = []

    for i, (pt, h) in enumerate(harminv_objects):

        x, y, z = pt.x, pt.y, pt.z

        for m in h.modes:

            freq = m.freq
            Q = m.Q
            amp = np.abs(m.amp)
            err = m.err
            decay = m.decay

            all_data.append([
                i, x, y, z,
                freq, Q, decay, amp, err
            ])

    if len(all_data) == 0:
        print("No Harminv modes found.")
        return

    all_data = np.array(all_data)

    # -----------------------------------------
    # save data
    # -----------------------------------------
    header = (
        "id x y z freq Q decay amplitude error\n"
        "Harminv modal analysis"
    )

    np.savetxt(
        os.path.join(harminv_dir, save_name),
        all_data,
        header=header
    )

    # -----------------------------------------
    # split columns
    # -----------------------------------------
    freq = all_data[:, 4]
    Q = all_data[:, 5]
    decay = all_data[:, 6]
    amp = all_data[:, 7]
    err = all_data[:, 8]

    # -----------------------------------------
    # PLOTS
    # -----------------------------------------

    # --- Amplitude ---
    line_plotter(
        freq,
        amp,
        xlabel="Frequency",
        ylabel="|Amplitude|",
        title="Harminv: Amplitude vs Frequency",
        save_path=harminv_dir,
        save_name="harminv_amplitude.png",
    )

    # --- Q factor ---
    line_plotter(
        freq,
        Q,
        xlabel="Frequency",
        ylabel="Q factor",
        title="Harminv: Q vs Frequency",
        save_path=harminv_dir,
        save_name="harminv_Q.png",
    )

    # --- decay ---
    line_plotter(
        freq,
        decay,
        xlabel="Frequency",
        ylabel="Decay",
        title="Harminv: Decay vs Frequency",
        save_path=harminv_dir,
        save_name="harminv_decay.png",
    )

    # --- Error ---
    line_plotter(
        freq,
        err,
        xlabel="Frequency",
        ylabel="Error",
        title="Harminv: Error vs Frequency",
        save_path=harminv_dir,
        save_name="harminv_error.png",
    )
    return all_data
