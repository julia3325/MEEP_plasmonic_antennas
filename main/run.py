import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    
    # default values
    # mode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    # material_name = sys.argv[2] if len(sys.argv) > 2 else "air"
    mode = 1

    if mode == 1:
        bowtie_calculate_resonant_peaks()

    elif mode == 2:
        bowtie_AuTiSiO2_opt()

    elif mode == 3:
        hybridbar_calculate_resonant_peaks()

    elif mode == 4:
        hybridbar_AuTiSiO2_opt()

    elif mode == 5:
        splitbar_calculate_resonant_peaks()

    elif mode == 6:
            check_antenna_geometry(kind="hybridbar", gap=30, W=240, L_bar=1800, L_tip=200,
                           length=1100, radius=5, resolution=200,
                           with_substrate=True, view_nm=None,
                           out_dir="results/geom_check")
            check_antenna_geometry(kind="hybridbar", gap=10, W=240, L_bar=1800, L_tip=200,
                                       length=1100, radius=5, resolution=200,
                                       with_substrate=True, view_nm=None,
                                       out_dir="results/geom_check")
            check_antenna_geometry(kind="hybridbar", gap=50, W=240, L_bar=1800, L_tip=200,
                                                   length=1100, radius=5, resolution=200,
                                                   with_substrate=True, view_nm=None,
                                                   out_dir="results/geom_check")
            check_antenna_geometry(kind="hybridbar", gap=10, W=240, L_bar=1800, L_tip=200,
                                                   length=1100, radius=0, resolution=200,
                                                   with_substrate=True, view_nm=None,
                                                   out_dir="results/geom_check")
    
    else:
        print("Invalid mode. Please choose a mode between 1 and 2.")

if __name__ == "__main__":
    run()
