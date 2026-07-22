import sys, os, meep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments import *

def run():
    meep.Simulation.eps_averaging = False
    
    # default values
    # mode = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    # material_name = sys.argv[2] if len(sys.argv) > 2 else "air"
    mode = 4

    if mode == 1:
        calculate_resonant_peaks()

    elif mode == 2:
        bowtie_AuTiSiO2_opt()

    elif mode == 3:
        hybridbar_calculate_resonant_peaks()

    elif mode == 4:
        hybridbar_AuTiSiO2_opt()
    
    else:
        print("Invalid mode. Please choose a mode between 1 and 2.")

if __name__ == "__main__":
    run()
