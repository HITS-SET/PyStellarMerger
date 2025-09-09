from PyStellarMerger.io import stellarmodel
from PyStellarMerger.pymmams import mmas
from PyStellarMerger.calc.computeenergy import compute_stellar_energy
from PyStellarMerger.calc.massloss import mass_loss
from PyStellarMerger.remeshing.mixing import mixing_product
from PyStellarMerger.remeshing.remeshing import mix_separately
import time
import argparse
import numpy as np
from PyStellarMerger.remeshing import remeshing
from datetime import datetime
import pathlib
import json
from collections import defaultdict
import os

def read_input(infile):
    """
    Read input parameters from infile and check their validity. If a parameter is not present in the input file, use the default value.
    """
    # Set default values
    defaults = {
            'primary_star': 'primary.data', 
            'secondary_star': 'secondary.data', 
            'n_target_shells': 10000, 
            'enable_mixing': False, 
            'mixing_shells': 200, 
            'enable_remeshing': True, 
            'remeshing_shells': 110, 
            'enable_shock_heating': True, 
            'f_mod': 1.0, 
            'relaxation_profiles': True, 
            'extrapolate_shock_heating': True, 
            'initial_buoyancy': False, 
            'final_buoyancy': False, 
            'chemical_species': ['h1', 'he3', 'he4', 'c12', 'n14', 'o16'],
            'fill_missing_species': False,
            'massloss_fraction': -1.0, 
            'output_dir': '', 
            'output_diagnostics': True
            }

    parameters = defaultdict(lambda: None, defaults)

    # Load parameters from input file, only updating the parameters that are present
    with open(infile, "r") as f:
        parameters.update(json.load(f))

    # Validate some input parameters:
    if not isinstance(parameters["n_target_shells"], int) or parameters["n_target_shells"] < 0:
        raise ValueError("n_target_shells must be an integer > 0.")
    
    if not isinstance(parameters["mixing_shells"], int) or parameters["mixing_shells"] < 0:
        raise ValueError("mixing_shells must be an integer > 0.")
    
    if not isinstance(parameters["remeshing_shells"], int) or parameters["remeshing_shells"] < 0:
        raise ValueError("remeshing_shells must be an integer > 0.")

    if not (isinstance(parameters["f_mod"], float) or isinstance(parameters["f_mod"], int)) or parameters["f_mod"] < 0.0:
        raise ValueError("f_mod must be >= 0.0.")
    
    if not (isinstance(parameters["massloss_fraction"], float), isinstance(parameters["massloss_fraction"], int)) or parameters["massloss_fraction"] >= 1.0:
        raise ValueError("massloss_fraction must be < 1.0.")
    
    return parameters

def main():
    """
    Main method of the merger code.
    Start by getting the merger settings from the input file given in the command line call, then load progenitors.
    Continue by merging and mixing/remeshing. Finally, write merger product to file.
    """
    parser = argparse.ArgumentParser(description="Python Make Me A [Massive] Star",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", nargs=1, action="store", type=str,
                        help="Input JSON file specifying merger parameters.",
                        metavar=("input_file"), required=True, default=["input.json"])
    args = parser.parse_args()
    parameters = read_input(args.i[0])

    start = time.time()  # For measuring the time taken for the merger

    # Set some initial parameters
    shock_heating_flag = parameters["enable_shock_heating"]
    dump_mixed = parameters["enable_mixing"]
    chemical_species = np.array(parameters["chemical_species"])
    fill_missing_species = parameters["fill_missing_species"]
    relaxation_profiles = parameters["relaxation_profiles"]
    remesh = parameters["enable_remeshing"]
    remeshing_shells = parameters["remeshing_shells"]
    diagnostics = parameters["output_diagnostics"]

    if parameters["massloss_fraction"] < 0:
        mass_loss_flag = False # If massloss_fraction is negative, disable constant mass loss and use the inbuilt mass loss prescription
        mass_loss_fraction = 0.0 # Has no effect, as inbuilt prescription will be used
    else:
        mass_loss_flag = True
        mass_loss_fraction = float(parameters["massloss_fraction"])

    sh_extrapolation = parameters["extrapolate_shock_heating"]
    initial_buoyancy = parameters["initial_buoyancy"]
    final_buoyancy = parameters["final_buoyancy"]
    f_heat_factor = float(parameters["f_mod"])

    # Transform the input file paths to absolute paths
    primary_file = os.path.abspath(parameters["primary_star"])
    secondary_file = os.path.abspath(parameters["secondary_star"])

    n_shells = parameters["n_target_shells"] # Target number of shells for the product
    n_mixing_shells = parameters["mixing_shells"]  # Number of shells over which is averaged when mixing

    # Check if the output directory exists, if not, create it
    output_folder = os.path.abspath(parameters["output_dir"])

    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except FileNotFoundError as exc:
            print(exc)
            print("Could not find parent directory of output folder. Quit.")
            exit(1)

    print("\nPython Make Me A [Massive] Star\n")
    print(f"Computation started at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Primary: {primary_file}")
    print(f"Secondary: {secondary_file}")
    print(f"Desired number of shells before mixing: {n_shells}")
    print(f"Perform mixing: {dump_mixed}")
    print(f"Number of mixing shells: {n_mixing_shells}")
    print(f"Perform remeshing: {remesh}")
    print(f"Number of remeshing shells: {remeshing_shells}")
    print(f"Extrapolation of shock-heating parameters: {sh_extrapolation}")
    print(f"Output progenitors before shock-heating: {initial_buoyancy}")
    print(f"Output progenitors after shock-heating: {final_buoyancy}")
    print(f"Chemical species: {', '.join(str(i) for i in chemical_species)}")
    print(f"Fill missing chemical species with zero: {fill_missing_species}")
    print(f"Constant mass loss: {mass_loss_flag}")
    if mass_loss_flag:
        print(f"Mass loss fraction: {mass_loss_fraction}")
    print(f"Perform shock-heating: {shock_heating_flag}")
    print(f"Shock heating factor: {f_heat_factor}")
    print(f"Generate relaxation profiles: {relaxation_profiles}")
    print(f"Output folder: {output_folder}\n")

    # Begin the merger procedure by initializing the models
    model_a = stellarmodel.model(chemical_species) # Primary component
    model_b = stellarmodel.model(chemical_species) # Secondary component

    r_p = 0.0
    v_inf = 0.0

    # Check if input files are .txt or profile, then read the models.
    if primary_file[-5::] == ".data" or primary_file[-8::] == ".data.gz": # If the input files are (gzipped) MESA models, use MESA reader to load the stars
        model_a.read_mesa_profile(primary_file, fill_missing_species)
        model_b.read_mesa_profile(secondary_file, fill_missing_species)
    elif primary_file[-4::] == ".txt": # If the input models are in the simple column format, use our basic loading function
        model_a.read_basic(primary_file)
        model_b.read_basic(secondary_file)
    else:
        print("Unknown progenitor file format, quit.")
        exit(0)

    # Set the passive scalar for tracking material coming from the primary and secondary
    model_a.passive_scalar = np.ones(model_a.n_shells)  # Primary: Passive scalar = 1
    model_b.passive_scalar = np.zeros(model_b.n_shells)  # Secondary: Passive scalar = 0

    # Check if the input models are identical, if so, set the twin flag which will adjust the shock heating procedure such that both progenitors get heated by the same amount
    if model_a.n_shells == model_b.n_shells:
        if model_a == model_b:
            model_a.twin_flag = True
            model_b.twin_flag = True
            print("Identical progenitors detected, adjusted shock heating procedure.")

    # Initialize a merger object and merge the stars
    merger = mmas.mmas(model_a, model_b, r_p, v_inf)
    model_p = merger.merge_stars_consistently(n_shells, shock_heating_flag, mass_loss_flag, mass_loss_fraction, output_folder, f_heat_factor, sh_extrapolation, initial_buoyancy, final_buoyancy)

    # Print the total energy in the progenitors and the product
    energy_a = compute_stellar_energy(model_a)
    energy_b = compute_stellar_energy(model_b)
    energy_p = compute_stellar_energy(model_p)
    print(f"energy_a = {energy_a}")
    print(f"energy_b = {energy_b}")
    print(f"energy_a + energy_b = {energy_a+energy_b}, energy_p = {energy_p}")

    # Calculate and print the relative energy difference between the progenitors and the product, scaled with 1/(fractional mass loss). Not output when mass_loss_fractio = 0.
    de = (energy_p - (energy_a + energy_b)) / (energy_a + energy_b)
    try:
        de *= 1.0 / (mass_loss(model_a.star_mass, model_b.star_mass, mass_loss_flag, mass_loss_fraction) / 100.0)
        print(f"de = {de}")
    except ZeroDivisionError: 
        print(f"Zero mass loss assumed, can't compute de.")

    # Write unmixed product to file
    model_p.write_basic(os.path.join(output_folder, "merged_unmixed.txt"))

    # Apply mixing (Gaburov+2008) and save product if according flag is set
    if dump_mixed:  
        mixed_p = mixing_product(n_mixing_shells, model_p)
        print("Writing mixed product to file.")
        mixed_p.write_basic(os.path.join(output_folder, "merged_mixed.txt"))
        if relaxation_profiles:
            print("Writing entropy profile...")
            mixed_p.write_entropy_profile(os.path.join(output_folder, "merger_mixed_entropy.dat"))
            print("Writing composition profile...")
            mixed_p.write_composition_profile(os.path.join(output_folder, "merger_mixed_composition.dat"))

    # Apply custom remeshing scheme (Heller+2025) and write to file
    if remesh:
        remeshed_p = mix_separately(model_p, remeshing_shells)
        print("Writing remeshed product to file.")
        remeshed_p.write_basic(os.path.join(output_folder, "merged_remeshed.txt"))
        if relaxation_profiles:
            print("Writing entropy profile...")
            remeshed_p.write_entropy_profile(os.path.join(output_folder, "merger_remeshed_entropy.dat"))
            print("Writing composition profile...")
            remeshed_p.write_composition_profile(os.path.join(output_folder, "merger_remeshed_composition.dat"))

    end = time.time()
    t_min = int((end-start)/60)
    t_sec = int(round((end-start) % 60, 0))

    # If set, write some information about the merger to file
    if diagnostics:
        with open(os.path.join(output_folder, "merger_info.txt"), "w") as f:
            f.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write(f"Primary: {primary_file}\n")
            f.write(f"Secondary: {secondary_file}\n")
            f.write(f"Chemical species: {', '.join(str(i) for i in chemical_species)}\n")
            f.write(f"Desired shells: {n_shells}\n")
            f.write(f"Perform shock heating: {shock_heating_flag}\n")
            f.write(f"Shock heating modification factor: {f_heat_factor}\n")
            f.write(f"Mix model: {dump_mixed}\n")
            if dump_mixed:
                f.write(f"Number of mixing shells: {n_mixing_shells}\n")
            f.write(f"Remesh model: {remesh}\n")
            if remesh:
                f.write(f"Number of remeshing shells: {remeshing_shells}\n")
            f.write(f"Constant mass loss: {mass_loss_flag}\n")
            if mass_loss_flag:
                f.write(f"Mass loss fraction: {mass_loss_fraction}\n")
            else:
                f.write(f"Mass loss fraction: {mass_loss(model_a.star_mass, model_b.star_mass, mass_loss_flag, mass_loss_fraction) / 100.0}\n")
            f.write(f"Generate relaxation profiles: {relaxation_profiles}\n")
            f.write(f"Output folder: {output_folder}\n")
            f.write(f"Time taken: {t_min:02}:{t_sec:02} minutes\n")

    print(f"Merger done! Time taken: {t_min:02}:{t_sec:02} minutes.")


if __name__ == "__main__":
    main()
