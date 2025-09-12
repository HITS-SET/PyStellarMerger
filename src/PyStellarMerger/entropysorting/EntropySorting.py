import argparse
import numpy as np
from scipy import constants as con
import mesa_reader as mr
import os
from datetime import datetime
import pathlib
import json
import pickle
from collections import defaultdict
from datetime import datetime
from PyStellarMerger.io import stellarmodel
from PyStellarMerger.calc.massloss import mass_loss
import importlib.resources

def read_input(infile):
    """
    Read input parameters from infile and check their validity.
    """
    # Set default values
    defaults = {'primary_star': 'primary.data', 
            'secondary_star': 'secondary.data', 
            'output_raw': True, 
            'enable_remeshing': True, 
            'remeshing_shells': 110, 
            'enable_mixing': False, 
            'mixing_shells': 200, 
            'relaxation_profiles': True, 
            'chemical_species': ['h1', 'he3', 'he4', 'c12', 'n14', 'o16'], 
            'fill_missing_species': False,
            'massloss_fraction': 0.05, 
            'output_dir': '', 
            'output_diagnostics': True}

    parameters = defaultdict(lambda: None, defaults)

    # Load parameters from input file, only updating the parameters that are present
    with open(infile, "r") as f:
        parameters.update(json.load(f))

    # Validate some input parameters:    
    if not isinstance(parameters["remeshing_shells"], int) or parameters["remeshing_shells"] < 0:
        raise ValueError("remeshing_shells must be an integer > 0.")
    
    if not isinstance(parameters["mixing_shells"], int) or parameters["mixing_shells"] < 0:
        raise ValueError("mixing_shells must be an integer > 0.")
    
    if not (isinstance(parameters["massloss_fraction"], float) or isinstance(parameters["massloss_fraction"], int)) or parameters["massloss_fraction"] >= 1.0 or parameters["massloss_fraction"] < 0.0:
        raise ValueError("massloss_fraction must be < 1.0 and >= 0.0.")
    
    return parameters

def get_amass(chemical_species):
    """
    Gets the atomic masses of all chemical species from the isotopes.pickle file.
    """
    with importlib.resources.files("PyStellarMerger.data").joinpath("isotopes.pkl").open("rb") as f:
        isotopes = pickle.load(f)
        amass = np.array([(1+float(isotopes[iso][1]))/float(isotopes[iso][0]) for iso in chemical_species])

    return amass

def new_star(m, s, t, rho, comp, ps):
    """
    Merges the progenitors by stacking their shells such that the product has monotonically increasing entropy.
    """

    sorted_idxs = np.argsort(s)
    s_res = s[sorted_idxs]
    dm_res = m[sorted_idxs]
    t_res = t[sorted_idxs]
    rho_res = rho[sorted_idxs]
    comp_res = comp[:, sorted_idxs]
    ps_res = ps[sorted_idxs]

    return dm_res, s_res, t_res, rho_res, comp_res, ps_res


def mloss_remesh(dm, elements, X, mu, ps, density, s_tot, n_shells, f_ml, amass):
    """
    The same remeshing scheme as in PyMMAMS (remesh_func in remeshing.py). 
    This function additionally applies mass loss before remeshing by removing the mass from the surface.
    We also do not separate "single-" and "double-valued" regions, because ES tends to produce less distinct transitions between regions of different origin (primary vs. secondary).
    """
    # msol = 1.9892e33
    # dm /= msol  # Convert dm from grams to msun
    mass = np.cumsum(dm)
    last_mass = mass
    dm_lim = mass[-1]/n_shells

    # First, apply mass loss by removing the mass from the surface
    deltaM = mass[-1]*f_ml
    ml_idx = np.argmin(abs(mass - (mass[-1] - deltaM))) + 1  # +1 because we want to include the final shell when slicing
    
    mass = mass[0:ml_idx]
    dm = dm[0:ml_idx]
    elements = elements[:, 0:ml_idx]
    X = X[0:ml_idx]
    ps = ps[0:ml_idx]
    density = density[0:ml_idx]
    s_tot = s_tot[0:ml_idx]
    mu = mu[0:ml_idx]

    while True:
        dmu = np.zeros(mu.shape[0])
        comb_id = np.array([], dtype=int)

        skip = False
        skip_twice = False

        for i in np.arange(1, len(mu)-1, 1):
            dmu[i] = abs(mu[i] - mu[i-1]) # Calculate the absolute difference in mean molecular weight between adjacent shells

        for i in np.arange(1, len(mu)-1, 1):  # Find pairs of shells that can be combined
            if skip:
                skip = False
                continue
            if skip_twice:
                skip_twice = False
                continue

            if ps[i-1] == ps[i] and (dm[i-1] + dm[i]) < dm_lim:
                comb_id = np.append(comb_id, i-1)
                skip = True
                continue
            elif ps[i] == ps[i+1] and (dm[i] + dm[i+1]) < dm_lim:
                comb_id = np.append(comb_id, i)
                skip = True
                skip_twice = True
                continue

            elif dmu[i-1] <= dmu[i] and (dm[i-1] + dm[i]) < dm_lim:
                comb_id = np.append(comb_id, i-1)
                skip = True  # If two shells can be combined skip next shell in order to not combine an already combined shell
            elif dmu[i-1] > dmu[i] and (dm[i] + dm[i+1]) < dm_lim:
                comb_id = np.append(comb_id, i)
                skip = True
                skip_twice = True

        new_mass = np.array([])
        new_dm = np.array([])
        new_X = np.array([])
        new_elements = np.zeros((elements.shape[0], 1))  # Dummy entry such that we can write to the array directly 
        new_ps = np.array([])
        new_density = np.array([])
        new_s_tot = np.array([])
        new_mu = np.array([])

        # Combine determined pairs of shells:
        skip = False
        for j in range(len(mu)):
            if skip:
                skip = False
                continue
            if j in comb_id:
                new_mass = np.append(new_mass, mass[j] + dm[j+1])
                new_dm = np.append(new_dm, dm[j] + dm[j+1])
                new_X = np.append(new_X, (dm[j]*X[j] + dm[j+1]*X[j+1]) / (dm[j]+dm[j+1]))
                new_elements = np.append(new_elements, np.reshape((dm[j]*elements[:, j]+dm[j+1]*elements[:, j+1])/(dm[j]+dm[j+1]), (elements.shape[0], 1)), axis=1)
                if j == 0:  # Delete first dummy entry after writing first shell's composition
                    new_elements = np.delete(new_elements, [0], axis=1)
                new_ps = np.append(new_ps, (dm[j]*ps[j]+dm[j+1]*ps[j+1])/(dm[j]+dm[j+1]))
                new_density = np.append(new_density, (dm[j]+dm[j+1])/((dm[j]/density[j])+(dm[j+1]/density[j+1])))
                new_s_tot = np.append(new_s_tot, (dm[j]*s_tot[j]+dm[j+1]*s_tot[j+1])/(dm[j]+dm[j+1]))
                skip = True
            else:
                new_mass = np.append(new_mass, mass[j])
                new_dm = np.append(new_dm, dm[j])
                new_X = np.append(new_X, X[j])
                new_elements = np.append(new_elements, np.reshape(elements[:, j], (elements.shape[0], 1)), axis=1)
                if j == 0:
                    new_elements = np.delete(new_elements, [0], axis=1)
                new_ps = np.append(new_ps, ps[j])
                new_density = np.append(new_density, density[j])
                new_s_tot = np.append(new_s_tot, s_tot[j])
            
        new_mu = compute_mu(new_elements, amass)

        if np.array_equal(new_mass, last_mass):
            break
        else:
            last_mass = new_mass
            mass = new_mass
            dm = new_dm
            X = new_X
            elements = new_elements
            ps = new_ps
            density = new_density
            s_tot = new_s_tot
            mu = new_mu

    return new_mass, new_dm, new_elements, new_ps, new_density, new_s_tot, new_mu


def compute_mu(elements, amass):
    mean_mu = []

    for i in range(elements.shape[1]):
        mean_mu.append(1.0 / sum(elements[:, i]*amass))

    return np.array(mean_mu)


def write_composition_profile(mass, elements, fname="merger_Ssorted_composition.dat"):
    clipped_elements = np.where(elements < 1e-20, 0, elements)
    with open(fname, "w") as f:
        f.write(f"{len(mass)} {len(elements)}\n")
        for i in np.flip(range(mass.shape[0])):  # Flip such that output file is ordered surface -> core
            f.write(f"{(mass[-1] - mass[i]) / mass[-1]} ")
            for j in range(len(elements)):
                f.write(f"{clipped_elements[j, i]} ")
            f.write('\n')


def write_entropy_profile(mass, entropy, fname="merger_Ssorted_entropy.dat"):
    '''
    mass = mass coordinates of shells in solar masses
    entropy = specific entropy in erg/g/K
    '''
    with open(fname, "w") as f:
        f.write(f"{len(mass)}\n")
        for i in np.flip(range(mass.shape[0])):  # Flip such that output file is ordered surface -> core
            f.write(f"{(mass[-1] - mass[i]) / mass[-1]} {entropy[i]}\n")
            

def write_t_rho_profile(mass, t, rho, fname="merger_Ssorted_entropy.dat"):
    with open(fname, "w") as f:
        f.write(f"{len(mass)}\n")
        for i in np.flip(range(mass.shape[0])):
            f.write(f"{(mass[-1] - mass[i]) / mass[-1]} {rho[i]} {t[i]}\n")


def write_density_eint_profile(mass, rho, eint, fname="merger_Ssorted_entropy.dat"):
    with open(fname, "w") as f:
        f.write(f"{len(mass)}\n")
        for i in np.flip(range(mass.shape[0])):
            f.write(f"{(mass[-1] - mass[i]) / mass[-1]} {rho[i]} {eint[i]}\n")


def write_merger(mass, dm, elements, chemical_species, ps, density, s_tot, mean_mu, fname="merger_Ssorted.txt"):
    # Clip the element array so that abundances below 10^-20 are set to zero
    elements[elements < 1e-20] = 0

    with open(fname, "w") as f:
        f.write(f"# Entropy Sorting merger, performed at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        f.write(f"# star_mass = {mass[-1]:.16f}\n")
        f.write(f"# num_shells = {len(mass)}\n")
        f.write(f"# chemical_species = {', '.join(str(i) for i in chemical_species)}\n")
        output_columns = np.append(np.array(["mass", "dm", "density", "entropy", "passive_scalar", "mean_mu"]), chemical_species)
        f.write(f"{'id':<8}")
        for name in output_columns: # Write column headers
            f.write(f"{name:<26}")
        f.write("\n")

        output_data = np.column_stack((mass, dm, density, s_tot, ps, mean_mu, np.column_stack(elements)))

        for i, row in enumerate(output_data):
            f.write(f"{i:<8}")
            for col in row:
                f.write(f"{col:<26.16e}")
            f.write("\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Stellar mergers by entropy sorting.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", nargs=1, action="store", type=str,
                        help="Input JSON file specifying merger parameters.",
                        metavar=("input_file"), required=True, default=["input.json"])
    args = parser.parse_args()
    parameters = read_input(args.i[0])

    primary_file = os.path.abspath(parameters["primary_star"])
    secondary_file = os.path.abspath(parameters["secondary_star"])
    chemical_species = np.array(parameters["chemical_species"])
    fill_missing_species = parameters["fill_missing_species"]
    rawmod = parameters["output_raw"]
    remesh = parameters["enable_remeshing"]
    n_remesh = parameters["remeshing_shells"]
    mixing = parameters["enable_mixing"]
    n_mix = parameters["mixing_shells"]
    relaxation_profiles = parameters["relaxation_profiles"]
    mass_loss_fraction = parameters["massloss_fraction"]
    output_folder = os.path.abspath(parameters["output_dir"])
    diagnostics = parameters["output_diagnostics"]

    print("\nPython Entropy Sorting Merger\n")
    print(f"Computation started at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Primary: {primary_file}")
    print(f"Secondary: {secondary_file}")
    print(f"Output unmixed/non-remeshed model: {rawmod}")
    print(f"Perform remeshing: {remesh}")
    if remesh:
        print(f"Number of remeshing shells: {n_remesh}")
    print(f"Perform mixing: {mixing}")
    if mixing:
        print(f"Number of mixing shells: {n_mix}")
    print(f"Chemical species: {', '.join(str(i) for i in chemical_species)}")
    print(f"Fill missing species with zero abundance: {fill_missing_species}")
    print(f"Fractional mass loss: {mass_loss_fraction}")
    print(f"Output relaxation profiles: {relaxation_profiles}")
    print(f"Output folder: {output_folder}\n")

    # Check if the output directory exists, if not, create it
    output_folder = os.path.abspath(parameters["output_dir"])

    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except FileNotFoundError as exc:
            print(exc)
            print("Could not find parent directory of output folder. Quit.")
            exit(1)

    model_a = stellarmodel.model(chemical_species)
    model_b = stellarmodel.model(chemical_species)

    # Check if input files are .txt or profile, then read the models.
    if primary_file[-5::] == ".data" or primary_file[-8::] == ".data.gz": # If the input files are (gzipped) MESA models, use MESA reader to load the stars
        model_a.read_mesa_profile(primary_file, fill_missing_species, load_entropy=True)
        model_b.read_mesa_profile(secondary_file, fill_missing_species, load_entropy=True)
    elif primary_file[-4::] == ".txt": # If the input models are in the simple column format, use our basic loading function
        model_a.read_basic(primary_file, fill_missing_species, load_entropy=True)
        model_b.read_basic(secondary_file, fill_missing_species, load_entropy=True)
    else:
        print("Unknown progenitor file format, quit.")
        exit(0)

    if mass_loss_fraction < 0.0: # If mass loss fraction is negative, use MMAMS prescription
        mass_loss_fraction = mass_loss(model_a.star_mass, model_b.star_mass, do_const=False)

    # Load quantities needed for merger
    dm = np.append(model_a.dm, model_b.dm)
    s_tot = np.append(model_a.entropy, model_b.entropy)

    T = np.append(model_a.temperature, model_b.temperature)
    Rho = np.append(model_a.density, model_b.density)

    amass = get_amass(chemical_species)

    comp_tot = np.zeros((len(chemical_species), int(model_a.n_shells+model_b.n_shells)))

    for i, species in enumerate(chemical_species):
        comp_tot[i, :] = np.append(model_a.elements[i],model_b.elements[i])

    # Define a passive scalar, 1 for shells from primary, 0 for secondary
    ps = np.append(np.full(model_a.n_shells, 1), np.full(model_b.n_shells, 0))

    # Merge stars using entropy sorting
    dm_res, s_res, t_res, rho_res, ele, ps_res = new_star(dm, s_tot, T, Rho, comp_tot, ps)
    m_res = np.cumsum(dm_res) # Convert sorted dm back into proper mass coordinate

    if rawmod:
        write_merger(m_res, dm_res, ele, chemical_species, ps_res, rho_res, s_res, compute_mu(ele, amass), os.path.join(output_folder, "Ssorted_merger_raw.txt"))
        if relaxation_profiles:
            write_composition_profile(m_res, ele, os.path.join(output_folder, "Ssorted_merger_composition_raw.dat"))
            write_entropy_profile(m_res, s_res, os.path.join(output_folder, "Ssorted_merger_entropy_raw.dat"))

    if remesh:
        print("Remeshing ...")
        mean_mu = compute_mu(ele, amass)
        m_resr, dmr, eler, psr, rho_resr, s_resr, mu_resr = mloss_remesh(dm_res, ele, ele[np.argwhere(chemical_species == "h1")[0][0]], mean_mu, ps_res, rho_res, s_res, n_remesh, mass_loss_fraction, amass)
        # Write remeshed model
        write_merger(m_resr, dmr, eler, chemical_species, psr, rho_resr, s_resr, mu_resr, os.path.join(output_folder, "Ssorted_merger_remeshed.txt"))
        print("Done!")
        if relaxation_profiles:
            write_composition_profile(m_resr, eler, os.path.join(output_folder, "Ssorted_merger_composition_r.dat"))
            write_entropy_profile(m_resr, s_resr, os.path.join(output_folder, "Ssorted_merger_entropy_r.dat"))
    

    if diagnostics:
        with open(os.path.join(output_folder, "merger_info.txt"), "w") as f:
            f.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            f.write(f"Primary: {primary_file}\n")
            f.write(f"Secondary: {secondary_file}\n")
            f.write(f"Write raw model: {rawmod}\n")
            f.write(f"Remeshing: {remesh}\n")
            if remesh:
                f.write(f"Number of shells after remeshing: {n_remesh}\n")
            f.write(f"Mixing: {mixing}\n")
            if mixing:
                f.write(f"Number of shells after mixing: {n_mix}\n")
            f.write(f"Chemical species: {', '.join(str(i) for i in chemical_species)}\n")
            f.write(f"Mass loss fraction: {mass_loss_fraction}\n")
            f.write(f"Output relaxation profiles: {relaxation_profiles}\n")
            f.write(f"Output folder: {output_folder}\n")

if __name__ == '__main__':
    main()
