from PyStellarMerger.pymmams import PyMMAMS
from PyStellarMerger.io import stellarmodel
from PyStellarMerger.pymmams.mmas import mmas
from PyStellarMerger.calc.computeenergy import compute_stellar_energy
from PyStellarMerger.calc.massloss import mass_loss
from PyStellarMerger.remeshing.mixing import mixing_product
from PyStellarMerger.remeshing.remeshing import mix_separately
import os
import numpy as np
from collections import defaultdict

class StellarMerger:
    def __init__(self, parameters):
        self.parameters = parameters
        print(self.parameters)
        self.validateInput()

        # Load the progenitor stars
        self.model_a = stellarmodel.model(self.parameters["chemical_species"])
        self.model_b = stellarmodel.model(self.parameters["chemical_species"])

        if self.parameters["primary_star"][-5::] == ".data" or self.parameters["primary_star"][-8::] == ".data.gz": # If the input files are (gzipped) MESA models, use MESA reader to load the stars
            self.model_a.read_mesa_profile(self.parameters["primary_star"], self.parameters["fill_missing_species"])
            self.model_b.read_mesa_profile(self.parameters["secondary_star"], self.parameters["fill_missing_species"])
        elif self.parameters["primary_star"][-4::] == ".txt": # If the input models are in the simple column format, use our basic loading function
            self.model_a.read_basic(self.parameters["primary_star"])
            self.model_b.read_basic(self.parameters["secondary_star"])
        else:
            print("Unknown progenitor file format, quit.")
            exit(1)

        # Set passive scalar of primary and secondary progenitor
        self.model_a.passive_scalar = np.ones(self.model_a.n_shells)  # Primary: Passive scalar = 1
        self.model_b.passive_scalar = np.zeros(self.model_b.n_shells)  # Secondary: Passive scalar = 0



    def validateInput(self):
        """
        Checks validity of input parameters and sets defaults for missing parameters.

        Parameters:
            input (dict): Dictionary containing the input parameters.

        Returns:
            parameters (dict): Validated and completed merger parameters.
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

        merger_params = defaultdict(lambda: None, defaults)

        # Update parameters with provided input
        merger_params.update(self.parameters)

        # Validate some input parameters:
        if not isinstance(self.parameters["n_target_shells"], int) or self.parameters["n_target_shells"] < 0:
            raise ValueError("n_target_shells must be an integer > 0.")
        
        if not isinstance(self.parameters["mixing_shells"], int) or self.parameters["mixing_shells"] < 0:
            raise ValueError("mixing_shells must be an integer > 0.")
        
        if not isinstance(self.parameters["remeshing_shells"], int) or self.parameters["remeshing_shells"] < 0:
            raise ValueError("remeshing_shells must be an integer > 0.")

        if not (isinstance(self.parameters["f_mod"], float) or isinstance(self.parameters["f_mod"], int)) or self.parameters["f_mod"] < 0.0:
            raise ValueError("f_mod must be >= 0.0.")
        
        if not (isinstance(self.parameters["massloss_fraction"], float), isinstance(self.parameters["massloss_fraction"], int)) or self.parameters["massloss_fraction"] >= 1.0:
            raise ValueError("massloss_fraction must be < 1.0.")
        
        # Set the mass loss behavior
        if self.parameters["massloss_fraction"] < 0:
            self.parameters["mass_loss_flag"] = False # If massloss_fraction is negative, disable constant mass loss and use the inbuilt mass loss prescription
            self.parameters["mass_loss_fraction"] = 0.0 # Has no effect, as inbuilt prescription will be used
        else:
            self.parameters["mass_loss_flag"] = True
            self.parameters["mass_loss_fraction"] = float(self.parameters["massloss_fraction"])

        return None

    def PyMMAMS(self):
        merger = mmas(self.model_a, self.model_b)
        self.model_remnant = merger.merge_stars_consistently(self.parameters["n_target_shells"], self.parameters["enable_shock_heating"], self.parameters["mass_loss_flag"], self.parameters["mass_loss_fraction"], self.parameters["output_dir"], self.parameters["f_mod"], self.parameters["extrapolate_shock_heating"], self.parameters["initial_buoyancy"], self.parameters["final_buoyancy"])

        # Compute the total energy of the progenitors and the remnant
        energy_a = compute_stellar_energy(self.model_a)
        energy_b = compute_stellar_energy(self.model_b)
        energy_p = compute_stellar_energy(self.model_remnant)
        print(f"energy_a = {energy_a}")
        print(f"energy_b = {energy_b}")
        print(f"energy_a + energy_b = {energy_a+energy_b}, energy_p = {energy_p}")

        de = (energy_p - (energy_a + energy_b)) / (energy_a + energy_b)
        try:
            de *= 1.0 / (mass_loss(self.model_a.star_mass, self.model_b.star_mass, self.parameters["mass_loss_flag"], self.parameters["mass_loss_fraction"]) / 100.0)
            print(f"de = {de}")
        except ZeroDivisionError: 
            print(f"Zero mass loss assumed, can't compute de.")

        # Write unmixed product to file
        self.model_remnant.write_basic(os.path.join(self.parameters["output_dir"], "merged_unmixed.txt"))

        if self.parameters["enable_mixing"]:  
            mixed_p = mixing_product(self.parameters["mixing_shells"], self.model_remnant)
            print("Writing mixed product to file.")
            mixed_p.write_basic(os.path.join(self.parameters["output_dir"], "merged_mixed.txt"))
            if self.parameters["relaxation_profiles"]:
                print("Writing entropy profile...")
                mixed_p.write_entropy_profile(os.path.join(self.parameters["output_dir"], "merger_mixed_entropy.dat"))
                print("Writing composition profile...")
                mixed_p.write_composition_profile(os.path.join(self.parameters["output_dir"], "merger_mixed_composition.dat"))

        # Apply custom remeshing scheme (Heller+2025) and write to file
        if self.parameters["enable_remeshing"]:
            remeshed_p = mix_separately(self.model_remnant, self.parameters["remeshing_shells"])
            print("Writing remeshed product to file.")
            remeshed_p.write_basic(os.path.join(self.parameters["output_dir"], "merged_remeshed.txt"))
            if self.parameters["relaxation_profiles"]:
                print("Writing entropy profile...")
                remeshed_p.write_entropy_profile(os.path.join(self.parameters["output_dir"], "merger_remeshed_entropy.dat"))
                print("Writing composition profile...")
                remeshed_p.write_composition_profile(os.path.join(self.parameters["output_dir"], "merger_remeshed_composition.dat"))

        print("PyMMAMS merger done.")

