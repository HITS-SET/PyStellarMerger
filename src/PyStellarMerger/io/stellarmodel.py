import numpy as np
import os
import pickle
import mesa_reader as mr
from datetime import datetime
from PyStellarMerger.data.units import *
import importlib.resources

class model:
    def __init__(self, nuclear_net, n_shells=0):
        """
        n_shells: Number of shells the model should have.
        nuclear_net: Python list of all chemical species to track in the model.
        Initialize a new stellar model object, which holds all the information about the star.
        """
        self.n_shells = n_shells
        self.star_mass = 0.0
        self.star_radius = 0.0
        self.star_age = 0.0
        self.twin_flag = False

        # Create empty arrays for the shell data
        self.id = np.zeros(self.n_shells, dtype=int)  # int so that n_shells can be used in loops
        self.dm = np.zeros(self.n_shells)
        self.mass = np.zeros(self.n_shells)
        self.radius = np.zeros(self.n_shells)
        self.density = np.zeros(self.n_shells)
        self.pressure = np.zeros(self.n_shells)
        self.temperature = np.zeros(self.n_shells)
        self.mean_mu = np.zeros(self.n_shells)
        self.buoyancy = np.zeros(self.n_shells)
        self.beta = np.zeros(self.n_shells)
        self.e_thermal = np.zeros(self.n_shells)
        self.passive_scalar = np.zeros(self.n_shells)
        self.element_names = np.array(nuclear_net)

        if "h1" not in self.element_names:
            raise Exception("Missing h1 from nuclear network.")

        with importlib.resources.files("PyStellarMerger.data").joinpath("isotopes.pkl").open("rb") as f:
            isotopes = pickle.load(f) # Read in all isotopes from pickle file
            unknown_elements = np.setdiff1d(self.element_names, list(isotopes.keys()))
            if len(unknown_elements) > 0:
                raise Exception(f"Error, unknown element{'s'[:len(unknown_elements)^1]} given: ", ", ".join(unknown_elements))
            self.am = np.array([(1+float(isotopes[iso][1]))/float(isotopes[iso][0]) for iso in self.element_names]) # Get number of particles (1+Z, nucleus+electrons) per atomic mass unit

        self.elements = np.zeros((len(self.element_names), self.n_shells)) # Create 2D array for mass fraction profiles of all elements

    def __eq__(self, other):
        """
        other: Stellar model to compare to.
        This function returns True if both stellar models are identical in terms of mass coordinates, temperature, density, and mass fraction profiles.
        """
        return (np.all(self.mass == other.mass) and np.all(self.density == other.density) and np.all(self.temperature == other.temperature) and np.all(self.elements == other.elements))
    
    def read_mesa_profile(self, filename, fill_missing_species=False):
        """
        filename: Path to file to be read.
        This function uses MESA reader to read a stellar model from a MESA profile.
        """
        mesa_model = mr.MesaData(filename)

        # Load header data
        self.n_shells = mesa_model.num_zones
        self.star_mass = mesa_model.star_mass
        self.star_radius = mesa_model.radius[0]
        self.star_age = 0 # Set by PyMMAMS in mmas.compute_extra()

        # Verify that all required columns are present
        needed_quantities = ["mass", "dm", "radius", "density", "pressure", "temperature", "mu"]
        needed_columns = np.append(needed_quantities, self.element_names)
        column_names = mesa_model.bulk_names

        # First check that all required quantities are present
        if not set(needed_quantities).issubset(column_names):
            missing_columns = [x for x in needed_columns if x not in column_names]
            raise Exception(f"Error, missing column{'s'[:len(missing_columns)^1]}: ", ", ".join(missing_columns))

        # Next check if all chemical species are present in the MESA profile. Set the missing abundances to zero if fill_missing_species is True
        if not set(self.element_names).issubset(column_names):
            if not fill_missing_species:
                missing_elements = [x for x in self.element_names if x not in column_names]
                raise Exception(f"Error, missing element{'s'[:len(missing_elements)^1]}: ", ", ".join(missing_elements))
            else:
                print(f"Warning, missing element{'s'[:len(self.element_names)^1]}: ", ", ".join([x for x in self.element_names if x not in column_names]), ". Setting abundances to zero.")
                for el in self.element_names:
                    if el not in column_names:
                        setattr(mesa_model, el, np.zeros(self.n_shells)) # Artificially set the abundances in the MESA profile object to zero.

        # Load the shell data from the profile, flip arrays because MESA orders from surface to center
        self.id = mesa_model.zone - 1  # because MESA starts counting shells at 1
        self.dm = np.flip(mesa_model.dm) / uMSUN 
        self.mass = np.flip(mesa_model.mass)
        self.radius = np.flip(mesa_model.radius)
        self.density = np.flip(mesa_model.Rho)
        self.pressure = np.flip(mesa_model.P)
        self.temperature = np.flip(mesa_model.T)
        self.mean_mu = np.flip(mesa_model.mu)
        self.buoyancy = np.zeros(self.n_shells) # Buoyancy is computed later by PyMMAMS itself
        self.beta = np.zeros(self.n_shells)
        self.e_thermal = np.zeros(self.n_shells)

        self.elements = np.zeros((len(self.element_names), self.n_shells))

        for i, element in enumerate(self.element_names):
            self.elements[i] = np.flip(getattr(mesa_model, element))

        return None
    
    def read_basic(self, filename, load_merger_product=False):
        """
        filename: Path to file to be read.
        This function reads from the "basic" file format specified in the example stellar model files.
        """
        comment_lines = 0
        with open(filename, "r") as f: # Find beginning of actual data 
            for i, line in enumerate(f):
                if line[0] != "#":
                    comment_lines = i
                    break

        column_names = np.loadtxt(filename, unpack=True, skiprows=comment_lines, max_rows=1, dtype=str)
        needed_quantities = ["mass", "dm", "radius", "density", "pressure", "temperature", "mean_mu"]
        needed_columns = np.append(needed_quantities, self.element_names)

        if set(needed_columns).issubset(column_names): # Check if all of the required quantities are available in the file
            for col in needed_quantities:
                idx = np.argwhere(column_names == col)[0] # Get position of the column in the input file for loading
                columndata = np.loadtxt(filename, unpack=True, skiprows=comment_lines+1, dtype=float, usecols=idx)
                setattr(self, col, columndata)

            self.n_shells = len(self.mass)
            self.star_mass = self.mass[-1]
            self.star_radius = self.radius[-1]
            self.id = np.arange(0, self.n_shells, 1)

            self.elements = np.zeros((len(self.element_names), self.n_shells))
            for i, el in enumerate(self.element_names): # Load mass fraction profiles for all elements
                idx = np.argwhere(column_names == el)[0] # Get position of the current element's column
                columndata = np.loadtxt(filename, unpack=True, skiprows=comment_lines+1, dtype=float, usecols=idx)
                self.elements[i] = columndata

        else:
            missing_columns = [x for x in needed_columns if x not in column_names]
            raise Exception(f"Error, missing column{'s'[:len(missing_columns)^1]}: ", ", ".join(missing_columns))
        
        if load_merger_product: # In case we are loading from a completed merger, also load the passive scalar profile
            idx = np.argwhere(column_names == "passive_scalar")[0]
            self.passive_scalar = np.loadtxt(filename, unpack=True, skiprows=comment_lines+1, dtype=float, usecols=idx)

        return None
    
    def write_basic(self, filename):
        """
        filename: File to write stellar model data to.
        This function writes the stellar model data to the file "filename".
        """
        # Clip the element array so that abundances below 10^-20 are set to zero
        self.elements[self.elements < 1e-20] = 0
        with open(filename, "w") as f:
            f.write(f"# {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
            f.write(f"# star_mass = {self.star_mass:.16f}\n")
            f.write(f"# num_shells = {self.n_shells}\n")
            f.write(f"# element_names = {', '.join(str(i) for i in self.element_names)}\n")

            output_columns = np.append(np.array(["mass", "dm", "radius", "density", "pressure", "e_thermal", "buoyancy", "temperature", "mean_mu", "passive_scalar"]), self.element_names)
            f.write(f"{'id':<8}")
            for name in output_columns: # Write column headers
                f.write(f"{name:<26}")
            f.write("\n")
            output_data = np.column_stack((self.mass, self.dm, self.radius, self.density, self.pressure, self.e_thermal, self.buoyancy, self.temperature, self.mean_mu, self.passive_scalar, np.column_stack(self.elements)))
            for i, row in enumerate(output_data):
                f.write(f"{i:<8}")
                for col in row:
                    f.write(f"{col:<26.16e}")
                f.write("\n")

        return None
    
    def write_composition_profile(self, filename="composition_relaxation.dat"):
        """
        filename: File to which to write composition relaxation profile.
        This function writes a composition profile of all elements supplied in the input JSON which can be used to import the merger remnant into MESA using its "relaxation" routine (Paxton+2018).
        """
        self.elements[self.elements < 1e-20] = 0
        with open(filename, "w") as f:
            f.write(f"{self.n_shells} {len(self.elements)}\n")
            for i in np.flip(range(self.mass.shape[0])):  # Flip so that file is ordered from surface to center
                f.write(f"{(self.mass[-1] - self.mass[i]) / self.mass[-1]} ")
                for j in range(len(self.elements)):
                    f.write(f"{self.elements[j, i]} ")
                f.write('\n')
        
        return None

    def write_entropy_profile(self, filename="entropy_relaxation.dat"):
        """
        filename: File to which to write entropy relaxation profile.
        This function writes a temperature and density profile which can be used to import the merger remnant into MESA using its "relaxation" routine (Paxton+2018).
        """
        with open(filename, "w") as f:
            f.write(f"{self.n_shells}\n")
            for i in np.flip(range(self.mass.shape[0])):
                f.write(f"{(self.mass[-1] - self.mass[i]) / self.mass[-1]} {self.density[i]} {self.temperature[i]}\n")

        return None