from PyStellarMerger.io import stellarmodel
from PyStellarMerger.calc.massloss import mass_loss
from PyStellarMerger.calc.computeenergy import compute_stellar_energy
import numpy as np
from PyStellarMerger.calc.eos import compute_entropy, compute_temperature, compute_energy, compute_mu
from PyStellarMerger.data.units import *
from PyStellarMerger.pymmams.mergestars import merge_stars
from copy import deepcopy
import sys
import scipy
import os


class merge_stars_consistently_params:
    def __init__(self, e_tot, n_shells, m_ejecta, mass_loss_flag, mass_loss_fraction, sh_extrapolation):
        self.e_tot = e_tot
        self.n_shells = n_shells
        self.m_ejecta = m_ejecta
        self.mass_loss_flag = mass_loss_flag
        self.mass_loss_fraction = mass_loss_fraction
        self.sh_extrapolation = sh_extrapolation


class mmas:
    def __init__(self, m_a, m_b, v0=0, r0=0):
        self.r_p = r0
        self.v_inf = v0
        self.model_a = m_a
        self.model_b = m_b
        self.product = stellarmodel.model(m_a.element_names)
        self.mixed_product = stellarmodel.model(m_a.element_names)

    def compute_extra(self):
        """
        This function computes the mean molecular weight, temperature, buoyancy, thermal energy and density for each shell in both progenitors according to the MMAMS equation of state.
        """
        mass = 0.0
        radius = 0.0

        for i in range(self.model_a.n_shells):
            mean_mu = compute_mu(self.model_a.elements[:, i], self.model_a.am)

            self.model_a.mean_mu[i] = mean_mu
            self.model_a.temperature[i] = compute_temperature(self.model_a.density[i], self.model_a.pressure[i], self.model_a.mean_mu[i])
            self.model_a.buoyancy[i] = compute_entropy(self.model_a.density[i], self.model_a.temperature[i], self.model_a.mean_mu[i])
            self.model_a.e_thermal[i] = compute_energy(self.model_a.density[i], self.model_a.temperature[i], self.model_a.mean_mu[i])
            Pgas = self.model_a.density[i] / (self.model_a.mean_mu[i] * uM_U) * uK * self.model_a.temperature[i]
            self.model_a.beta[i] = Pgas / self.model_a.pressure[i]

            mass = max(mass, self.model_a.mass[i])
            radius = max(radius, self.model_a.radius[i])

        self.model_a.star_mass = mass
        self.model_a.star_radius = radius

        model_a_H1 = self.model_a.elements[np.where(self.model_a.element_names == "h1")[0][0]]

        if model_a_H1[0] < 0.1:
            self.model_a.star_age = 1.0
        else:
            self.model_a.star_age = 0.0

        mass = 0.0
        radius = 0.0

        for i in range(self.model_b.n_shells):
            mean_mu = compute_mu(self.model_b.elements[:, i], self.model_b.am)

            self.model_b.mean_mu[i] = mean_mu
            self.model_b.temperature[i] = compute_temperature(self.model_b.density[i], self.model_b.pressure[i], self.model_b.mean_mu[i])
            self.model_b.buoyancy[i] = compute_entropy(self.model_b.density[i], self.model_b.temperature[i], self.model_b.mean_mu[i])
            self.model_b.e_thermal[i] = compute_energy(self.model_b.density[i], self.model_b.temperature[i], self.model_b.mean_mu[i])
            Pgas = self.model_b.density[i] / (self.model_b.mean_mu[i] * uM_U) * uK * self.model_b.temperature[i]
            self.model_b.beta[i] = Pgas / self.model_b.pressure[i]

            mass = max(mass, self.model_b.mass[i])
            radius = max(radius, self.model_b.radius[i])

        self.model_b.star_mass = mass
        self.model_b.star_radius = radius

        model_b_H1 = self.model_b.elements[np.where(self.model_b.element_names == "h1")[0][0]]

        if model_b_H1[0] < 0.1:
            self.model_b.star_age = 1.0
        else:
            self.model_b.star_age = 0.0

        temp_model = deepcopy(self.model_a)

        if self.model_a.star_mass < self.model_b.star_mass:
            self.model_a = self.model_b
            self.model_b = temp_model

        print(f"model_a.star_age = {self.model_a.star_age}")
        print(f"model_b.star_age = {self.model_b.star_age}")

    def shock_heating_func(self, model, a, b):
        """
        model: Progenitor model to apply shock heating to.
        a: Shock heating parameter a, obtained by interpolating on the calibration by Gaburov+2008 in the function shock_heating.
        b: Shock heating parameter b, obtained in the same way as a.
        This function applies shock heating to a progenitor model.
        """
        density_max = 0
        pressure_max = 0
        entropy_min = 1e300

        for i in range(model.n_shells):
            model.buoyancy[i] = compute_entropy(model.density[i], model.temperature[i], model.mean_mu[i])
            density_max = max(density_max, model.density[i])
            pressure_max = max(pressure_max, model.pressure[i])
            entropy_min = min(entropy_min, model.buoyancy[i])

        for i in range(model.n_shells):
            model.buoyancy[i] = compute_entropy(model.density[i], model.temperature[i], model.mean_mu[i])
            de = a + b * np.log10(model.pressure[i] / pressure_max)
            if de > 100:
                de = 100
            model.buoyancy[i] += entropy_min * 10.0 ** de

        return 0

    def shock_heating(self, ff, sh_extrapolation):
        """
        ff: Shock heating factor f_heat that ensures energy conservation.
        sh_extrapolation: Flag to enable or disable extrapolation of shock heating parameters.
        This function finds the shock heating parameters a and b by considering the primary progenitor's evolutionary stage and the binary system's mass ratio q.
        """
        tams_dm = np.array([8.3, 8.9, 5.0, 1.9])
        tams_q = np.array([0.8, 0.4, 0.2, 0.1])
        tams_params = np.array([[-0.12, -0.63, -1.025, -0.77],
                                [0.25, -0.37, -1.32, -0.95],
                                [0.35, -0.22, -0.84, -0.77],
                                [0.22, -0.17, -0.59, -0.80]])

        hams_dm = np.array([6.8, 4.7, 2.1, 0.8])
        hams_q = np.array([0.8, 0.4, 0.2, 0.1])
        hams_params = np.array([[-0.48, -0.79, -0.89, -0.85],
                                [-0.22, -0.66, -0.88, -0.79],
                                [-0.09, -0.44, -0.77, -0.85],
                                [-0.13, -0.28, -0.80, -1.03]])

        if self.model_a.star_age == 0:
            dm_cur = hams_dm
            q_cur = hams_q
            fit_cur = hams_params
        else:
            dm_cur = tams_dm
            q_cur = tams_q
            fit_cur = tams_params

        q = self.model_b.star_mass / self.model_a.star_mass
        if q > 1:
            print('Error, q = ', q, ' > 1, should be <= 1.')
            sys.exit(0)

        q = min(max(q, 0.05), 1.0)

        n = 4

        xq = np.zeros(n)
        xa = np.zeros(n)
        xb = np.zeros(n)
        xc = np.zeros(n)
        xd = np.zeros(n)

        if self.model_a.twin_flag and self.model_b.twin_flag:
            for j in range(n):
                i = n - 1 - j
                xq[j] = q_cur[i]
                xa[j] = fit_cur[i, 0]
                xb[j] = fit_cur[i, 1]
                xc[j] = fit_cur[i, 0]  # Use the shock heating parameters from the SPH-primary for both progenitors if they are identical
                xd[j] = fit_cur[i, 1]
        else:
            for j in range(n):
                i = n - 1 - j
                xq[j] = q_cur[i]
                xa[j] = fit_cur[i, 0]
                xb[j] = fit_cur[i, 1]
                xc[j] = fit_cur[i, 2]
                xd[j] = fit_cur[i, 3]

        # Get a, b, c and d using linear interpolation. User input determines if values below 0.1 or above 0.8 are being extrapolated or taken equal to closest boundary.
        if sh_extrapolation:
            if q > 0.8 or q < 0.1:
                print("Mass ratio outside of range 0.1 - 0.8. Extrapolating shock heating parameters.")

            fa = scipy.interpolate.interp1d(xq, xa, fill_value="extrapolate") 
            fb = scipy.interpolate.interp1d(xq, xb, fill_value="extrapolate")
            fc = scipy.interpolate.interp1d(xq, xc, fill_value="extrapolate")
            fd = scipy.interpolate.interp1d(xq, xd, fill_value="extrapolate")
            
            a = fa(q)
            b = fb(q)
            c = fc(q)
            d = fd(q)

        else:
            if q > 0.8 or q < 0.1:
                print("Mass ratio outside of range 0.1 - 0.8. Taking same parameters as closest outer boundary.")

            a = np.interp(q, xq, xa)
            b = np.interp(q, xq, xb)
            c = np.interp(q, xq, xc)
            d = np.interp(q, xq, xd)

        a += np.log10(ff)
        c += np.log10(ff)

        # Apply shock heating to progenitors
        a_success = self.shock_heating_func(self.model_a, a, b)
        b_success = self.shock_heating_func(self.model_b, c, d)

        if a_success != 0:
            print('Shock heating for model_a was unsuccessful!')
            return -1
        if b_success != 0:
            print('Shock heating for model_b was unsuccessful!')
            return -1

        return 0

    def merge_stars_consistently_eq(self, f_heat, params):
        """
        f_heat: Shock heating factor that ensures energy conservation.
        params: Parameters for the merger product.
        This function applies shock heating to the progenitors for a given f_heat and then performs the merger. The difference in total energy before and after the merger is then returned.
        """
        status = self.shock_heating(f_heat, params.sh_extrapolation)
        if status < 0:
            return -1e-99
        merger_product = merge_stars(1.0, params.n_shells, self.model_a, self.model_b, params.mass_loss_flag, params.mass_loss_fraction)

        energy_p = compute_stellar_energy(merger_product) # Gravitational potential energy + thermal energy
        vesc = np.sqrt(merger_product.star_mass / merger_product.star_radius) # Escape velocity of the ejecta
        energy_ej = params.m_ejecta * vesc ** 2 # Kinetic energy carried away by the ejecta

        print(f"energy_p = {energy_p}")
        print(f"params.e_tot = {params.e_tot}")
        print(f"energy_ej = {energy_ej}")
        print("-~~*~~-")

        return energy_p + energy_ej - params.e_tot

    def merge_stars_consistently(self, n_shells, flag_do_shock_heating, mass_loss_flag, mass_loss_fraction, output_folder, f_heat_factor, sh_extrapolation, initial_buoyancy, final_buoyancy, **kwargs):
        """
        n_shells: Number of shells the unmixed merger product should roughly have.
        flag_do_shock_heating: Flag to enable or disable shock heating.
        mass_loss_flag: Flag to enable or disable constant mass loss.
        mass_loss_fraction: If mass_loss_flag is set to False, use this fraction of the total mass as mass loss.
        output_folder: Folder to save the buoyancy profiles before and after shock heating to.
        f_heat_factor: Factor to scale the shock heating by.
        sh_extrapolation: Flag to enable or disable extrapolation of shock heating parameters.
        initial_buoyancy: Flag to save the progenitors' buoyancy profiles before shock heating.
        final_buoyancy: Flag to save the progenitors' buoyancy profiles after shock heating.
        This function iteratively finds the shock heating factor f_heat_root that ensures energy conservation in the merger product. 
        This step is skipped if flag_do_shock_heating is set to False. 
        The shock heating modification factor (if different from unity) is applied after f_heat_root has been found. Note that energy conservation is not guaranteed in this case.
        """
        self.compute_extra()

        # Save progenitors with initial buoyancy profiles
        if initial_buoyancy:
            self.model_a.write_basic(os.path.join(output_folder, "primary_initial_buoyancy.txt"))
            self.model_b.write_basic(os.path.join(output_folder, "secondary_initial_buoyancy.txt"))

        f_lost = mass_loss(self.model_a.star_mass, self.model_b.star_mass, mass_loss_flag, mass_loss_fraction) / 100.0
        m_ejecta = (self.model_a.star_mass + self.model_b.star_mass) * f_lost

        energy_a = compute_stellar_energy(self.model_a)
        energy_b = compute_stellar_energy(self.model_b)

        e_tot = energy_a + energy_b
        p = merge_stars_consistently_params(e_tot, 1000, m_ejecta, mass_loss_flag, mass_loss_fraction, sh_extrapolation)

        if flag_do_shock_heating and f_heat_factor != 0: # If shock heating is enabled, try to find f_heat_root and shock heat progenitors before merger
            f_heat_min = 0.1
            f_heat_max = 10.0

            # Calculate lower bound test interval for shock heating
            print("----------------------------------------------------")
            print("Computing f_heat using brent method:")
            f1 = self.merge_stars_consistently_eq(f_heat_min, p)
            if f1 == -1e99:
                f_heat_min *= 2.0

            while f1 == -1e99 and f_heat_min < 1.0:
                f1 = self.merge_stars_consistently_eq(f_heat_min, p)
                if f1 == -1e99:
                    f_heat_min *= 2.0

            # Calculate upper bound test interval for shock heating
            f2 = self.merge_stars_consistently_eq(f_heat_max, p)
            if f2 == -1e99:
                f_heat_max /= 2.0

            while f2 == -1e99 and f_heat_max > 1.0:
                f2 = self.merge_stars_consistently_eq(f_heat_max, p)
                if f2 == -1e99:
                    f_heat_max /= 2.0

            #print(f"f1 = {f1}")
            #print(f"f2 = {f2}")

            if f1 == -1e99 or f2 == -1e99: # No shock heating
                f2 = f1

            if f1 * f2 < 0 and flag_do_shock_heating: # Find roots of merge_stars_consistently_eq to obtain f_heat
                print("\n\n\n  Solving for f_heat ...  \n\n\n")
                iter_max = 1000
                
                f_heat_root, f_heat_root_info = scipy.optimize.brentq(self.merge_stars_consistently_eq, f_heat_min, f_heat_max, args=(p), full_output=True, maxiter=iter_max, rtol=1e-3)
                
                print(f"Converged in {f_heat_root_info.iterations} iterations.")

                print("Solved energy consistency.")
                print(f"f_heat = {f_heat_root}")
                print("Now build high-resolution model.\n")

                if f_heat_factor != 1.0:
                    print(f"Modified shock heating by a factor of {f_heat_factor:.2f}.")
                    f_heat_mod = f_heat_root * f_heat_factor # Apply our scaling factor to the shock heating factor
                else:
                    f_heat_mod = f_heat_root

                shock_heating_status = self.shock_heating(f_heat_mod, sh_extrapolation) # Perform shock heating with the modified f_heat

                if final_buoyancy:
                    self.model_a.write_basic(os.path.join(output_folder, "primary_shock_heated.txt"))
                    self.model_b.write_basic(os.path.join(output_folder, "secondary_shock_heated.txt"))

            else:
                if flag_do_shock_heating:
                    print("\n\n\n  Failed to solve for f_heat ... Assuming no shock-heating.  \n\n\n")
                else:
                    print("\n\n\n  Assuming no shock-heating.  \n\n\n")
            
            product = merge_stars(1.0, n_shells, self.model_a, self.model_b, mass_loss_flag, mass_loss_fraction)

        else: # If shock heating has been disabled manually, directly merge stars without trying to find f_heat_root
            product = merge_stars(1.0, n_shells, self.model_a, self.model_b, mass_loss_flag, mass_loss_fraction)

        # Compute shell dm from shell mass coordinates
        m_last = 0
        for i, mass in enumerate(product.mass):
            product.dm[i] = mass - m_last
            m_last = mass

        return product
