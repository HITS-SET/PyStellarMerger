from PyStellarMerger.io import stellarmodel
from PyStellarMerger.calc.eos import compute_energy, compute_pressure, compute_temperature, compute_entropy, compute_mu
import numpy as np
from copy import deepcopy


def mixing_product(n, model):
    """
    n: Number of shells the product should have after mixing.
    model: Stellar model to be mixed.
    This function applies the original MMAMS mixing scheme to a stellar model. 
    Multiple shells, starting in the center, are combined into one and their properties computed by considering conservation of the total mass of chemical species and applying the MMAMS equation of state.
    """
    unmixed_product = deepcopy(model)  # Copy the unmixed product as not to overwrite it
    mixed_product = stellarmodel.model(nuclear_net=unmixed_product.element_names)

    dm_bin = unmixed_product.star_mass / n

    m_prev = 0
    m_bin = 0

    Mtot = 0
    Utot = 0
    Vtot = 0
    r_mean = 0

    n_in_bin = 0
    j = 0

    elements_tot = np.zeros(len(unmixed_product.elements))
    passive_scalar_tot = 0

    for i in range(unmixed_product.n_shells):
        dm = unmixed_product.mass[i] - m_prev
        m_prev = unmixed_product.mass[i]

        r_mean += unmixed_product.radius[i]
        n_in_bin += 1

        m_bin += dm
        Mtot += dm
        Vtot += dm/unmixed_product.density[i]

        elements_tot += dm * unmixed_product.elements[:, i]
        passive_scalar_tot += dm * unmixed_product.passive_scalar[i]

        Utot += compute_energy(unmixed_product.density[i], unmixed_product.temperature[i], unmixed_product.mean_mu[i]) * dm

        if m_bin > dm_bin:
            mixed_product.id = np.append(mixed_product.id, j)
            mixed_product.dm = np.append(mixed_product.dm, m_bin)
            mixed_product.radius = np.append(mixed_product.radius, r_mean / n_in_bin)
            mixed_product.mass = np.append(mixed_product.mass, Mtot)
            mixed_product.density = np.append(mixed_product.density, m_bin / Vtot)

            elements_tot = np.reshape(elements_tot, (len(unmixed_product.elements), 1))  # Reshape so that the below append behaves as expected
            mixed_product.elements = np.append(mixed_product.elements, elements_tot / m_bin, axis=1)  # Appending the mixed abundances to the mixed product's elements array
            mixed_product.passive_scalar = np.append(mixed_product.passive_scalar, passive_scalar_tot / m_bin)
            mixed_product.mean_mu = np.append(mixed_product.mean_mu, compute_mu(mixed_product.elements[:, j], mixed_product.am))

            mixed_product.e_thermal = np.append(mixed_product.e_thermal, Utot / m_bin)
            mixed_product.pressure = np.append(mixed_product.pressure, compute_pressure(mixed_product.density[j], mixed_product.e_thermal[j], mixed_product.mean_mu[j]))
            mixed_product.temperature = np.append(mixed_product.temperature, compute_temperature(mixed_product.density[j], mixed_product.pressure[j], mixed_product.mean_mu[j]))
            mixed_product.buoyancy = np.append(mixed_product.buoyancy, compute_entropy(mixed_product.density[j], mixed_product.temperature[j], mixed_product.mean_mu[j]))

            m_bin -= dm_bin

            j += 1

            m_bin = 0
            Utot = 0
            Vtot = 0
            r_mean = 0

            elements_tot = np.zeros(len(unmixed_product.elements))
            passive_scalar_tot = 0

            n_in_bin = 0

    mixed_product.star_mass = mixed_product.mass[-1]
    mixed_product.star_radius = mixed_product.radius[-1]
    mixed_product.star_age = 0  
    mixed_product.n_shells = j

    return mixed_product
