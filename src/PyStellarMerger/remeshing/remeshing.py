import argparse
import numpy as np
from copy import deepcopy
from PyStellarMerger.io import stellarmodel
from PyStellarMerger.calc import eos
import json

def find_dv_regions(mass, mu, dmu_lim=1e-2):
    """
    Find the ids of the shells at which the mean molecular weight becomes double-valued.
    Capable of finding large steps in mu such that these aren't smoothed out when remeshing.
    dmu_lim is the limit for the standard deviation of mu between two shells at which a double-valued region/step is
    declared. Smaller values mean that even small fluctuations of mu will be interpreted as being part of a double-valued
    region (so these regions become bigger), while larger values allow for larger fluctuations, decreasing the size of
    the double-valued regions. The value given has been determined empirically by looking at different unmixed mergers
    and judging 'by eye' where the region-boundaries should be.
    """
    lower = np.array([0], dtype=int)
    upper = np.array([], dtype=int)

    dmu = np.zeros(len(mu))

    for i, m in enumerate(mu):
        if i == len(mu)-1:
            dmu[i] = dmu[i-1] # For the last shell, insert the value of the second-to-last shell to avoid an index error
            break
        dmu[i] = mu[i+1] - mu[i] # Calculate the difference in mean molecular weight between adjacent shells

    dmu = abs(dmu)/np.amax(abs(dmu)) # Normalize such that maximum dmu = 1

    search = True
    end = False
    lo = 0
    up = 0

    while search:
        for i in np.arange(up, mu.shape[0], 1):
            if dmu[i] > dmu_lim:
                lo = i
                break
            if i == mu.shape[0]-1:
                end = True

        if end:
            break

        for j in range(lo, mu.shape[0], 1):
            d_dm_idx = np.argmin(abs(mass[j:-1] - mass[j] - mass[-1]/40)) + j
            if np.all(dmu[j:d_dm_idx] < dmu_lim/2): # Smaller tolerance in dmu for finding end of double-valued region
                up = j
                # end = True
                break
            if j == mu.shape[0] - 1:
                upper = np.append(upper, mu.shape[0])
                search = False

        # Check if the zone that was found is either a double-valued region or a step
        if mass[up] - mass[lo] > 0.2 or abs(mu[up] - mu[lo]) > 0.01:
            lower = np.append(lower, np.array([lo, up]))
            upper = np.append(upper, np.array([lo, up]))

    if len(upper) > 0:
        if upper[-1] != len(mass) - 1:
            upper = np.append(upper, len(mass))
    else:
        upper = np.append(upper, len(mass))

    return lower, upper


def find_dv_regions_std(mass, X, Xstd_lim=1e-4):
    """
    First version, using standard deviation of hydrogen abundance between shells. Replaced with better version.
    Find the ids of the shells at which the hydrogen abundance becomes double-valued.
    Capable of finding large steps in X such that these aren't smoothed out when remeshing.
    Xstd_lim is the limit for the standard deviation of X between two shells at which a double-valued region/step is
    declared. Smaller values mean that even small fluctuations of X will be interpreted as being part of a double-valued
    region (so these regions become bigger), while larger values allow for larger fluctuations, decreasing the size of
    the double-valued regions. The value given has been determined empirically by looking at different unmixed mergers
    and judging 'by eye' where the region-boundaries should be.
    """
    lower = np.array([0], dtype=int)
    upper = np.array([], dtype=int)

    Xstd = np.zeros(X.shape[0])
    #Xstd_lim = 1e-4

    for i in np.arange(1, X.shape[0] - 1, 1):
        Xstd[i] = np.std(X[i - 1:i + 1])

    search = True
    end = False
    lo = 0
    up = 0

    while search:
        for i in np.arange(up, X.shape[0], 1):
            if Xstd[i] > Xstd_lim:
                lo = i
                # print(lo)
                # end = True
                break
            if i == X.shape[0]-1:
                end = True

        if end:
            break

        for j in range(lo, X.shape[0], 1):
            d_dm_idx = np.argmin(abs(mass[j:-1] - mass[j] - 0.3)) + j
            if np.all(Xstd[j:d_dm_idx] < Xstd_lim):
                up = j
                # end = True
                break
            if j == X.shape[0] - 1:
                upper = np.append(upper, X.shape[0])
                search = False

        # Check if the zone that was found is either a double-valued region or a step
        if mass[up] - mass[lo] > 0.2 or abs(X[up] - X[lo]) > 0.005:
            lower = np.append(lower, np.array([lo, up]))
            upper = np.append(upper, np.array([lo, up]))

    if len(upper) > 0:
        if upper[-1] != len(mass) - 1:
            upper = np.append(upper, len(mass))
    else:
        upper = np.append(upper, len(mass))

    return lower, upper


def remesh_func(mass, dm, X, elements, ps, radius, density, temperature, mu, am, dm_lim):
    last_mass = mass
    # dm_lim = 0.1

    while True:
        dmu = np.zeros(mu.shape[0])
        comb_id = np.array([], dtype=int)

        skip = False
        skip_twice = False
        for i in np.arange(1, len(mu)-1, 1):
            dmu[i] = abs(mu[i] - mu[i-1]) # Calculate the difference in mean molecular weight between adjacent shells

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
        new_elements = np.zeros((len(am), 0))
        new_ps = np.array([])
        new_radius = np.array([])
        new_density = np.array([])
        new_temperature = np.array([])
        new_mu = np.array([])
        new_pressure = np.array([])
        new_e_thermal = np.array([])
        new_entropy = np.array([])

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
                new_elements = np.append(new_elements, np.reshape((dm[j]*elements[:, j]+dm[j+1]*elements[:, j+1])/(dm[j]+dm[j+1]), (len(am), 1)), axis=1)
                new_ps = np.append(new_ps, (dm[j]*ps[j]+dm[j+1]*ps[j+1])/(dm[j]+dm[j+1]))
                new_radius = np.append(new_radius, (radius[j] + radius[j+1])/2)
                new_density = np.append(new_density, (dm[j]+dm[j+1])/((dm[j]/density[j])+(dm[j+1]/density[j+1])))
                new_mu = np.append(new_mu, eos.compute_mu(new_elements[:, -1], am))
                new_e_thermal = np.append(new_e_thermal, (eos.compute_energy(density[j], temperature[j], mu[j])*dm[j] + eos.compute_energy(density[j+1], temperature[j+1], mu[j+1])*dm[j+1])/(dm[j]+dm[j+1]))
                new_pressure = np.append(new_pressure, eos.compute_pressure(new_density[-1], new_e_thermal[-1], new_mu[-1]))
                new_temperature = np.append(new_temperature, eos.compute_temperature(new_density[-1], new_pressure[-1], new_mu[-1]))
                new_entropy = np.append(new_entropy, eos.compute_entropy(new_density[-1], new_temperature[-1], new_mu[-1]))
                skip = True
            else:
                new_mass = np.append(new_mass, mass[j])
                new_dm = np.append(new_dm, dm[j])
                new_X = np.append(new_X, X[j])
                new_elements = np.append(new_elements, np.reshape(elements[:, j], (len(am), 1)), axis=1)
                new_ps = np.append(new_ps, ps[j])
                new_radius = np.append(new_radius, radius[j])
                new_density = np.append(new_density, density[j])
                new_mu = np.append(new_mu, eos.compute_mu(elements[:, j], am))
                new_e_thermal = np.append(new_e_thermal, eos.compute_energy(density[j], temperature[j], mu[j]))
                new_pressure = np.append(new_pressure, eos.compute_pressure(density[j], new_e_thermal[-1], mu[j]))
                new_temperature = np.append(new_temperature, eos.compute_temperature(density[j], new_pressure[-1], mu[j]))
                new_entropy = np.append(new_entropy, eos.compute_entropy(density[j], temperature[j], mu[j]))

        if np.array_equal(new_mass, last_mass):
            break
        else:
            last_mass = new_mass
            mass = new_mass
            dm = new_dm
            X = new_X
            elements = new_elements
            ps = new_ps
            radius = new_radius
            density = new_density
            temperature = new_temperature
            mu = new_mu

    return new_mass, new_dm, new_elements, new_ps, new_radius, new_density, new_temperature, new_mu, new_pressure, new_e_thermal, new_entropy


def mix_separately(model, remeshing_shells):
    dv_product = deepcopy(model)
    remeshed_product = stellarmodel.model(nuclear_net=dv_product.element_names)

    X = dv_product.elements[np.where(dv_product.element_names == "h1")[0][0]]

    lower, upper = find_dv_regions(dv_product.mass, X)

    m_prev = 0
    for j in range(dv_product.n_shells):
        dv_product.dm[j] = dv_product.mass[j] - m_prev
        m_prev = dv_product.mass[j]

    for i, (lo, up) in enumerate(zip(lower, upper)):
        tempmodel = remesh_func(dv_product.mass[lo:up], dv_product.dm[lo:up], X[lo:up], dv_product.elements[:, lo:up],
                                dv_product.passive_scalar[lo:up],
                                dv_product.radius[lo:up], dv_product.density[lo:up],
                                dv_product.temperature[lo:up], dv_product.mean_mu[lo:up],
                                dv_product.am, dv_product.mass[-1]/remeshing_shells)

        remeshed_product.mass = np.append(remeshed_product.mass, tempmodel[0])
        remeshed_product.dm = np.append(remeshed_product.dm, tempmodel[1])
        remeshed_product.elements = np.append(remeshed_product.elements, tempmodel[2], axis=1)
        remeshed_product.passive_scalar = np.append(remeshed_product.passive_scalar, tempmodel[3])
        remeshed_product.radius = np.append(remeshed_product.radius, tempmodel[4])
        remeshed_product.density = np.append(remeshed_product.density, tempmodel[5])
        remeshed_product.temperature = np.append(remeshed_product.temperature, tempmodel[6])
        remeshed_product.mean_mu = np.append(remeshed_product.mean_mu, tempmodel[7])
        remeshed_product.pressure = np.append(remeshed_product.pressure, tempmodel[8])
        remeshed_product.e_thermal = np.append(remeshed_product.e_thermal, tempmodel[9])
        remeshed_product.buoyancy = np.append(remeshed_product.buoyancy, tempmodel[10])

    remeshed_product.n_shells = len(remeshed_product.mass)
    remeshed_product.star_mass = remeshed_product.mass[-1]
    remeshed_product.star_radius = remeshed_product.radius[-1]
    remeshed_product.id = np.arange(0, remeshed_product.n_shells, 1)

    return remeshed_product

def manual_remesh(raw_model, remeshing_shells):
    """
    This function is useful for manually remeshing a model, for example when looking to repeat the remeshing for a different number of shells.
    The script looks for a "merged_unmixed.txt" file in the current working directory, and remeshes it with the number of shells given in the argument.
    """
    print(f"Remeshing model with {remeshing_shells} shells.")
    remeshed_model = mix_separately(raw_model, remeshing_shells)
    print("Successfully remeshed model.")
    return remeshed_model



if __name__ == "__main__":
    """
    When calling remeshing.py directly from the command line, the manual_remesh function is called. 
    This snippet expects a "input.json" file to be present for the nuclear network, and a "merger_unmixed.txt" stellar model file for the actual unmixed merger model.
    """
    parser = argparse.ArgumentParser(description="Remeshing module of Make Me A [Massive] Star in Python",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", nargs=1, action="store", type=int,
                        help="The number of shells the product should roughly have after remeshing.",
                        metavar=("remeshing_shells"), required=True, default=[110])
    args = parser.parse_args()
    remeshing_shells = args.n[0]

    with open("input.json", "r") as f:
        parameters = json.load(f)

    raw_model = stellarmodel.model(nuclear_net=np.array(parameters["chemical_species"]))
    raw_model.read_basic("merged_unmixed.txt", load_merger_product=True)
    remeshed_model = manual_remesh(raw_model, remeshing_shells)

    print("Writing remeshed model and relaxation profiles to file ...")
    remeshed_model.write_basic("merged_remeshed.txt")
    remeshed_model.write_composition_profile("merger_Ssorted_composition_remeshed.dat")
    remeshed_model.write_entropy_profile("merger_Ssorted_entropy_remeshed.dat")
    print("Done!")
