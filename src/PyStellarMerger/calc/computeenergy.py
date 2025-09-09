from PyStellarMerger.data.units import *
import numpy as np
from scipy.integrate import quad


def integrand(mass, arr_mass, arr_radius, arr_temp, arr_dens, arr_m_mu):
    radius = np.interp(mass, arr_mass, arr_radius)
    temp = np.interp(mass, arr_mass, arr_temp)
    dens = np.interp(mass, arr_mass, arr_dens)
    m_mu = np.interp(mass, arr_mass, arr_m_mu)

    f = -mass/radius
    de_therm = 1.5*uK*temp/m_mu/uM_U + uA_RAD*temp**4/dens
    de_therm *= 1.0/(uG*uMSUN/uRSUN)

    #print("----------------------------------")
    #print(f"radius = {radius}")
    #print(f"temp = {temp}")
    #print(f"dens = {dens}")
    #print(f"m_mu = {m_mu}")
    #print(f"f = {f}")
    #print(f"f + de_therm = {f + de_therm}")
    return f + de_therm


def compute_stellar_energy(model):
    """
    model = Input stellar model.
    Computes the total energy (gravitational + thermal) in the star through numerical integration.
    """
    n_shells = model.n_shells

    arr_mass = np.zeros(n_shells+1)
    arr_radius = np.zeros(n_shells+1)
    arr_temp = np.zeros(n_shells+1)
    arr_dens = np.zeros(n_shells+1)
    arr_m_mu = np.zeros(n_shells+1)
    #print(arr_temp)

    for i in range(n_shells):
        if i > 0 and model.mass[i] <= arr_mass[i-1]:
            n_shells = i
            break

        arr_mass[i + 1] = model.mass[i]
        arr_radius[i + 1] = model.radius[i]
        arr_temp[i + 1] = model.temperature[i]
        arr_dens[i + 1] = model.density[i]
        arr_m_mu[i + 1] = model.mean_mu[i]

    arr_mass[0] = 0.0
    arr_radius[0] = 0.0
    arr_temp[0] = arr_temp[1]
    arr_dens[0] = arr_dens[1]
    arr_m_mu[0] = arr_m_mu[1]

    # In case the if-condition in the for loop is triggered, the array's sizes must be reduced to the new n_shells to emulate the behaviour of the original code. This is done by selecting the first n_shells elements.
    arr_mass = arr_mass[0:n_shells+1]
    arr_radius = arr_radius[0:n_shells+1]
    arr_temp = arr_temp[0:n_shells+1]
    arr_dens = arr_dens[0:n_shells+1]
    arr_m_mu = arr_m_mu[0:n_shells+1]

    #print(arr_temp)

    m_0 = 0.0
    m_1 = model.mass[n_shells - 1]

    # Numerically integrate over the whole star to find its total energy. Do not increase error tolerances (atol, rtol), as this can lead to unexpected behavior.
    result = quad(integrand, m_0, m_1, args=(arr_mass, arr_radius, arr_temp, arr_dens, arr_m_mu), limit=1000)
    result = result[0]

    return result
