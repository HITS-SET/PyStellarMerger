import numpy as np
from PyStellarMerger.data.units import *
from numba import jit


@jit(nopython=True)
def beta_func(beta, y):
    zeta = 5.0*np.log(1.0 - beta) - 8.0*np.log(beta) + 32.0/beta
    zeta = y - zeta

    return zeta


@jit(nopython=True)
def compute_beta(pressure, entropy, mean_mu):
    eps = 1.0e-7
    max_iter = -int(np.log(eps)/np.log(2.0)) + 1

    beta_min = eps
    beta_max = 1.0 - eps

    delta = 3 * uK**4 / uA_RAD
    y = 3.0*np.log(pressure) - 5*np.log(delta) + 12*np.log(entropy) + 20*np.log(mean_mu*uM_U)

    beta = 0

    for i in range(max_iter):
        beta = 0.5 * (beta_min + beta_max)
        zeta = beta_func(beta, y)

        if zeta < 0:
            beta_min = beta
        else:
            beta_max = beta

    beta = 0.5 * (beta_min + beta_max)

    return beta


@jit(nopython=True)
def compute_entropy(density, temperature, mean_mu):
    Pgas = density/(mean_mu*uM_U) * uK * temperature
    Prad = uA_RAD/3.0 * temperature**4
    Ptot = Pgas + Prad
    beta = Pgas/Ptot
    A = np.log(Pgas) - 5.0/3.0*np.log(density) + 8.0/(3.0*beta)
    A = np.exp(A)

    return A


@jit(nopython=True)
def calc_temp(q, r):
    k = 0.125 * q**2
    kh = 0.5 * k

    if kh**2 - (r/3)**3 <= 0:
        print('Error in calc_temp: Imaginary result.')

    piece1 = kh + (kh**2 - (r/3)**3)**0.5
    piece2 = kh - (kh ** 2 - (r / 3) ** 3)**0.5

    y1 = piece1**(1/3)

    y2 = -(np.abs(piece2))**(1/3)

    yy = y1 + y2

    aa = 2 * yy

    b = -q
    b2 = aa**0.5
    c2 = 0.5 * b / b2 + yy

    x3 = 0.5*(-b2 + (b2**2 - 4*c2)**0.5)

    if piece1 < 0:
        print('piece 1 < 0')

    if piece1 == -piece2:
        print('piece1 = piece2, (rad pressure dominates)')
        x3 = (-r - q*(-r - q*(-r)**0.25)**0.25)**0.25

    if piece2 >= 0:
        x3 = -(r + (r/q)**4)/q

    return x3


@jit(nopython=True)
def compute_temperature(density, pressure, mean_mu):
    t0 = 1.0e6
    r = -3*pressure/uA_RAD * 1.0/(t0**4)
    q = 3*density * uK/(uA_RAD*mean_mu*uM_U) * 1/(t0**3)
    temperature = t0*calc_temp(q, r)

    return temperature


@jit(nopython=True)
def compute_density(pressure, entropy, mean_mu):
    beta = compute_beta(pressure, entropy, mean_mu)
    density = (beta*pressure)/entropy * np.exp(8.0/(3.0*beta))
    density = density**(3.0/5.0)

    return density


@jit(nopython=True)
def compute_energy(density, temperature, mean_mu):
    ugas = 1.5*uK/uM_U * temperature/mean_mu
    urad = uA_RAD*temperature**4 / density

    return ugas + urad


@jit(nopython=True)
def compute_pressure(density, energy, mean_mu):
    mean_mu *= uM_U

    t0 = 1.0e6
    r = -density*energy/uA_RAD * 1.0 / t0**4
    q = (uK/uA_RAD)/(5.0/3.0 - 1) * density/mean_mu * 1.0/t0**3

    temperature = t0 * calc_temp(q, r)

    p_gas = density/mean_mu * uK * temperature
    p_rad = uA_RAD/3.0 * temperature**4
    pressure = p_gas + p_rad

    return pressure


@jit(nopython=True)
def compute_mu(elements, am):
    mean_mu = 1.0 / sum(elements*am)
    return mean_mu
