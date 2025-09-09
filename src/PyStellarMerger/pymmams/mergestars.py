from PyStellarMerger.data.units import *
from PyStellarMerger.pymmams.sortmodel import sort_model
import scipy
from copy import deepcopy
from PyStellarMerger.io import stellarmodel
from PyStellarMerger.calc.massloss import mass_loss
from PyStellarMerger.calc.eos import compute_density, compute_temperature, compute_energy
import numpy as np

p_unit = uG * uMSUN**2 / uRSUN**4
rho_unit = uMSUN / uRSUN**3

NEXCEED = 10
FAILED = 11
NEGATIVE_PRESSURE = 12

ERROR_TRY0 = 1.0e-6
NTRY = 3000
ERROR_SCALE_INC = 3.9
ERROR_SCALE_DEC = 2.0


class merge_stars_params:
    def __init__(self):
        self.Morig = []
        self.mass = []
        self.radius = []
        self.pressure = []

        self.m_product = 0
        self.p_old = 0

        self.m_min_A = 0
        self.m_min_B = 0
        self.m_max_A = 0
        self.m_max_B = 0

        self.Mass_A = np.array([])
        self.Mass_B = np.array([])

        self.m_mu_A = np.array([])
        self.m_mu_B = np.array([])

        self.entr_A = np.array([])
        self.entr_B = np.array([])

        self.n_shells = 0
        self.status = 0

        self.error_try = 0
        self.initial_error_try = 0


class hse_func_params:
    def __init__(self):
        self.Morig_temp = []
        self.eggleton_mu = [0]

        self.mcur_A = 0
        self.mcur_B = 0
        self.m_max_A = 0
        self.m_max_B = 0

        self.Mass_A = np.array([])
        self.Mass_B = np.array([])

        self.m_mu_A = np.array([])
        self.m_mu_B = np.array([])

        self.entr_A = np.array([])
        self.entr_B = np.array([])

        self.negative_pressure = 0
        self.which_star = 0

        self.status = 0


def hse_func(mu, y, parameters):
    pressure = y[0]
    x = y[1]

    if pressure < 0:
        parameters.negative_pressure = 1
        #print("Negative pressure encountered")
        parameters.status = -1
        # Original code returns GSL_FAILURE here

    dm = mu**1.5

    if len(parameters.eggleton_mu) > 0:
        dm -= parameters.eggleton_mu[-1]**1.5

    mc_at = min(parameters.mcur_A + dm, parameters.m_max_A)
    rho_A = compute_density(pressure*p_unit, np.interp(mc_at, parameters.Mass_A, parameters.entr_A), np.interp(mc_at, parameters.Mass_A, parameters.m_mu_A)) / rho_unit

    mc_bt = min(parameters.mcur_B + dm, parameters.m_max_B)
    rho_B = compute_density(pressure*p_unit, np.interp(mc_bt, parameters.Mass_B, parameters.entr_B), np.interp(mc_bt, parameters.Mass_B, parameters.m_mu_B)) / rho_unit

    if parameters.mcur_A == parameters.m_max_A:
        rho_A = -1
    elif parameters.mcur_B == parameters.m_max_B:
        rho_B = -1

    if rho_A > rho_B:
        density = rho_A
        parameters.Morig_temp.append(mc_at)
        parameters.which_star = 1
    else:
        density = rho_B
        parameters.Morig_temp.append(-mc_bt)
        parameters.which_star = 2

    mu_x = (4.0*PI/3 * density)**(2.0/3.0)

    if x > 0:
        mu_x = mu/x

    dydmu = np.array([-3.0/(8*PI) * mu_x**2, 3.0/(4*PI) * np.sqrt(mu_x)/density])

    return dydmu


def hse_jac():
    return 0


def solve_HSE(p_centre, params):
    eps_abs = 0
    eps_rel = params.error_try

    params_hse = hse_func_params()

    params_hse.mcur_A = params.m_min_A
    params_hse.m_max_A = params.m_max_A
    params_hse.m_mu_A = params.m_mu_A
    params_hse.entr_A = params.entr_A

    params_hse.mcur_B = params.m_min_B
    params_hse.m_max_B = params.m_max_B
    params_hse.m_mu_B = params.m_mu_B
    params_hse.entr_B = params.entr_B

    params_hse.Mass_A = params.Mass_A
    params_hse.Mass_B = params.Mass_B

    params_hse.eggleton_mu = [0]

    params_hse.Morig_temp = []

    m_c = 0.0
    m_1 = params.m_product**(2.0/3.0)
    dm = 1.0e-6
    y = np.array([p_centre / p_unit, 0.0])

    params.Morig = []
    params.mass = []
    params.radius = []
    params.pressure = []

    params_hse.negative_pressure = 0

    dop853solver = scipy.integrate.DOP853(lambda t, y: hse_func(t, y, params_hse), m_c, y, m_1, first_step=dm, rtol=eps_rel, atol=eps_abs)
    counter = 0
    
    while m_c < m_1:
        params_hse.Morig_temp = []
        params_hse.which_star = 0
        #print(f"{counter} m_c={m_c}, dm = {dm}, y = [{y[0]}, y1 = {y[1]}], status={dop853solver.status}")
        counter += 1
        dop853solver.step()
        m_c = dop853solver.t
        dm = dop853solver.t_old
        y = dop853solver.y

        dm_2 = m_c**1.5

        if len(params_hse.eggleton_mu) > 0:
            dm_2 -= params_hse.eggleton_mu[-1]**1.5
            params_hse.eggleton_mu.append(m_c)
            #print(f"size > 0, m_c = {m_c}")
        else:
            params_hse.eggleton_mu.append(m_c)
            #print(f"size < 0, m_c = {m_c}")

        if params_hse.which_star == 1:
            params_hse.mcur_A += dm_2
            params_hse.mcur_A = min(params_hse.mcur_A, params.m_max_A)
        else:
            params_hse.mcur_B += dm_2
            params_hse.mcur_B = min(params_hse.mcur_B, params.m_max_B)

        if params_hse.negative_pressure == 1:
            params.pressure.append(-p_unit)
            #print("NEGATIVE_PRESSURE")
            return NEGATIVE_PRESSURE

        if len(params.mass) + 1 > params.n_shells*3:
            #print("NEXCEED")
            return NEXCEED

        if params_hse.status != 0 or dop853solver.status == "failed":
            #print("FAILED")
            return FAILED

        params.mass.append(m_c**1.5)
        params.pressure.append(y[0]*p_unit)
        params.radius.append(np.sqrt(y[1]))
        params.Morig.append(params_hse.Morig_temp[-1])

    return 0


def merge_stars_eq(p_centre, params):
    params.error_try = params.initial_error_try
    do_loop = True
    pc = 0
    #print(f"p_centre = {p_centre}")

    while do_loop:
        if pc > 10:
            print(f"p_centre = {p_centre}, pc = {pc}, p.n_shells = {params.n_shells}, p.error_try = {params.error_try}, p.mass_size = {len(params.mass)}")
        params.status = solve_HSE(p_centre*p_unit, params)
        pc += 1

        if pc >= 100:
            print("ODE solver can't find fitting model, quit.")
            exit(1)

        if params.status == NEXCEED:
            params.error_try *= ERROR_SCALE_INC
            #print("NEXCEED")
        else:
            if len(params.mass) < params.n_shells:
                params.error_try *= 1.0/ERROR_SCALE_DEC
                #print(f"ERROR_SCALE_DEC, {len(params.mass)}, {params.n_shells}")
            else:
                do_loop = False
                #print("do_loop = False")

    dp = params.pressure[-1]/p_centre/p_unit
    #print(f"dp = {dp}")
    return dp


def merge_stars(f_lost, n_desired_shells, m_a, m_b, mass_loss_flag, mass_loss_fraction):
    # Work with copies of the models for now
    model_a = deepcopy(m_a)
    model_b = deepcopy(m_b)

    # Sort model_a
    Mass_a, entr_a, m_mu_a, id_sA = sort_model(model_a)

    m_min_A = max(Mass_a[1], Mass_a[2]*1.0e-4)
    m_max_A = (1.0 - 1.0e-10)*Mass_a[-1]

    # Sort model_b
    Mass_b, entr_b, m_mu_b, id_sB = sort_model(model_b)

    m_min_B = max(Mass_b[1], Mass_b[2] * 1.0e-4)
    m_max_B = (1.0 - 1.0e-10) * Mass_b[-1]

    # Merge the stars
    p = merge_stars_params()
    p.m_min_A = m_min_A
    p.m_min_B = m_min_B
    p.m_max_A = m_max_A
    p.m_max_B = m_max_B
    p.Mass_A = Mass_a
    p.Mass_B = Mass_b
    p.entr_A = entr_a
    p.entr_B = entr_b
    p.m_mu_A = m_mu_a
    p.m_mu_B = m_mu_b
    p.m_product = p.m_max_A + p.m_max_B
    f_lost = (1 - f_lost*mass_loss(model_a.star_mass, model_b.star_mass, mass_loss_flag, mass_loss_fraction)/100.0)
    p.m_product *= f_lost

    if n_desired_shells > 10000:
        p.n_shells = 10000
    else:
        p.n_shells = n_desired_shells

    print("Bracketing central pressure: ")
    p_0 = (model_a.pressure[0] + model_b.pressure[0]) / p_unit

    p.initial_error_try = ERROR_TRY0
    merge_stars_eq(p_0, p)
    p.initial_error_try = p.error_try

    sgn = p.pressure[-1]

    iterb_max = 100
    iterb = 0
    factor = 2
    stop = False
    p_last = p_0

    while not stop:
        if iterb > iterb_max:
            print("Error: Failed to bracket root. Quit.")
            exit(1)
        iterb += 1

        p_last = p_0

        if sgn < 0:
            p_0 *= factor
        else:
            p_0 *= 1.0 / factor
        #print(f"p_0 = {p_0}")
        merge_stars_eq(p_0, p)
        #print(f"p.pressure[-1] = {p.pressure[-1]}")

        if sgn*p.pressure[-1] < 0:
            stop = True

    p_min = min(p_last, p_0)
    p_max = max(p_last, p_0)

    print(f"p is in [{p_min}, {p_max}]")
    p.n_shells = n_desired_shells

    print("\n----------------------------------------------------\n")
    print("Computing p_centre using falsepos method:")
    
    iter_max = 1000

    p_centre, p_centre_info = scipy.optimize.brentq(merge_stars_eq, p_min, p_max, args=(p), full_output=True, maxiter=iter_max, rtol=1e-3)

    print(f"Converged in {p_centre_info.iterations} iterations.")

    print("\n----------------------------------------------------\n")

    print(f"p_centre = {p_centre}")

    n_A = model_a.n_shells
    mass_A = model_a.mass
    chemicals_A = model_a.elements # Put all elements and passive scalar into one array

    n_B = model_b.n_shells
    mass_B = model_b.mass
    chemicals_B = model_b.elements # Put all elements and passive scalar into one array

    mass_size = len(p.mass)

    # Create an empty model in which we can store the product, make sure it uses the same nuclear network as the progenitors
    product = stellarmodel.model(nuclear_net=model_a.element_names, n_shells=mass_size-1)

    product.star_mass = p.mass[mass_size-1]
    product.star_radius = p.radius[mass_size-1]

    print(f"mass_size = {mass_size}")

    for i in range(mass_size-1):
        product.radius[i] = p.radius[i]
        product.mass[i] = p.mass[i]
        product.pressure[i] = p.pressure[i]

        if p.Morig[i] > 0:
            m = p.Morig[i]
            product.buoyancy[i] = np.interp(m, Mass_a, entr_a)
            product.mean_mu[i] = np.interp(m, Mass_a, m_mu_a)

            # For each element j interpolate the progenitors abundance to find abundance at new mass coordinate in product shell i
            for j in range(model_a.elements.shape[0]):
                product.elements[j, i] = np.interp(m, mass_A, chemicals_A[j])
            product.passive_scalar[i] = np.interp(m, mass_A, model_a.passive_scalar)  # Keep passive scalar separate from the other elements

        else:
            m = -p.Morig[i]
            product.buoyancy[i] = np.interp(m, Mass_b, entr_b)
            product.mean_mu[i] = np.interp(m, Mass_b, m_mu_b)

            for k in range(model_b.elements.shape[0]):
                product.elements[k, i] = np.interp(m, mass_B, chemicals_B[k])
            product.passive_scalar[i] = np.interp(m, mass_B, model_b.passive_scalar)

        product.density[i] = compute_density(product.pressure[i], product.buoyancy[i], product.mean_mu[i])
        product.temperature[i] = compute_temperature(product.density[i], product.pressure[i], product.mean_mu[i])
        product.e_thermal[i] = compute_energy(product.density[i], product.temperature[i], product.mean_mu[i])

    print("Done merging stars.")

    return product
