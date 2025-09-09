def mass_loss(mass_a, mass_b, do_const=False, const_mass_loss=0.1):
    """
    mass_a = Mass of star a
    mass_b = Mass of star b
    do_const = Flag for constant mass loss. If set to True the const_mass_loss parameter will be used.
    const_mass_loss = Fraction of total mass of the system that is lost during the merger.
    q = Mass ratio
    f_ml = Mass loss in percent
    Routine for calculating fractional mass loss.
    """

    q = min(mass_a, mass_b) / max(mass_a, mass_b)

    if do_const:
        f_ml = 100 * const_mass_loss
    else:
        f_ml = 8.36 * q ** (-2.58) * (2 * q / (1 + q)) ** 4.28

    print(f"f_ml = {f_ml}")

    return f_ml
