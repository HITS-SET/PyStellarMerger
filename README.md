# Python Make Me A Massive Star (PyMMAMS) 
## Introduction
This repository contains the "Make Me A Massive Star" code, originally presented in [Gaburov et al., (2008)](https://ui.adsabs.harvard.edu/abs/2008MNRAS.383L...5G/abstract), rewritten in Python. We have added the capability of directly loading MESA models using [MESA reader](https://github.com/wmwolf/py_mesa_reader), tracking of a custom set of chemical species, and a new remeshing scheme for alleviating double-valuedness in the raw merger product. Furthermore, we have fixed minor bugs affecting the shock heating procedure.

We use isotope data taken from MESA's isotope data file ("$MESA_DIR/chem/data/isotopes.data") (Paxton et al., [2011](https://ui.adsabs.harvard.edu/abs/2011ApJS..192....3P/abstract), [2013](https://ui.adsabs.harvard.edu/abs/2013ApJS..208....4P/abstract), [2015](https://ui.adsabs.harvard.edu/abs/2015ApJS..220...15P/abstract), [2018](https://ui.adsabs.harvard.edu/abs/2018ApJS..234...34P/abstract), [2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..243...10P/abstract); Jermyn et al., [2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..265...15J/abstract)) to support custom chemical species.

## Dependencies
The following packages are required to run PyMMAMS:
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Numba](https://github.com/numba/numba)
- [MESA Reader](https://github.com/wmwolf/py_mesa_reader)

All dependencies can be installed via pip.

## Usage
The merger code is invoked by 
```
python3 PyMMAMS.py -i input.json
```
An example input file is given [here](examples/input.json). The code expects the following parameters to be present:

- **primary_star**: Path to the primary star model. MESA Reader is used to read files ending in ".data" and ".data.gz". Files ending ".txt" can also be supplied; these should follow the [example input files](examples/primary.txt). 
- **secondary_star**: Path to the secondary star model. It needs to be of the same file type as the primary star model.
- **n_target_shells**: Rough number of shells the merger remnant should have **before** mixing or remeshing. The actual number of shells will be between **n_target_shells** and $3\cdot$**n_target_shells**.
- **enable_mixing**: Whether the merger product should be mixed before output. This refers to the mixing routine employed in the original MMAMS code.
- **mixing_shells**: Maximum number of shells the merger remnant should have **after** mixing.
- **enable_remeshing**: Whether the merger product should be remeshed before output. The remeshing scheme does not mix over steep composition gradients and conserves the total mass of the individual chemical species.
- **remeshing_shells**: Rough number of shells the merger product should have **after** remeshing. The actual number of shells will usually be larger by about a factor of two. Increasing/decreasing this parameter can help alleviate possible spikiness in the composition profiles of the remeshed merger products.
- **enable_shock_heating**: Whether the progenitors should be shock-heated before the merger.
- **f_mod**: Modification factor for the shock heating. Only values $\geq 0.0$ are permitted. A value of $1.0$ corresponds to the original MMAMS shock heating.
- **relaxation_profiles**: Whether chemical composition and temperature-density profiles should be output for relaxation in MESA.
- **extrapolate_shock_heating**: If set to true, the shock heating parameters are extrapolated in edge cases (mass ratios below $0.1$ and above $0.8$). If false, the values of the closest boundary are used.
- **initial_buoyancy**: Whether files containing the stellar structure including the buoyancy profile of the progenitors before shock heating should be output.
- **final_buoyancy**: Whether files containing the stellar structure including the buoyancy profile of the progenitors after shock heating should be output.
- **chemical_species**: List of all chemical species to track during the merger. The names of the species are identical to the ones used in MESA. Hydrogen ("h1") is required.
- **fill_missing_species**: If set to "true", chemical species missing from the input profiles will be assumed to have mass fraction 0. 
- **massloss_fraction**: Fraction of mass that should be lost from the system during the merger. It needs to be $<1.0$. Negative values invoke the original MMAMS mass loss prescription.
- **output_dir**: Directory to which the output files should be written.
- **output_diagnostics**: Whether a file containing all merger parameters should be written after completing the merger.

## Input stellar models
For compatibility with other stellar evolution codes besides MESA the code can also load stellar models from generic .txt files (see [primary.txt](examples/primary.data) and [secondary.txt](examples/secondary.data)). The files are organized as follows: Lines starting with "#" are regarded as comments and ignored. The first line without a hash at the beginning is taken as the beginning of the data and has to contain the names of the data columns. The code expects to find the following columns describing the stellar model's shells: 

- **mass**: The shell's mass coordinate in $\mathrm{M_\odot}$.
- **dm**: The shell's mass in $\mathrm{M_\odot}$.
- **radius**: The shell's radial coordinate in $\mathrm{R_\odot}$.
- **density**: The shell's density in $\mathrm{g\,cm^{-3}}$.
- **pressure**: The shell's total pressure in $\mathrm{g\,cm^{-1}\,s^{-2}}$. 
- **temperature**: The shell's temperature in $\mathrm{K}$.
- **mean_mu**: The shell's mean molecular weight $\mathrm{\mu}$.
- **h1, ...**: The mass fraction of all the chemical species that were defined in the input JSON file ([example](examples/input.json)). 

The columns can be ordered arbitrarily. Additional columns that don't fit any of the expected columns are ignored.

## Example merger
An example merger of a $\approx 10\,\mathrm{M_\odot}$ MS primary star with a $\approx 7\,\mathrm{M_\odot}$ companion of the same age is given in [examples/](examples/). The expected output is saved in [examples/example_output](examples/example_output/).
