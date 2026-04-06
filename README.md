# transferentropy-dorsogna

## Repository details
This presents the code for the undergraduate thesis, "Investigating Information Transfer in Emerging Phenotypes of the D’Orsogna Model" by Deanne Gabrielle Algenio, Vince Dexter Ang, and Josh Tiffton Bolipata under the advisory of Dr. Clark Go.

## Repository structure

### Python scripts
* `models/` contains scripts simulating the D'Orsogna model and other baseline models.
* `calculateTE.py` implements transfer entropy computation using JIDT, applied to state variations from the D'Orsogna phenotypes.
* `utils.py` provides helper functions used throughout the repository.
* `main.py` integrates `models/` and `calculateTE.py` to run transfer entropy computations end-to-end.

### Notebooks
* `00-preliminaries.ipynb` installs required packages and clones the [JIDT repository](https://github.com/jlizier/jidt), which is central to the transfer entropy computation.
* `01-investigate-dorsogna.ipynb` visualizes D'Orsogna model behaviour across multiple phenotypes.
* `02-export-TE.ipynb` executes the central computation, computing transfer entropy across the many simulations.
* `03-investigate-TE.ipynb` through `09-empirical-comparison.ipynb` implement the various analytical methods used in the paper.

### Data and outputs
* `empirical_data/` contains the data used for empirical validation in the thesis.
* Output folders (`csvs_...`, `simulations_...`, and `graphs/`) store generated joblib files, CSVs, PNGs, and GIFs produced by the code.

