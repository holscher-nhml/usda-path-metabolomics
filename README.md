This repository contains the (anonymized) raw data and analysis code used for [Fecal Metabolites as Biomarkers for Predicting Food Intake by Healthy Adults](https://www.sciencedirect.com/science/article/pii/S0022316623086856?via%3Dihub) by Shinn et al. The following is a description of the files in the repository:

- `LICENSE.md` describes the open-source license the code (**but not the data**) in this repository is released under.

- `metabolomics_baseline_2021_2.csv`, `metabolomics_end_2021_2.csv`, and `metadata.csv` contain the data files used by the authors.

- `notebook.ipynb` contains the Jupyter notebook used to perform the analyses and produce plots included in the accompanied manuscript. The file is ready to run from this repository as-is in a Jupyter environment provided the dependencies in `requirements.txt` are met. Use `pip install -r requirements.txt` in your Python 3.7+ (virtual) environment to install all required dependencies for this notebook.