# Non-stationary-time-series-attribution-for-heatwaves-over-Europe

The increasing occurrence of extreme weather events since the beginning of the 21st century 
has led to the development of new methods to attribute extreme events to anthropogenic climate 
change. The way in which the extreme event is defined has a major influence on the attribution 
result. A frequently disregarded or overlooked aspect concerns the temporal dependence and the 
clustering of extremes. 
This study presents an approach for attributing complete time series during extreme events to 
anthropogenic forcing. The approach is based on a non-stationary Markov process using bivariate 
extreme value theory to model the temporal dependence of the time series. We calculate the 
likelihood ratio of an observational time series from ERA5 given the distributions as estimated 
from CMIP6 simulations with historical natural-only and natural and anthropogenic forcing 
scenarios. The spatial fields are condensed by the extremal pattern index as a compact 
description of spatial extremes. In addition, the study examines the extent to which attribution 
statements about the occurrence of extreme heat events change when the effect of the mean warming 
is eliminated. The resulting attribution statement provides very strong evidence for the scenario 
with anthropogenic drivers over Europe, especially since the beginning of the 21st century. For 
central and southern Europe, the influence of anthropogenic greenhouse gas emissions on heatwaves 
could already have been proven in the 1970s using today's methods. There is no reliable signal 
apart from a general increase in temperature, neither in terms of the temporal dependence of 
extreme heat days nor in terms of the shape of the extreme value distribution.

## Contents of this repository

### `epi_output/`
Contains the results of the **Extremal Pattern Index (EPI)** analysis based on CMIP6 simulations 
and ERA5. The output is organized by spatial region:

- **`reg_16/`** – Results for region 16 (Central Europe)
- **`reg_17/`** – Results for region 17 (Southern Europe)
- **`reg_19/`** – Results for region 19 (Northern Europe)

Each subfolder contains subfolders for the different CMIP6 models and ERA5, which contain 
NetCDF files (`.nc`) with the EPI computed from CMIP6 simulations under natural-only (`nat`) 
and anthropogenic + natural (`hist`) forcing scenarios, as well as from ERA5 reanalysis data.

### Markov model output
Due to file size constraints, the Markov model output files are not included in this repository.
The results can be fully reproduced by running `attribution_analysis.ipynb` using the data 
in `epi_output/`. The visualisation notebook can be requested from the authors.

### Jupyter Notebooks

- **`attribution_analysis.ipynb`** – Full analysis pipeline from data loading to final 
  estimation of the non-stationary Markov model. Produces the Markov output for all 9 CMIP6 
  models and all three regions.
- **`visualisation.ipynb`** – Loads the fitted model output and ERA5 data and produces the 
  main result figures. Available from the authors on request.

All input data required to run the notebooks is provided in `epi_output/`.
No additional data download is necessary.

## Workflow notes & Computational requirements

To ensure reproducibility, all relevant steps are documented as Jupyter Notebooks.
The core functions are provided in `markov_functions.py`, which is shared between the 
analysis and visualisation notebooks.

The software versions used to run the analyses are the following:

Python (3.13.3)
- `cartopy` (0.24.0)
- `matplotlib` (3.10.3)
- `numpy` (2.2.6)
- `pandas` (2.2.3)
- `scipy` (1.15.2)
- `statsmodels` (0.14.5)
- `tqdm` (4.67.1)
- `xarray` (2025.1.2)

## References
Beirlant, J., Goegebeur, Y., Segers, J., and Teugels, J.: Statistics of Extremes: Theory and 
Applications, Wiley, ISBN 0471976474, 522 pp., 2004.

Cooley, D. and Thibaud, E.: Decompositions of dependence for high-dimensional extremes, 
Biometrika, 106, 587–604, https://doi.org/10.1093/biomet/asz028, 2019.

Jiang, Y., Cooley, D., and Wehner, M. F.: Principal Component Analysis for Extremes and 
Application to U.S. Precipitation, Journal of Climate, 33, 6441–6451, 
https://doi.org/10.1175/JCLI-D-19-0413.1, 2020.

Smith, R. L., Tawn, J. A., and Coles, S. G.: Markov chain models for threshold exceedances, 
Biometrika, 84, 249–268, https://doi.org/10.1093/biomet/84.2.249, 1997.

Szemkus, S. and Friederichs, P.: Spatial patterns and indices for heat waves and droughts over 
Europe using a decomposition of extremal dependency, Advances in Statistical Climatology, 
Meteorology and Oceanography, 10, 29–49, https://doi.org/10.5194/ascmo-10-29-2024, 2024.
