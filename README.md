# Relative Utility Value (RUV)

[![Tests](https://github.com/richardlaugesen/relative-utility-value/actions/workflows/python-package.yml/badge.svg)](https://github.com/richardlaugesen/relative-utility-value/actions/workflows/python-package.yml)
![Coverage](https://img.shields.io/badge/dynamic/json?color=green&label=Coverage&query=$.files[%27coverage.json%27].content&url=https://api.github.com/gists/a08622619e06b2157bee092f47e404d9)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

[Open Science](https://en.wikipedia.org/wiki/Open_science) infrastructure written in [Python](https://python.org/).

RUV is a library to quantify the value of forecast information for decision making using the Relative Utility Value metric. This reference implementation is reasonably computationally efficient and parallelises timesteps over available CPU cores. The software includes a small library of standard utility functions, damage functions, and economic models. The scope is intentionally narrow and does not include any figure generation, data loading and saving, other metrics, or analysis functionality.

Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.

## Quick start for examples

Install [Poetry](https://python-poetry.org/) then run

    poetry install

    poetry shell

    jupyter notebook

Open up the browser

    http://localhost:8888/tree/examples

## Functionality

### Input data

The following data over a historical period is required, all are numpy arrays and can be either deterministic or ensemble:

- Forecasts under evaluation
- Reference forecasts
- Observations

The decision context needs to be fully defined:

- Utility function
- Damage function
- Economic model
- Decision making method
- Decision type and thresholds
- Alpha values

The forecast value reaults of RUV are identical to REV (Richardson, 2000) when a specific decision context is used.

### Output data

A Python dictionary with the following keys, all values are numpy arrays:

- decision_definition
- fcst_likelihoods
- ref_likelihoods
- obs_likelihoods
- fcst_avg_ex_post
- obs_avg_ex_post
- ref_avg_ex_post
- fcst_spends
- obs_spends
- ref_spends
- fcst_ex_post
- obs_ex_post
- ref_ex_post
- ruv

RUV values are returned for each alpha value requested in the decision context. Other values are returned for every combination of alpha and timestep defined by the observation data.

### Utility functions

The RUV method has a foundation in Expected Utility Theory and as such requires a Utility function to be specified by the user. This maps an outcome from the economic model to utility of the decision-maker. Any custom function can be provided, and the following utility functions are implemented in the library:

- Exponential utility
- Isoelastic utility
- Hyperbolic absolute risk aversion

These can, for example, be used to model constant absolute risk aversion (CARA), constant relative risk aversion (CRRA), and neutral risk aversion. 

### Damage functions

The RUV method also requires a Damage function to be specified. This returns the damages at a specified value. Any custom function can be provided, and the following standard damage functions are implemented in the library:

- Binary
- Linear
- Logistic
- Logistic with forced zero
- User defined with linear interpolation through arbitrary points

### Economic models

The RUV method also requires an economic model to be specified. This returns the net expense following a mitigation of the damages. Cost-loss is the only economic model implemented in the library but any custom function can be provided. 

### Decision types

The RUV method can quantify forecast value for binary, multi-categorical, or continuous-value decisions through user defined decision thresholds. A continuous-value decision type is specified by setting the decision threshold to None.

### Decision making methods

To calculate RUV it must simulate decision making at each timestep, there are four methods implemented:

- optimise_over_forecast_distribution - use the whole forecast distribution at each timestep to determine the optimium amount to spend on mitigation which is expected to maximise utility
- critical_probability_threshold_fixed - convert the probabilistic forecast into deterministic using a critical probability threshold for all timesteps, spend enough to mitigate the damages from the flowclass it falls within 
- critical_probability_threshold_equals_alpha - same as fixed threshold approach but set the critical probability threshold to equal alpha
- critical_probability_threshold_max_ruv - same as the equals-alpha approach but set the critical probability threshold to maximise forecast value for each alpha

### Helper functions

- Generate reference forecasts to replicate REV event frequency approach.
- Convert between CARA risk aversion coefficient, risk premium, and probability premium (Babcock, 1993).
- Calculate emperical cummulative probability distribution function efficiently.
- Check if forecasts are deterministic or ensemble.

## Software

### Features

- Simple front end
- Unit tests with close to 100% coverage
- Examples as Jupyter notebooks
- Free and Open Source Software licence
- Community code of conduct
- Python v3 with standard code conventions
- Only three core dependencies (numpy, scipy, pathos)
- Sane versioning and DOI

### Conventions

- Functional programming style
- Data Classes used to pass between functions
- Clarity over cleverness
- DRY and YAGNI
- Vanilla python and numpy arrays
- Function and variable naming is unambiguous

### Dependencies

Defined in (pyproject.toml) file for [Poetry](https://python-poetry.org/) dependency management.

## Code of conduct

We encourage you to contribute! Everyone interacting with this project is expected to follow the [Code of Conduct](code_of_conduct.md). 

## Contact

Richard Laugesen - [Web](https://laugesen.com.au) / [Email](mailto://ruv@laugesen.com.au)
