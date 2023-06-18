# Relative Utility Value (RUV)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

[Open Science](https://en.wikipedia.org/wiki/Open_science) infrastructure written in [Python](https://python.org/).

RUV is a library to quantify the value of forecast information for decision making using the Relative Utility Value metric. This reference implementation is reasonably computationally efficient and parallelises timesteps over available CPU cores. It includes a library of core utility functions, damage functions, and economic models. It has a tight scope and does not include any figure generation, data loading and saving, other metrics, or analysis functionality.

Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.

## Functionality

### Utility functions

The RUV method has a foundation in Expected Utility Theory and as such requires a Utility function to be specified by the user. This maps an outcome from the economic model to decision-maker utility. Any custom function can be provided, and the following utility functions are implemented in the library:

- Exponential utility
- Isoelastic utility
- Hyperbolic absolute risk aversion

These can, for example, be used to model constant absolute risk aversion (CARA), constant relative risk aversion (CRRA), and neurtral risk aversion. 

Helper functions included to convert between CARA risk aversion coefficient, risk premium, and probability premium (Babcock, 1993).

### Damage functions

The RUV method also requires a Damage function to be specified. This returns the cumulative damages up to a specified value. Any custom function can be provided, and the following standard damage functions are implemented in the library:

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

To calculate RUV it must simulate decision making at each timestep, there are three methods implemented:

- optimise_over_forecast_distribution uses the forecast distribution at each timestep to determine the optimium amount to spend on mitigation to maximise utility
- critical_probability_threshold_fixed
- critical_probability_threshold_equals_alpha

## Software

### Features

- Simple front end
- Unit tests with close to 100% coverage
- Examples as Jupyter notebooks
- Free and Open Source Software licence
- Community code of conduct
- Python v3 with standard code conventions
- Only three dependencies (numpy, scipy, pathos)
- Sane versioning and DOI

### Conventions

- Functional programming style
- Clarity over cleverness
- DRY and YAGNI
- Vanilla python and numpy arrays
- Functions return dictionaries with descriptively named keys when needed
- Function and variable naming is unambiguous

### Dependencies

Sufficiently modern version of python 3, numpy, scipy.

## Code of conduct

We encourage you to contribute! Everyone interacting with this project is expected to follow the [Code of Conduct](code_of_conduct.md). 

## Contact

Richard Laugesen - [Web](https://richardlaugesen.com) / [Email](mailto://ruv@richardlaugesen.com)
