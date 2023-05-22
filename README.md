# Relative Utility Value (RUV)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)
[![An ethical open source project](https://img.shields.io/badge/ethical-source-%23bb8c3c?labelColor=393162)](https://ethicalsource.dev/definition/)

RUV is a library to quantify the value of forecast information for decision making using the Relative Utility Value metric (Laugesen, 2023). 

[Open Science](https://en.wikipedia.org/wiki/Open_science) infrastructure written in [Python](https://python.org/).

## Status

Under ongoing development and not ready for general use. Don't expect backward compatibility at this stage.

## Installing

Empty

## Publications

Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.

## Functionality

Reference implementation of the Relative Utility Value metric (RUV) and a small library of utility functions, damage functions, and economic models. It does not include any plotting, other metrics, data loading and saving, or analysis.

### Utility functions

The RUV method has a foundation in Expected Utility Theory and as such requires a Utility function to be specified by the user. This maps an outcome from the economic model to decision-maker utility. Any custom function can be provided, and the following standard damage functions are implemented in the library:

- Constant absolute risk aversion (CARA)
- Constant relative risk aversion (CRRA)
- Risk neutral (linear)

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

## Software

### Features

- Simple front end
- Unit tests with 100% coverage
- Examples as Jupyter notebooks
- Free and Open Source Software licence
- Community code of conduct
- Python v3 with standard code conventions
- Only two dependencies (numpy, scipy)
- Sane versioning and DOI

### Conventions

- Functional style programming
- Clarity over cleverness
- DRY and YAGNI
- Vanilla python and numpy arrays
- Functions return dictionaries with descriptively named keys when needed
- Function and variable naming is unambiguous

### Dependencies

Sufficiently modern version of python 3, numpy, scipy.

## Code of conduct

We encourage you to contribute! Everyone interacting with this project is expected to follow the [Code of Conduct](code_of_conduct.md). 

## Copyright and licence

Copyright 2023 Richard Laugesen

RUV is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

VOF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER)

## Contact

Richard Laugesen - [Web](https://richardlaugesen.com) / [Github](https://github.com/richardlaugesen) / [Twitter](https://twitter.com/richardlaugesen) / [LinkedIn](https://www.linkedin.com/in/richardlaugesen/) / [Email](mailto://ruv@richardlaugesen.com)
