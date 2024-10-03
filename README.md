# Relative Utility Value (RUV)

[![Tests](https://github.com/richardlaugesen/relative-utility-value/actions/workflows/python-package.yml/badge.svg)](https://github.com/richardlaugesen/relative-utility-value/actions/workflows/python-package.yml)
![Coverage](https://img.shields.io/badge/dynamic/json?color=green&label=Coverage&query=$.files[%27coverage.json%27].content&url=https://api.github.com/gists/a08622619e06b2157bee092f47e404d9)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

[Open Science](https://en.wikipedia.org/wiki/Open_science) infrastructure written in [Python](https://python.org/).

RUV is used to quantify the value of forecast information for decision-making. 
It is very flexible and can accomodate a wide range of decisions. 
This package is a reference implementation of the RUV method.

It includes a set of commonly used decision rules, utility functions, damage functions, and economic models.
The implementation is sufficiently computationally efficient for most situations and parallelises timesteps over available CPU cores. The primary focus is on clarity and flexibility.

The scope is intentionally narrow and does not include any figure generation, data loading and saving, other metrics, or analysis functionality. 
These functions are intended to be implemented in a larger workflow or analysis pipeline which calls the main entry point of this library. 

## Publications

The method and software package are introduced in detail in the following publications. We suggest reading these to understand the context and motivation for the software.

*Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.*

*Laugesen, R., Thyer, M., McInerney, D., Kavetski, D. (2024). Software to quantify the value of forecasts for decision-making: sensitivity to damages case study. Manuscript submitted for publication in Environmental Modelling & Software.*

## Installation

The package requires Python (>=3.8) and a few dependencies. The dependencies are defined in the (pyproject.toml) file for [Poetry](https://python-poetry.org/) dependency management.

Install the required dependencies using Poetry and then run the unit tests with the following commands:

    poetry install

    poetry shell

    poetry run pytest

## Example

The package includes a set of examples which loosely correspond with figures in the publications noted above. 

These are all implemented as Jupyter notebooks in the `examples` directory.

## Templates

RUV is designed to be tailored to the decision being evaluated. 
This may require the development of custom components to define the decision context. 
A set of templates to help the user get started is included in `templates` directory.

Please consider contributing your new components to the repository to help others.

## Code of conduct

We encourage you to contribute! Everyone interacting with this project is expected to follow the [Code of Conduct](code_of_conduct.md). 

## Contact

Richard Laugesen - [Web](https://laugesen.com.au) / [Email](mailto://ruv@laugesen.com.au)
