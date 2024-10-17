# RUVPY

[![Tests](https://github.com/richardlaugesen/ruvpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/richardlaugesen/ruvpy/actions/workflows/python-package.yml)
![Coverage](https://img.shields.io/badge/dynamic/json?color=green&label=Coverage&query=$.files[%27coverage.json%27].content&url=https://api.github.com/gists/a08622619e06b2157bee092f47e404d9)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.13939199.svg)](https://doi.org/10.5281/zenodo.13939199)

[Open Science](https://en.wikipedia.org/wiki/Open_science) infrastructure written in [Python](https://python.org/).

RUVPY is a library which can be used in your software to quantify the value of forecast information for decision-making.

It is a reference implementation of the Relative Utility Value (RUV) method, which is very flexible and can accommodate a wide range of decisions.

It includes a set of commonly used decision rules, utility functions, damage functions, and economic models.
The implementation is sufficiently computationally efficient for most situations and parallelises timesteps over available CPU cores. 
The primary focus of this implementation is clarity and flexibility.

The scope is intentionally narrow and does not include any figure generation, data loading and saving, other metrics, or analysis functionality. 
These functions are intended to be implemented in a larger workflow or analysis pipeline which calls the main entry point of this library. 

## Publications

The method and software package are introduced in detail in the following publications. We suggest reading these to understand the context and motivation for the software.

*Laugesen, R., Thyer, M., McInerney, D., and Kavetski, D.: Flexible forecast value metric suitable for a wide range of decisions: application using probabilistic subseasonal streamflow forecasts, Hydrol. Earth Syst. Sci., 27, 873–893, https://doi.org/10.5194/hess-27-873-2023, 2023.*

*Laugesen, R., Thyer, M., McInerney, D., Kavetski, D. (2024). Software library to quantify the value of forecasts for decision-making: Case study on sensitivity to damages (under review).*

## Installation

The package is available on PyPI and can be installed using pip:

    pip install ruvpy

## Documentation

Generated documentation is available at https://richardlaugesen.github.io/ruvpy/.

## Examples

The package includes a set of examples corresponding to each figure in the publications noted above. 
These are all implemented as Jupyter notebooks in the `examples` directory.

## Templates

RUV is designed to be tailored to the decision being evaluated. 
This may require the development of custom components to define the decision context in RUVPY. 
A set of templates to help you get started is included in `templates` directory.

Please consider [contributing](CONTIBUTING.md) your new components to the repository to help others.

## Development

The main package requires Python (>=3.8), NumPy, SciPy, and Pathos.
The examples additionally require XArray, Pandas, Jupyter, and Matplotlib, the tests require Pytest and Statsmodels, 
and generating docs requires pdoc3.

All dependencies are defined in an included pyproject.toml file ready for use with [Poetry](https://python-poetry.org/) 
or Setuptools.

For example, the following commands can be used once Poetry is installed to set the environment up:

    poetry install
    poetry shell

And then to run the unit tests use: 

    pytest

To run examples use:

    jupyter notebook

Regenerate documentation using:

    pdoc --html --output-dir docs ruvpy --force

## Attribution

This project is licensed under the [Apache License 2.0](LICENSE), which allows for free use, modification, 
and distribution of the code under the terms of the license.

We would like to acknowledge and thank the [RUVPY developers](AUTHORS) for their efforts in maintaining and improving 
this project. Please see the [AUTHORS](AUTHORS) file for a full list of contributors.

For proper citation of this project, please refer to the [CITATION.cff](CITATION.cff) file, which provides guidance on 
how to cite the software and relevant publications.

## Code of conduct

We encourage you to [contribute](CONTRIBUTING.md)! Everyone interacting with this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contact

Richard Laugesen ([richard@laugesen.com.au](mailto://richard@laugesen.com.au))
