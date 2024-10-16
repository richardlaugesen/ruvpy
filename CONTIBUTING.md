# Contributing to RUVPY

First off, thank you for considering contributing to RUVPY! This project is built on open collaboration, and your contributions help make it better for everyone.

The following guidelines aim to provide a smooth and constructive contribution process. Please take a moment to review them.

## Table of Contents

1. [How Can I Contribute?](#how-can-i-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Features](#suggesting-features)
    - [Submitting Code](#submitting-code)
2. [Coding Standards](#coding-standards)
3. [Pull Request Process](#pull-request-process)
4. [Running Tests](#running-tests)
5. [Getting Added to the AUTHORS File](#getting-added-to-the-authors-file)
6. [License](#license)

## How Can I Contribute?

### Reporting Bugs

If you discover a bug in RUVPY, please check the [existing issues](https://github.com/richardlaugesen/ruvpy/issues) first to see if it has already been reported. If not, you can open a new issue and include:

- A clear, descriptive title.
- A description of the problem, including steps to reproduce it.
- Relevant code snippets or a minimal, reproducible example.
- Information about your environment (OS, Python version, etc.).

### Suggesting Features

If you have an idea for a new feature, we'd love to hear it! Please submit a feature request as a [new issue](https://github.com/richardlaugesen/ruvpy/issues), including:

- A detailed explanation of the feature.
- Why you think it would be beneficial to RUVPY users.
- Any examples of how it could be used.

### Submitting Code

We welcome contributions in the form of pull requests! Before submitting a pull request (PR):

1. **Fork the repository** and create a new branch from `main` (e.g., `feature/my-new-feature`).
2. Make your changes, following the guidelines in the [Coding Standards](#coding-standards) section.
3. Ensure all existing tests pass and add new tests for any new functionality.
4. Commit your changes with a clear message (e.g., `Add feature for flexible decision thresholds`).
5. Push your branch to your fork and open a pull request.

We aim to review pull requests as soon as possible. Please be patient and ready to make any necessary revisions based on feedback.

## Coding Standards

Please follow these guidelines to ensure consistency across the codebase:

- **PEP 8**: Follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) for Python code.
- **Type Annotations**: Use type hints and annotations where appropriate.
- **Docstrings**: Write clear, descriptive [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for all functions and classes.
- **Testing**: Ensure that all new code is covered by tests. We use [pytest](https://docs.pytest.org/) for testing.
- **Comments**: Use comments to explain the reasoning behind complex code. Avoid obvious comments that explain what is already clear from the code itself.

## Pull Request Process

1. Ensure your pull request is **up-to-date with the latest main branch** before submitting.
2. If your pull request adds new functionality, make sure to include or update relevant documentation.
3. Verify that your code passes all tests by running `pytest`.
4. Open a pull request and provide a detailed description of your changes.
5. Wait for the review process. We might ask for revisions or feedback, which you can address by adding new commits to your PR branch.
6. Once approved, your PR will be merged into the main branch.

## Running Tests

To run the test suite locally, follow these steps:

1. Ensure you have the required dependencies installed. You can do this by installing the development environment using `Poetry`:

    ```bash
   poetry install
   poetry shell
    ```

2. Run the tests using `pytest`:

    ```bash
    pytest
    ```

Make sure all tests pass before submitting your contribution. If you’ve added new functionality, write tests for it and verify that they pass.

## Getting Added to the AUTHORS File

If you make a significant contribution to the project, we'd love to recognize your work! After your pull request is merged, you are welcome to:

1. Open another pull request to add yourself to the [AUTHORS](AUTHORS) file.
2. Include your name, email, and affiliation (if applicable) following the format of existing entries.

By adding yourself to the `AUTHORS` file, you're helping ensure that contributors are properly credited for their work.

## License

By contributing to this repository, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), which governs the use of this project.
