#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-core=2
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
#SBATCH -n 25

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=richard.laugesen@adelaide.edu.au

POETRY="$HOME/.local/bin/poetry"

$POETRY install

$POETRY run python shape_figure.py
