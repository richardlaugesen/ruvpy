import sys

sys.path.append('..')

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

@dataclass
class Results:
    ruv: np.array
    fcst_avg_ex_post: np.array
    obs_avg_ex_post: np.array
    ref_avg_ex_post: np.array
    fcst_spends: np.array
    obs_spends: np.array
    ref_spends: np.array
    fcst_likelihoods: np.array
    ref_likelihoods: np.array
    fcst_ex_post: np.array
    obs_ex_post: np.array
    ref_ex_post: np.array

@dataclass
class Inputs:
    obs: np.array
    fcsts: np.array
    refs: np.array
