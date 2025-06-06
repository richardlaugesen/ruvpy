from typing import Callable
import numpy as np


# ------------------------------------------------
# DAMAGE FUNCTION
# ------------------------------------------------

def damage_function_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    damage_function_param_1 = params['damage_function_param_1']
    damage_function_param_2 = params['damage_function_param_2']
    # ...

    # define the actual damage function
    def damage_function(magnitude: np.ndarray) -> np.ndarray:

        # do something with the parameters, perhaps adjust the magnitude
        magnitude = magnitude / damage_function_param_1

        return magnitude

    return damage_function
