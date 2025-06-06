from typing import Callable
import numpy as np


# ------------------------------------------------
# ECONOMIC MODEL
#
# Note for now it does not use the closure
# pattern as in the other components and only
# accepts single parameter economic models
# ------------------------------------------------

def economic_model_template(econ_param_1: float, values: np.ndarray, spend: float, damage_function: Callable) -> np.ndarray:
    damages = damage_function(values)
    outcome = damages * econ_param_1 - spend
    return outcome


# fast analytical solution for spend, or return None
def economic_model_analytical_spend_template(econ_param_1: float, value: float, damage_function: callable) -> float:
    return None

