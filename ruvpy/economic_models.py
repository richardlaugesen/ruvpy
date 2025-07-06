"""Economic models representing the cost and benefit structure.

These functions provide simple formulations for converting spending
and damages into a net utility.
"""

import numpy as np


def cost_loss(alpha: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    """Standard cost-loss model.

    Net utility is calculated as benefits minus damages and spending,
    with benefits capped at ``spend / alpha``.
    """
    damages = damage_function(values)
    benefits = np.minimum(spend/alpha, damages)
    return benefits - damages - spend


def cost_loss_analytical_spend(alpha: float, values: float, damage_function: callable) -> float:
    """Analytical solution for the optimal spend in a cost-loss model.

    Computes the spend that maximises expected profit for a
    deterministic forecast.
    """
    return damage_function(values) * alpha
