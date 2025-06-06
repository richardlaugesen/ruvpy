import numpy as np


def cost_loss(alpha: float, values: np.ndarray, spend: float, damage_function: callable) -> np.ndarray:
    """Standard cost-loss model."""
    damages = damage_function(values)
    benefits = np.minimum(spend/alpha, damages)
    return benefits - damages - spend


def cost_loss_analytical_spend(alpha: float, values: float, damage_function: callable) -> float:
    """Analytical solution for the optimal spend in a cost-loss model."""
    return damage_function(values) * alpha
