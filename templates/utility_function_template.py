from typing import Callable


# ------------------------------------------------
# UTILITY FUNCTION
# ------------------------------------------------

def utility_function_template(params: dict) -> Callable:

    # extract parameters from the dictionary if needed
    utility_params_1 = params['utility_params_1']
    utility_params_2 = params['utility_params_2']
    # ...

    def utility(c: float) -> float:

        # use the parameters to adjust the outcome
        if utility_params_2 == 0:
            return c

        else:
            c = c * utility_params_1

    return utility
