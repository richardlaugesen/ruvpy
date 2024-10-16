# Copyright 2024 Richard Laugesen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
