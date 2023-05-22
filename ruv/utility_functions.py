# Copyright 2023 Richard Laugesen
#
# This file is part of RUV
#
# RUV is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RUV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RUV.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# Constant Absolute Risk Aversion utility function
def cara(risk_aversion, outcome):
    return np.divide(-np.exp(np.multiply(-risk_aversion, outcome)), risk_aversion)
