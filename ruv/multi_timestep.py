# Copyright 2023 Richard Laugesen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ruv.single_timestep import *
from ruv.data_classes import *
from pathos.multiprocessing import ProcessPool as Pool      # pathos.pools


# Calculate RUV for a single economic parameter value, parallelises over timesteps
def multiple_timesteps(econ_par: float, data: InputData, context: DecisionContext, parallel_nodes: int, verbose: bool = False) -> SingleParOutput:
    if parallel_nodes == 1:
        results = []
        for t, ob in enumerate(data.obs):
            if not np.isnan(ob):    
                results.append(single_timestep(t, econ_par, data, context))
    else:
        args = []
        for t, ob in enumerate(data.obs):
            if not np.isnan(ob):
                args.append([t, econ_par, data, context])
        args = list(map(list, zip(*args)))
        with Pool(nodes=parallel_nodes) as pool:
            results = pool.map(single_timestep, *args, chunksize=(len(data.obs) // parallel_nodes))

    # TODO: refactor this into a Dict_to_SingleParOutput function or just make the single_timestep return SingleParOutput for a single timestep
    output = SingleParOutput(data.obs.shape[0])
    for result in results:
        t = result['t']
        output.obs_spends[t] = result['ob_spends']
        output.obs_ex_post[t] = result['ob_ex_post']
        output.fcst_spends[t] = result['fcst_spend']
        output.fcst_ex_post[t] = result['fcst_ex_post']
        output.ref_spends[t] = result['ref_spend']
        output.ref_ex_post[t] = result['ref_ex_post']
        output.fcst_expected_damages[t] = result['fcst_expected_damage']
        output.ref_expected_damages[t] = result['ref_expected_damage']
        output.obs_damages[t] = result['ob_damage']

    output.avg_fcst_ex_post = np.nanmean(output.fcst_ex_post)
    output.avg_obs_ex_post = np.nanmean(output.obs_ex_post)
    output.avg_ref_ex_post = np.nanmean(output.ref_ex_post)
    output.ruv = (output.avg_ref_ex_post - output.avg_fcst_ex_post) / (output.avg_ref_ex_post - output.avg_obs_ex_post)

    if verbose:
        print('Economic model parameter: %.3f   RUV: %.2f' % (econ_par, output.ruv))

    return output
