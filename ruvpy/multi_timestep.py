# Copyright 2024 RUVPY Developers

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from dask import delayed, compute
from pathos.multiprocessing import ProcessPool as Pool

from ruvpy.single_timestep import single_timestep
from ruvpy.data_classes import DecisionContext, SingleParOutput


def multiple_timesteps(obs: np.ndarray, fcsts: np.ndarray, refs: np.ndarray, econ_par: float, context: DecisionContext, parallel_nodes: int) -> SingleParOutput:
    """Evaluate a single economic parameter over all timesteps.

    The computation can be parallelised across timesteps using ``parallel_nodes``.
    """
    tasks = []
    for t, ob in enumerate(obs):
        if not np.isnan(ob):
            tasks.append(
                delayed(single_timestep)(t, econ_par, ob, fcsts[t], refs[t], context)
            )

    if parallel_nodes == 1:
        results = [task.compute() for task in tasks]
    else:
        results = list(
            compute(*tasks, scheduler="processes", num_workers=parallel_nodes)
        )

    output = _dict_to_output(results, obs.shape[0])
    output.ruv = _calc_ruv(output)

    return output


def multiple_timesteps_pathos(
    obs: np.ndarray,
    fcsts: np.ndarray,
    refs: np.ndarray,
    econ_par: float,
    context: DecisionContext,
    parallel_nodes: int,
) -> SingleParOutput:
    """Evaluate all timesteps using pathos for parallelism."""

    if parallel_nodes == 1:
        results = []
        for t, ob in enumerate(obs):
            if not np.isnan(ob):
                results.append(
                    single_timestep(t, econ_par, ob, fcsts[t], refs[t], context)
                )
    else:
        args = []
        for t, ob in enumerate(obs):
            if not np.isnan(ob):
                args.append([t, econ_par, ob, fcsts[t], refs[t], context])
        args = list(map(list, zip(*args)))

        with Pool(nodes=parallel_nodes) as pool:
            results = pool.map(
                single_timestep,
                *args,
                chunksize=(len(obs) // parallel_nodes),
            )

    output = _dict_to_output(results, obs.shape[0])
    output.ruv = _calc_ruv(output)

    return output


def _dict_to_output(results: dict, size: int) -> SingleParOutput:
    """Convert a list of timestep dictionaries to ``SingleParOutput``."""
    output = SingleParOutput(size)

    for result in results:
        t = result['t']
        output.obs_spends[t] = result['ob_spend'].item()
        output.obs_ex_post[t] = result['ob_ex_post'].item()
        output.fcst_spends[t] = result['fcst_spend'].item()
        output.fcst_ex_post[t] = result['fcst_ex_post'].item()
        output.ref_spends[t] = result['ref_spend'].item()
        output.ref_ex_post[t] = result['ref_ex_post'].item()

    return output


def _calc_ruv(output: SingleParOutput) -> float:
    """Compute the Relative Utility Value from accumulated results."""
    output.avg_fcst_ex_post = np.nanmean(output.fcst_ex_post)
    output.avg_obs_ex_post = np.nanmean(output.obs_ex_post)
    output.avg_ref_ex_post = np.nanmean(output.ref_ex_post)
    return (output.avg_ref_ex_post - output.avg_fcst_ex_post) / (output.avg_ref_ex_post - output.avg_obs_ex_post)
