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

from dask.distributed import Client
import dask.bag as db


# Calculate RUV for a single alpha value, parallelises over timesteps
def multiple_timesteps(alpha: float, data: InputData, context: DecisionContext, dask_client: Client=None, verbose: bool=False) -> SingleAlphaOutput:    

    # single thread in no dask_client
    if dask_client is None:
        results = []
        for t, ob in enumerate(data.obs):
            if not np.isnan(ob):    
                results.append(single_timestep(t, alpha, data, context))

    else:
        # partition all the timesteps with obs into chunks
        timesteps = np.nonzero(~np.isnan(data.obs))[0]
        num_processes = sum(dask_client.ncores().values())
        npartitions = int(len(timesteps) / np.sqrt(len(timesteps) / num_processes))
        timesteps_bag = db.from_sequence(timesteps, npartitions=npartitions)

        # process all timesteps in single partition sequentially
        def process_partition(ts, data_future, context_future):
            results = []
            for t in ts:
                results.append(single_timestep(t, alpha, data_future, context_future))
            return results

        # build dask tree to process partitions using immutable data and context across all workers
        data_future, context_future = dask_client.scatter([data, context], broadcast=True)
        results_bag = timesteps_bag.map_partitions(process_partition, data_future, context_future)            
        results_bag = dask_client.persist(results_bag)   # stop futures from being garbage collected

        # compute the lazy results and clean up
        results = dask_client.compute(results_bag).result()
        del data_future, context_future, results_bag

    # store all the results into output object
    output = SingleAlphaOutput(data.obs.shape[0])
    for result in results:
        t, obs_spends, obs_ex_post, fcst_spends, fcst_ex_post, ref_spends, ref_ex_post = result
        output.obs_spends[t] = obs_spends
        output.obs_ex_post[t] = obs_ex_post
        output.fcst_spends[t] = fcst_spends
        output.fcst_ex_post[t] = fcst_ex_post
        output.ref_spends[t] = ref_spends
        output.ref_ex_post[t] = ref_ex_post

    # calculate ruv for this alpha
    output.avg_fcst_ex_post = np.nanmean(output.fcst_ex_post)
    output.avg_obs_ex_post = np.nanmean(output.obs_ex_post)
    output.avg_ref_ex_post = np.nanmean(output.ref_ex_post)
    output.ruv = (output.avg_ref_ex_post - output.avg_fcst_ex_post) / (output.avg_ref_ex_post - output.avg_obs_ex_post)

    if verbose:
        print('Alpha: %.3f   RUV: %.2f' % (alpha, output.ruv))

    return output
