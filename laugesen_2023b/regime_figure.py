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

import sys
import numpy as np
import pandas as pd
from scipy import stats
import time
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

sys.path.append('..')

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *
from util import *

def regime_figure(awrc, name, start_lt, end_lt, area, alpha_step=0.2, parallel_nodes=8, restore_data_filepath=None, verbose=False):
    metadata = {
        'awrc': awrc,
        'name': name,
        'start_lt': start_lt,
        'end_lt': end_lt,
        'parallel_nodes': parallel_nodes,
        'alpha_step': alpha_step,
        'area': area
    }

    # load data
    obs, fcst, clim = load_data(awrc, start_lt, end_lt, area)

    print('Starting regime figure...')
    metadata['figure_name'] = 'regime'

    # generate results
    if restore_data_filepath is None:
        results = generate_results(obs, fcst, clim, alpha_step, parallel_nodes, verbose)
    else:
        output = restore_data(restore_data_filepath)
        results = output

    # store all output
    if restore_data_filepath is None:
        output = {
            'obs': obs,
            'fcst': fcst,
            'clim': clim
        }
        output.update(metadata)
        output.update(results)
        save_results(output)

    # generate and save figure
    fig = generate_figure(results, metadata)
    save_figure(fig, metadata)

    return output


def generate_results(obs, fcst, ref, alpha_step, parallel_nodes, verbose):
    print('\tGenerating results')

    alphas = np.arange(alpha_step, 1, alpha_step)
    print('%d alpha values to simulate' % len(alphas))

    target_unity_risk_aversion = 0.15
    max_damages = 10000
    target_risk_premium = risk_aversion_coef_to_risk_premium(target_unity_risk_aversion, 1) 
    adjusted_risk_aversion = risk_premium_to_risk_aversion_coef(target_risk_premium, max_damages)

    # Define decision context
    decision_definition = {
        'econ_pars': alphas,
        'target_unity_risk_aversion': target_unity_risk_aversion,
        'target_risk_premium': target_risk_premium,
        'adjusted_risk_aversion': adjusted_risk_aversion,
        'utility_function': [cara, {'A': adjusted_risk_aversion}],
        'economic_model': [cost_loss, cost_loss_analytical_spend],
        'decision_method': 'optimise_over_forecast_distribution',
        'decision_thresholds': None,
    }

    # Define different damage functions
    damage_functions = {}

    flow_damage_pairs = [
        (0, max_damages),
        (np.nanquantile(obs, 0.1), 0),      # bottom 10% of flow
        (np.nanquantile(obs, 0.95), 0),     # top 5% of flow
        (np.nanquantile(obs, 0.998), max_damages)
    ]
    damage_functions['low and high'] = [user_defined, {'interpolator': user_defined_interpolator(flow_damage_pairs)}]

    flow_damage_pairs = [
        (0, 0),
        (np.nanquantile(obs, 0.1), 0),
        (np.nanquantile(obs, 0.95), 0),
        (np.nanquantile(obs, 0.998), max_damages)
    ]
    damage_functions['only high'] = [user_defined, {'interpolator': user_defined_interpolator(flow_damage_pairs)}]

    flow_damage_pairs = [
        (0, max_damages),
        (np.nanquantile(obs, 0.1), 0),
        (np.nanquantile(obs, 0.95), 0),
        (np.nanquantile(obs, 0.998), 0)
    ]
    damage_functions['only low'] = [user_defined, {'interpolator': user_defined_interpolator(flow_damage_pairs)}]

    damage_functions['only high (logistic)'] = [logistic, {'k': 0.2, 'A': max_damages, 'threshold': np.nanquantile(obs, 0.991)}]

    # print out estimated time
    est_time = 0.0012/60 * len(damage_functions) * len(alphas) * len(obs)
    if est_time > 60:
        est_time /= 60
        time_str = '%.2f hours' % est_time
    else:
        time_str = '%.2f minutes' % est_time
    print('Estimated calculate time is %s' % time_str)

    # Calculate RUV for the different thresholds for damage function figure
    start_time = time.time()
    results = {}
    ruv_only = {}

    for i, (name, damage_function) in enumerate(damage_functions.items()):
        progressor(i, len(damage_functions), start_time)

        decision_definition['damage_function'] = damage_function
        results[name] = relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=parallel_nodes, verbose=verbose)
        ruv_only[name] = results[name]['ruv']

    ruv_only = pd.DataFrame(ruv_only, index=decision_definition['econ_pars'])

    # Generate streamflow-damage values for the different thresholds for damage function figure
    streamflow = np.arange(0, np.nanmax(obs) * 1.3, 0.01)
    streamflow_damages = pd.DataFrame(index=streamflow, columns=list(damage_functions.keys()))
    for name, damage_function in damage_functions.items():
        damage_fnc_mth = damage_function[0]
        damage_fnc_params = damage_function[1]
        damage_fnc = damage_fnc_mth(damage_fnc_params)
        streamflow_damages[name] = damage_fnc(streamflow)

    output = {
        'ruv_only': ruv_only, 
        'all_results': results,
        'damages_results': streamflow_damages,        
        'max_obs': np.nanmax(obs),
        'decision_definition': decision_definition,
        'damage_functions': damage_functions,
        'execution_time_min': (time.time() - start_time) / 60
    }
    return output


def generate_figure(results, metadata):
    print('\tGenerating figure')

    fig, axes = create_panel()
    # fig.suptitle('Impact of damage function regime, %s (%s) for lead-times %d to %d' 
    #              % (metadata['name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt']), 
    #              fontweight='semibold', fontsize='large')

    metadata['damages_title'] = 'Damage functions of different regimes'
    line_colors = list(LINE_COLORS.values())
    gen_damage_function_fig(results['damages_results'], metadata, None, results['max_obs'], axes[0], line_colors, LINE_STYLES)

    ax = axes[1]
    for i, column in enumerate(results['ruv_only'].columns):
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        line_color = line_colors[i % len(LINE_COLORS)]

        ax.plot(results['ruv_only'].index, results['ruv_only'][column], 
                linewidth=1, alpha=1, color=line_color, linestyle=line_style, 
                label='%s' % column)
 
    plt.ylim(0, 1)
    plt.xlim((0, 1))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Forecast value (RUV)')
    ax.set_title(r'Forecast value for different regimes', fontsize='medium')
    ax.legend()

    # Add labels to the top right corner of each panel
    axes[0].text(0.05, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
    axes[1].text(0.05, 0.95, '(b)', horizontalalignment='left', verticalalignment='top', transform=axes[1].transAxes)

    return fig


def main():
    parallel_nodes = 8
    alpha_resolution = 0.2 #02
    verbose = False
    #restore_data_filepath = None
    restore_data_filepath = 'figures/regime_405209_LT1-7.pkl.bz2'

    # awrc = '405219'
    # name = 'Goulburn River at Dohertys'
    # area = 700.2

    # awrc = '401012'
    # name = 'Murray River at Biggera'
    # area = 1257

    awrc = '405209'
    name = 'Acheron River at Taggerty'
    area = 629.4

    start_lt = 1
    end_lt = 7

    shape_output = regime_figure(awrc, name, start_lt, end_lt, area, alpha_step=alpha_resolution, parallel_nodes=parallel_nodes, restore_data_filepath=restore_data_filepath, verbose=verbose)

    print('%.2f minutes' % shape_output['execution_time_min'])


if __name__ == "__main__":
    main()
