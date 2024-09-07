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

def location_figure(awrc, name, start_lt, end_lt, area, select_alphas, select_thresholds, a_step=0.02, q_step=0.2, parallel_nodes=8, restore_data_filepath=None, show_percentiles=False, verbose=False):
    metadata = {
        'awrc': awrc,
        'name': name,
        'start_lt': start_lt,
        'end_lt': end_lt,
        'parallel_nodes': parallel_nodes,
        'q_step': q_step,
        'area': area,
        'select_alphas': select_alphas
    }

    # load data
    obs, fcst, clim = load_data(awrc, start_lt, end_lt, area)

    print('Starting location figure...')
    metadata['figure_name'] = 'location'

    # generate results
    if restore_data_filepath is None:
        results = generate_results(obs, fcst, clim, select_thresholds, a_step, q_step, parallel_nodes, verbose)
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
    fig = generate_figure(results, obs, metadata, select_thresholds, show_percentiles)
    save_figure(fig, metadata)

    return output


def generate_results(obs, fcst, ref, select_thresholds, a_step, q_step, parallel_nodes, verbose):
    print('\tGenerating results')

    alphas = np.arange(a_step, 1, a_step)
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
        #'decision_method': 'critical_probability_threshold_equals_par',
        'decision_thresholds': None,
        'damage_function': [logistic, {'k': 0.2, 'A': max_damages, 'threshold': np.nanquantile(obs, 0.99)}]
    }

    thresholds = np.arange(0, np.nanmax(obs) * 1.3, q_step)
    print('%d threshold values to simulate' % len(thresholds))

    # Calculate RUV for the different thresholds for damage function figure
    start_time = time.time()
    results = {}
    ruv_only = {}

    # print out estimated time
    est_time = 0.0012/60 * len(thresholds) * len(alphas) * len(obs)
    if est_time > 60:
        est_time /= 60
        time_str = '%.2f hours' % est_time
    else:
        time_str = '%.2f minutes' % est_time
    print('Estimated calculate time is %s' % time_str)

    # calculate RUV for each threshold
    for i, threshold in enumerate(thresholds):
        progressor(i, len(thresholds), start_time)

        decision_definition['damage_function'][1]['threshold'] = threshold
        results[threshold] = relative_utility_value(obs, fcst, ref, decision_definition, parallel_nodes=parallel_nodes, verbose=verbose)
        ruv_only[threshold] = results[threshold]['ruv']

    ruv_only_df = pd.DataFrame(ruv_only, index=decision_definition['econ_pars']).T

    # Generate streamflow-damage values for the different thresholds for damage function figure
    if select_thresholds is None:
        select_thresholds = thresholds[[0, int(len(thresholds)/4), int(len(thresholds)/2), 3*int(len(thresholds)/4), len(thresholds)-1]]

    streamflow = np.arange(0, np.nanmax(obs) * 1.3, 0.01)
    damage_fnc, params = decision_definition['damage_function']
    streamflow_damages = pd.DataFrame(index=streamflow, columns=select_thresholds)
    for threshold in select_thresholds:
        params['threshold'] = threshold
        streamflow_damages[threshold] = damage_fnc(params)(streamflow)

    output = {
        'ruv_only': ruv_only_df,
        'all_results': results, 
        'damages_results': streamflow_damages,
        'max_obs': np.nanmax(obs),
        'decision_definition': decision_definition,
        'thresholds': thresholds,
        'select_thresholds': select_thresholds,
        'execution_time_min': (time.time() - start_time) / 60
    }
    return output


def generate_figure(results, obs, metadata, select_thresholds, show_percentiles=True):
    print('\tGenerating figure')

    fig, axes = create_panel(3)
    # fig.suptitle('Impact of damage function location, %s (%s) for lead-times %d to %d' 
    #              % (metadata['name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt']), 
    #              fontweight='semibold', fontsize='large')

    left_panel_color = LINE_COLORS['dark_blue']
    right_panel_color = LINE_COLORS['dark_orange']

    # damage function panel
    metadata['damages_title'] = 'Three damage functions with different locations thresholds'
    gen_damage_function_fig(results['damages_results'], metadata, r'$q_\tau$', results['max_obs'], axes[0], left_panel_color, LINE_STYLES, select_thresholds)

    # value diagrams panel
    if select_thresholds is None:
        select_thresholds = results['select_thresholds']

    ax = axes[1]
    for i, threshold in enumerate(select_thresholds):
        line_style = LINE_STYLES[i % len(LINE_STYLES)]

        ax.plot(results['ruv_only'].T.index, results['ruv_only'].T[threshold],
                linewidth=1, alpha=1, color=left_panel_color, linestyle=line_style,
                label=r'$q_\tau$ = %.0f' % threshold)

    ax.axhline(0, color='grey', linewidth=0.5, alpha=0.3, linestyle='--', label='_hidden')

    ax.set_ylim(0, 1)
    ax.set_xlim((0, 1))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Forecast value (RUV)')
    ax.set_title('Value diagrams for the three damage functions', fontsize='medium')
    ax.legend()

    # continuous variation of threshold
    ax = axes[2]
    for i, column in enumerate(metadata['select_alphas']):
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(results['ruv_only'].index, results['ruv_only'][column], 
                linewidth=1, alpha=1, color=right_panel_color, linestyle=line_style, 
                label=r'$\alpha$ = %.1f' % column)

    ax.set_ylim(0, 1)
    ax.set_xlim((0, 150))

    ax.set_xlabel(r'Damage function threshold $q_\tau$ ($m^3/s$)')
    ax.set_ylabel('Forecast value (RUV)')
    ax.set_title(r'Forecast value for different values of $\alpha$', fontsize='medium')
    ax.legend()

    # Add labels to the top right corner of each panel
    axes[0].text(0.05, 0.95, '(a)', horizontalalignment='left', verticalalignment='top', transform=axes[0].transAxes)
    axes[1].text(0.95, 0.95, '(b)', horizontalalignment='right', verticalalignment='top', transform=axes[1].transAxes)
    axes[2].text(0.05, 0.95, '(c)', horizontalalignment='left', verticalalignment='top', transform=axes[2].transAxes)

    def flow_to_percentile(x):
        return stats.percentileofscore(obs[~np.isnan(obs)], x)

    def percentile_to_flow(x):
        return np.nanpercentile(obs, x)

    # TODO: this is not working, obs seems to be [] rather than the actual obs data
    if show_percentiles:
        secax = ax.secondary_xaxis(-0.15, functions=(flow_to_percentile, percentile_to_flow))
        secax.xaxis.set_major_locator(ticker.FixedLocator([50, 90, 99, 99.9, 100]))
        secax.set_xlabel('Percentile')

    return fig


def main():
    parallel_nodes = 8
    alpha_resolution = 0.02
    threshold_resolution = 1
    select_alphas = np.array([0.1, 0.5, 0.9])
    verbose = False
    show_percentiles = False
    select_thresholds = [50, 100, 150]
    #restore_data_filepath = None
    restore_data_filepath = 'figures/location_405209_LT1-7.pkl.bz2'

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

    shape_output = location_figure(awrc, name, start_lt, end_lt, area, select_alphas, a_step=alpha_resolution, q_step=threshold_resolution, parallel_nodes=parallel_nodes, restore_data_filepath=restore_data_filepath, show_percentiles=show_percentiles, select_thresholds=select_thresholds, verbose=verbose)

    print('%.2f minutes' % shape_output['execution_time_min'])


if __name__ == "__main__":
    main()
