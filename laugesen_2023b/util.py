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

# TODO: store all RUV results, and the RUV timeseries seperatly
# TODO: write results out to disk
# TODO: update legends so it says k= and alphas=
# TODO: add distinct colours and line styles
# TODO: create repo for muthre_data

import sys
import os
from matplotlib import pyplot as plt

sys.path.append('../..')

from muthre_results.muthre_results import load_from_csv, save_to_pickle


def load_data(awrc, start_lt, end_lt, scenario='muthre', input_path='../../muthre_results/muthre_csv'):
    print('Loading data')
    results = load_from_csv(awrc, start_lt, end_lt, scenario, input_path)
    return results['obs'], results['fcst'], results['clim']


def gen_damage_function_fig(results, metadata, ax):
    results['damages_results'].plot(ax=ax)
    ax.axvline(results['max_obs'], color='red', linewidth=0.5, alpha=0.5, linestyle='dotted', label='Max obs')
    ax.set_title(metadata['damages_title'], fontsize='medium')
    ax.set_xlabel(r'Streamflow ($m^3/s$)')
    ax.set_ylabel('Damages ($)')
    ax.legend()


def create_panel():
    plt.rcParams['figure.figsize'] = (11.5, 6)
    plt.rcParams['font.family'] = "calibri"
    plt.rcParams['font.size'] = "12.5"

    colors = {
        'black': '#000000',
        'light_orange': '#E69F00',
        'light_blue': '#56B4E9',
        'green': '#009E73',
        'yellow': '#F0E442',
        'dark blue': '#0072B2',
        'dark_orange': '#D55E00',
        'pink': '#CC79A7'
    } 

    fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
    plt.subplots_adjust(wspace=0.22, bottom=0.23)
    return fig, axes


def save_figure(fig, metadata, output_path='figures'):
    print('\tSaving figure')

    fig.savefig(os.path.join(output_path, 
                '%s_%s_LT%d-%d.png' % (metadata['figure_name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt'])),
                dpi=600, bbox_inches='tight', pad_inches=0.1)


# save everything to csv and json files and then zip it all up
def save_results(output):
    print('\tSaving output')
    pass
