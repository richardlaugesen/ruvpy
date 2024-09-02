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

# TODO: write results out to disk
# TODO: create repo for muthre_data

import sys
import os
import pickle
import bz2
import time
from logging import exception

from matplotlib import pyplot as plt

sys.path.append('../..')

from muthre_results.muthre_results import load_from_csv, save_to_pickle


LINE_COLORS = {
    'black': '#000000',
    'light_orange': '#E69F00',
    'light_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'dark_blue': '#0072B2',
    'dark_orange': '#D55E00',
    'pink': '#CC79A7'
} 


LINE_STYLES = ['-', '--', ':', '-.', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]


def load_data(awrc, start_lt, end_lt, area, scenario='muthre', input_path='../../muthre_results/muthre_csv'):
    print('Loading data')
    results = load_from_csv(awrc, start_lt, end_lt, scenario, input_path)
    results = mm_to_m3s(results, area)
    return results['obs'], results['fcst'], results['clim']

def restore_data(filepath):
    print('Restoring data')
    with bz2.BZ2File(filepath, 'rb') as f:
        return pickle.load(f)


def mm_to_m3s(results, area):
    factor = area * 1000 / (60 * 60 * 24)
    results['obs'] = results['obs'] * factor
    results['fcst'] = results['fcst'] * factor
    results['clim'] = results['clim'] * factor
    return results


def gen_damage_function_fig(damages_results, metadata, param_name, max_obs, ax, line_colors, line_styles):
    for i, column in enumerate(damages_results.columns):
        line_style = line_styles[i % len(line_styles)]
        
        if param_name:
            if param_name == 'k':
                label = r'%s = %.2f' % (param_name, column)
            else:
                label = r'%s = %.0f' % (param_name, column)
        else:
            label = column

        if type(line_colors) is str:
            line_color = line_colors
        else:
            line_color = line_colors[i % len(line_colors)]

        ax.plot(damages_results.index, damages_results[column], 
                linewidth=1, alpha=1, color=line_color, linestyle=line_style, 
                label=label)

    ax.axvline(max_obs, color='red', linewidth=0.5, alpha=0.3, linestyle='dotted', label='Max obs')
    ax.set_title(metadata['damages_title'], fontsize='medium')
    ax.set_xlabel(r'Streamflow ($m^3/s$)')
    ax.set_ylabel('Damages ($)')
    ax.legend()


def create_panel(panels=2):
    if panels == 2:
        plt.rcParams['figure.figsize'] = (11.5, 6)
    elif panels == 3:
        plt.rcParams['figure.figsize'] = (17.4, 6)
    else:
        print('Only 2 or 3 panel figures are implemented')

    plt.rcParams['font.family'] = "calibri"
    plt.rcParams['font.size'] = "12.5"

    fig, axes = plt.subplots(1, panels, sharey=False, sharex=False)
    plt.subplots_adjust(wspace=0.22, bottom=0.23)
    return fig, axes


def save_figure(fig, metadata, output_path='figures'):
    print('\tSaving figure')

    fig.savefig(os.path.join(output_path, 
                '%s_%s_LT%d-%d.png' % (metadata['figure_name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt'])),
                dpi=600, bbox_inches='tight', pad_inches=0.1)

    fig.savefig(os.path.join(output_path,
                '%s_%s_LT%d-%d.pdf' % (metadata['figure_name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt'])),
                bbox_inches='tight', pad_inches=0.1)

    fig.savefig(os.path.join(output_path,
                '%s_%s_LT%d-%d.svg' % (metadata['figure_name'], metadata['awrc'], metadata['start_lt'], metadata['end_lt'])),
                bbox_inches='tight', pad_inches=0.1)


def save_results(output, output_path='figures'):
    print('\tSaving output')

    filepath = os.path.join(output_path, '%s_%s_LT%d-%d.pkl.bz2' % (output['figure_name'], output['awrc'], output['start_lt'], output['end_lt']))
    with bz2.BZ2File(filepath, 'wb') as f:
        pickle.dump(output, f)

    filepath = os.path.join(output_path, '%s_ruv_only_%s_LT%d-%d.csv' % (output['figure_name'], output['awrc'], output['start_lt'], output['end_lt']))
    output['ruv_only'].to_csv(os.path.join(filepath), header=['k=%.4f' % v for v in output['all_results'].keys()], index=True, index_label='alpha')


def progressor(curr_num, total_num, start_time):
    progress = curr_num / total_num * 100
    print(curr_num, total_num, progress, int(progress) % 5)
    if curr_num > 0 and (total_num < 20 or int(progress) % 5 == 0 or curr_num == 2):
        curr_time = time.time()
        remaining_seconds = (curr_time - start_time) * (total_num - curr_num) / curr_num  
        remaining_minutes = remaining_seconds / 60
        remaining_hours = remaining_minutes / 60
        if remaining_hours >= 1:
            print('\t\t%.0f%% complete with %.1f hours remaining' % (progress, remaining_hours))
        elif remaining_minutes >= 1:
            print('\t\t%.0f%% complete with %.1f minutes remaining' % (progress, remaining_minutes))
        else:
            print('\t\t%.0f%% complete with %.0f seconds remaining' % (progress, remaining_seconds))
