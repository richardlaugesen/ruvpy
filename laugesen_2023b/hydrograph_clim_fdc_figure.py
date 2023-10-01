import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

from util import *


def hydrograph_clim_fdc_figure(awrc, name, start_lt=1, end_lt=7, verbose=False):
    metadata = {
        'awrc': awrc,
        'name': name,
        'start_lt': start_lt,
        'end_lt': end_lt,
    }

    # load data
    obs, fcst, clim = load_data(awrc, start_lt, end_lt)

    print('Starting shape figure...')
    metadata['figure_name'] = 'hydrograph_clim_fdc'

    # generate results
    results = generate_results(obs)

    # generate and save figure
    fig = generate_figure(results, metadata)
    save_figure(fig, metadata)

    # store all output
    output = {
        'obs': obs
    }
    output.update(metadata)
    output.update(results)
    save_results(output)

    return output


# TODO: make this more flexible to handle different date periods and start/end lead times
# Create a DataFrame with the correct date index for the daily data of the first week of each month
def generate_results(obs):
    start_date = "1991-01-01"
    dates_for_each_month = [pd.date_range(start=f"{year}-{month}-01", periods=7) for year in range(1991, 1991 + (len(obs) // 84) + 1) for month in range(1, 13)]
    correct_first_week_dates = [date for sublist in dates_for_each_month for date in sublist][:len(obs)]
    df = pd.DataFrame({"Date": correct_first_week_dates, "Flow_m3s": obs})

    results = {
        'obs_dataframe': df
    }
    return results


# TODO: update all the text
def generate_figure(results, metadata, color='#0072B2'):

    df = results['obs_dataframe']

    plt.rcParams['font.family'] = "calibri"
    plt.rcParams['font.size'] = "12.5"

    fig = plt.figure(figsize=(11.5, 8.5))

    # Panel 1: Hydrograph spanning the full width at the top
    ax1 = fig.add_subplot(2, 1, 1)
    df.set_index('Date')['Flow_m3s'].plot(ax=ax1, color=color)
    ax1.set_title("Daily Streamflow for the First Week of Each Month")
    ax1.set_ylabel("Streamflow (m3/s)")
    ax1.set_xlabel("Time")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Prepare data for the climatology
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    monthly_avg = df.groupby(['Year', 'Month']).mean()['Flow_m3s']

    # Panel 2: Climatology of Monthly Flow for the first week (bottom left)
    ax2 = fig.add_subplot(2, 2, 3)
    monthly_avg.unstack(level=1).boxplot(ax=ax2, grid=False, boxprops=dict(color=color), medianprops=dict(color=color), whiskerprops=dict(color=color), capprops=dict(color=color))
    ax2.set_title("Climatology of Monthly Flow (First Week)")
    ax2.set_ylabel("Daily average streamflow (m3/s)")
    ax2.set_xlabel("Month")
    ax2.set_xticklabels(calendar.month_abbr[3:] + calendar.month_abbr[1:3])

    # Prepare data for flow duration curve
    sorted_flows = np.sort(df['Flow_m3s'].values)[::-1]
    exceedance_probabilities = np.arange(1., len(sorted_flows) + 1) / len(sorted_flows) * 100

    # Panel 3: Flow Duration Curve (bottom right)
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(exceedance_probabilities, sorted_flows, color=color)
    ax3.set_yscale("log")
    ax3.set_title("Flow Duration Curve")
    ax3.set_xlabel("Exceedance Probability (%)")
    ax3.set_ylabel("Streamflow (m3/s, log scale)")
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    return fig
