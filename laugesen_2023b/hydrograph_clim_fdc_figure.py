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


# TODO: should be all the daily flow in each month, rather than an average of the first week
# TODO: make this more flexible to handle different date periods and start/end lead times
# Create a DataFrame with the correct date index for the daily data of the first week of each month
# def generate_results(obs):
#     start_date = "1991-01-01"
#     dates_for_each_month = [pd.date_range(start=f"{year}-{month}-01", periods=7) for year in range(1991, 1991 + (len(obs) // 84) + 1) for month in range(1, 13)]
#     correct_first_week_dates = [date for sublist in dates_for_each_month for date in sublist][:len(obs)]
#     df = pd.DataFrame({"Date": correct_first_week_dates, "Flow_m3s": obs})

#     results = {
#         'obs_dataframe': df
#     }
#     return results

def generate_results(obs, start_date="1991-01-01"):
    # Calculate the number of months based on the obs length
    num_months = len(obs) // 7
    
    # Create a full date range from start_date to the inferred end date
    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=num_months)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a DataFrame with all days and NaN values for Flow_m3s
    df = pd.DataFrame({"Date": all_dates, "Flow_m3s": np.nan})
    
    # Generate dates for the first week of each month within the range
    first_week_dates = [pd.date_range(start=f"{year}-{month}-01", periods=7) for year in range(pd.Timestamp(start_date).year, end_date.year + 1) for month in range(1, 13)]
    first_week_dates_flat = [date for sublist in first_week_dates for date in sublist if date in all_dates][:len(obs)]
    
    # Assign observed values for the first week of each month
    mask = df['Date'].isin(first_week_dates_flat)
    df.loc[mask, "Flow_m3s"] = obs

    results = {
        'obs_dataframe': df
    }
    return results


def generate_figure(results, metadata, color=LINE_COLORS['dark_blue']):

    df = results['obs_dataframe']

    plt.rcParams['font.family'] = "calibri"
    plt.rcParams['font.size'] = "12.5"

    fig = plt.figure(figsize=(11.5, 8.5))

    # Panel 1: Hydrograph spanning the full width at the top
    ax1 = fig.add_subplot(2, 1, 1)
    df.set_index('Date')['Flow_m3s'].plot(ax=ax1, color=color)
    ax1.set_title("Daily Streamflow (first Week of each month)")
    ax1.set_ylabel(r"Streamflow ($m^3/s$)")
    ax1.set_xlabel("Time")
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, va='top')

    # Panel 2: Monthly climatology (bottom left)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month 
    monthly_data = df.groupby('Month')
    monthly_daily_data = [group['Flow_m3s'] for name, group in monthly_data]
    monthly_daily_data = pd.DataFrame(monthly_daily_data).T
    monthly_daily_data.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    ax2 = fig.add_subplot(2, 2, 3)
    monthly_daily_data.boxplot(ax=ax2, grid=False, boxprops=dict(color=color), showfliers=False, medianprops=dict(color=color), whiskerprops=dict(color=color), capprops=dict(color=color))

    max_vals = monthly_daily_data.max()
    max_pos = list(range(1, 13))
    ax2.scatter(max_pos, max_vals, color=color, marker='o', s=15, label="Max Values")

    ax2.set_title("Monthly climatology")
    ax2.set_ylabel(r"Daily streamflow ($m^3/s$)")
    ax2.set_xlabel("Month")
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, va='top')   

    # Panel 3: Flow Duration Curve (bottom right)

    # TODO: move this into the generate_results function
    # Prepare data for flow duration curve
    flow = df['Flow_m3s'].dropna().values
    sorted_flows = np.sort(flow)[::-1]
    exceedance_probabilities = np.arange(1., len(sorted_flows) + 1) / len(sorted_flows) * 100

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(exceedance_probabilities, sorted_flows, color=color)
    ax3.set_xlim((0, 100))
    ax3.set_yscale("log")
    ax3.set_title("Flow duration curve")
    ax3.set_xlabel("Exceedance Probability (%)")
    ax3.set_ylabel(r"Streamflow ($m^3/s$)")
    ax3.text(0.95, 0.95, '(c)', transform=ax3.transAxes, va='top')

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    return fig
