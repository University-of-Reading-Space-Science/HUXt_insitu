# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:09:58 2025

@author: vy902033
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 08:51:29 2025

Analyse and compare InSitu-HUXt forecasts with WSA-HUXt
"""

# <codecell> get the OMNI data for the whole itnerval - both forecasts and validation


import datetime
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import glob
from astropy.time import Time
from sunpy.coordinates import sun


#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA
import huxt_insitu as His

import insitu
from taylor_diagram import combined_taylor_target_diagram, plot_taylor_target_diagram, compute_taylor_target_stats
import hse  # High-Speed Enhancement detection and analysis

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Time range for analysis
start_time = datetime.datetime(2008,1,1)
stop_time = datetime.datetime(2022,12,31)#datetime.datetime(2023,1,1)

# Flag to control whether to compute forecast time series or load from saved files
compute_forecast_timeseries_now = False

# Final per-forecast HSE counting can be very expensive; by default load cached CSVs.
recompute_per_forecast_hse_counts_now = False
per_forecast_hse_progress_every = 250

# Avoid blocking at script end unless explicitly requested.
wait_for_enter_before_exit = False

# HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs

# Forecast lead time configuration
min_lt = 1              # Minimum lead time (days)
max_lt = 27             # Maximum lead time (days)
plot_lead_time = 4      # Lead time used for plots (HSE, DTW, Taylor/Target diagrams, etc.)

# ICME removal parameters
icme_list = 'CaneRichardson'  # 'DONKI' or 'CaneRichardson'
pre_icme_buffer = 0.2   # days before ICME start
post_icme_buffer = 1    # days after ICME end
# Fill values for ICME removal if interpolation isn't possible
vsw_fill = 450 *u.km/u.s
bx_fill = 0

#example forecast time series
example_forecast_dates = [
    datetime.datetime(2017, 4, 15),
    datetime.datetime(2018, 5, 20),
    datetime.datetime(2019, 7, 15),  # Legacy example retained for downstream sections
    datetime.datetime(2020, 4, 15),
    datetime.datetime(2021, 11, 3),
]

#2d example plots
example_forecast_date = datetime.datetime(2019, 7, 15)

# High-Speed Enhancement (HSE) detection parameters
hse_rise_threshold = 150        # km/s speed increase over rise_hours
hse_rise_hours = 48             # hours over which to measure speed increase
hse_min_separation = 24         # minimum hours between separate HSE events
hse_smoothing_window = 8        # hours for smoothing window
hse_max_matching_dt = 72        # maximum hours for matching observed/forecast events

# Dynamic Time Warping (DTW) parameters
dtw_window_points = 30          # Sakoe-Chiba band constraint (in 8-hour points, ~10 days)
dtw_demo_start = datetime.datetime(2017, 1, 1)   # Demo period start for DTW visualization
dtw_demo_end = datetime.datetime(2017, 7, 1)     # Demo period end for DTW visualization

# HSE demonstration figure parameters (should match dtw_demo period)
hse_demo_start = datetime.datetime(2017, 1, 1)   # Demo period start for HSE visualization
hse_demo_end = datetime.datetime(2017, 7, 1)     # Demo period end for HSE visualization

# Consistent color scheme across all plots
color_insitu = '#2E86AB'  # Blue for InSitu-HUXt
color_wsa = '#A23B72'     # Pink/magenta for WSA-HUXt
color_obs = 'k'           # Black for observations



# Directory structure.  Environment overrides make the expensive raw-forecast
# rebuild possible without baking an author's filesystem layout into the code.
cwd = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(cwd)
figdir = os.getenv('HUXT_FIGURE_DIR', os.path.join(repo_root, 'figures'))
datadir = os.getenv('HUXT_DATA_DIR', os.path.join(repo_root, 'data'))
icmelist_path = os.path.join(datadir, 'Richardson_Cane_Porcessed_ICME_list.csv')
ishuxt_dir = os.getenv('HUXT_INSITU_RAW_DIR', os.path.join(datadir, 'raw', 'insitu_huxt'))
wsahuxt_dir = os.getenv('HUXT_WSA_RAW_DIR', os.path.join(datadir, 'raw', 'wsa_huxt'))
omni_cache = os.getenv('HUXT_OMNI_CACHE', os.path.join(datadir, 'omni_1hour_2007_2023.csv'))
os.makedirs(figdir, exist_ok=True)




# <codecell> get the OMNI data for the whole itnerval - both forecasts and validation

dl_starttime = start_time - datetime.timedelta(days=28)
dl_endtime = stop_time + datetime.timedelta(days=28)

if os.path.exists(omni_cache):
    print(f"Loading cached OMNI data from {omni_cache}")
    omni = pd.read_csv(omni_cache, parse_dates=['datetime'])
else:
    print("OMNI cache not found; downloading from NASA CDAWeb via SunPy...")
    omni = Hin.get_omni(dl_starttime, dl_endtime)
    omni.to_csv(omni_cache, index=False)
    print(f"Cached OMNI data at {omni_cache}")




# <codecell> remove ICMEs
#create a copy of the OMNI data for ICME removal, before any end padding
omni_noicmes = omni.copy()

#load the DONKI ICME list
if icme_list == 'DONKI':
    icmes = Hin.get_DONKI_ICMEs(dl_starttime, dl_endtime)
elif icme_list == 'CaneRichardson':
    icmes = insitu.ICMElist(icmelist_path)


# #remove all ICMEs
# omni_noicmes =  Hin.remove_ICMEs(omni, icmes, interpolate = True, 
#                  icme_buffer = icme_buffer, interp_buffer = sw_buffer,
#                  params = ['V', 'BX_GSE'], fill_vals = None)

params = ['V', 'BX_GSE']
# first remove all ICMEs and add NaNs to the required parameters
icmes['shock_mjd'] = Time(icmes['Shock_time'].to_numpy()).mjd
icmes['end_mjd'] = Time(icmes['ICME_end'].to_numpy()).mjd
    
for i in range(0, len(icmes)):

    icme_start = icmes['shock_mjd'][i] - pre_icme_buffer
    icme_stop = icmes['end_mjd'][i] + post_icme_buffer 

    mask_icme = ((omni_noicmes['mjd'] >= icme_start) &
                 (omni_noicmes['mjd'] <= icme_stop))

    if any(mask_icme):
        print('removing ICME #' + str(i))
        for param in params:
            omni_noicmes.loc[mask_icme, param] = np.nan


#now interp through all datagaps
# omni_noicmes = omni_noicmes.set_index('datetime')
# omni_noicmes[['V', 'BX_GSE']] = omni_noicmes[['V', 'BX_GSE']].interpolate(method='time').ffill().bfill()
# omni_noicmes = omni_noicmes.reset_index()



    

    
    
# <codecell> Compute or load forecast timeseries for different lead times
# =============================================================================
# If compute_forecast_timeseries_now = True, compile time series from raw files
# and save to data directory. Otherwise, load pre-computed files.
# =============================================================================

if compute_forecast_timeseries_now:
    print("Computing forecast time series from raw files...")
    
    # InSitu-HUXt forecasts
    #the list of forecasts at different lead times
    for_list = []
    for n in range(min_lt-1, max_lt):
        for_list.append([])
        
    dates = pd.date_range(start=start_time, end=stop_time, freq="D")    

    for ftime in dates:    
        print('Analysing InSitu-HUXt for ' + ftime.strftime("%Y-%m-%d"))
        
        #set up the directory path
        yr_str = str(ftime.year)
        mn_str = f"{ftime.month:02}"
        dy_str = f"{ftime.day:02}"
        
        filename_wildcard = 'InSitu_Earth_' + yr_str + mn_str + dy_str + '.dat'
        filepath_wildcard = os.path.join(ishuxt_dir, yr_str, filename_wildcard)
        
        # Find matching files
        matching_files = glob.glob(filepath_wildcard)
        
        #if the file exists, read in the sub-Earth values
        if matching_files:
            print('Loading: ' + ftime.strftime('%Y-%m-%d'))
            filepath = matching_files[0]
            
            ts = pd.read_csv(filepath, index_col=0, parse_dates=["time"] )
            
            #process the HUXt input into forecasts of different lead times
            for ifor in range(0,len(for_list)):
                start_window = ftime + datetime.timedelta(days = ifor + 1)
                stop_window = ftime + datetime.timedelta(days = ifor + 2.0)
                
                mask = ((ts['time'] >= start_window)  & (ts['time'] <= stop_window))
                for_list[ifor].append(ts.loc[mask])

    #combine the dataframes at each forecast lead time.
    combined_for_list = [pd.concat(df_list, ignore_index=True) for df_list in for_list]
    
    # WSA-HUXt forecasts for the same time period
    #the list of forecasts at different lead times
    for_wsa_list = []
    for n in range(min_lt-1, max_lt):
        for_wsa_list.append([])
        
    dates = pd.date_range(start=start_time, end=stop_time, freq="D")    
    for ftime in dates:
        print('Analysing WSA-HUXt for ' + ftime.strftime("%Y-%m-%d"))
        
        #set up the directory path
        yr_str = str(ftime.year)
        mn_str = f"{ftime.month:02}"
        dy_str = f"{ftime.day:02}"
        
        filename_wildcard = 'HUXt_Earth_' + yr_str + mn_str + dy_str + '.dat'
        filepath_wildcard = os.path.join(wsahuxt_dir, yr_str, filename_wildcard)
        
        # Find matching files
        matching_files = glob.glob(filepath_wildcard)
        
        #if the file exists, read in the sub-Earth values
        if matching_files:
            print('Loading: ' + ftime.strftime('%Y-%m-%d'))
            filepath = matching_files[0]
            
            ts = pd.read_csv(filepath, index_col=0, parse_dates=["time"] )
        
            #process the HUXt input into forecasts of different lead times
            for ifor in range(0,len(for_wsa_list)):
                start_window = ftime + datetime.timedelta(days = ifor + 1)
                stop_window = ftime + datetime.timedelta(days = ifor + 2.0)
                
                mask = ((ts['time'] >= start_window)  & (ts['time'] <= stop_window))
                for_wsa_list[ifor].append(ts.loc[mask])

    #combine the dataframes at each forecast lead time.
    combined_for_wsa_list = [pd.concat(df_list, ignore_index=True) for df_list in for_wsa_list]
    
    # Save the compiled time series to data directory
    print("\nSaving compiled forecast time series to data directory...")
    for lt in range(min_lt, max_lt+1):
        # Save InSitu-HUXt
        insitu_filename = f'insitu_huxt_forecasts_{lt}day_leadtime.csv'
        combined_for_list[lt-1].to_csv(os.path.join(datadir, insitu_filename), index=False)
        
        # Save WSA-HUXt
        wsa_filename = f'wsa_huxt_forecasts_{lt}day_leadtime.csv'
        combined_for_wsa_list[lt-1].to_csv(os.path.join(datadir, wsa_filename), index=False)
    
    print(f"Saved {max_lt - min_lt + 1} lead time files for each model to {datadir}")

else:
    print("Loading pre-computed forecast time series from data directory...")
    
    combined_for_list = []
    combined_for_wsa_list = []
    
    for lt in range(min_lt, max_lt+1):
        # Load InSitu-HUXt (MJD column is already saved in CSV)
        insitu_filename = f'insitu_huxt_forecasts_{lt}day_leadtime.csv'
        insitu_filepath = os.path.join(datadir, insitu_filename)
        df_insitu = pd.read_csv(insitu_filepath, parse_dates=['time'])
        combined_for_list.append(df_insitu)
        
        # Load WSA-HUXt (MJD column is already saved in CSV)
        wsa_filename = f'wsa_huxt_forecasts_{lt}day_leadtime.csv'
        wsa_filepath = os.path.join(datadir, wsa_filename)
        df_wsa = pd.read_csv(wsa_filepath, parse_dates=['time'])
        combined_for_wsa_list.append(df_wsa)
    
    print(f"Loaded {len(combined_for_list)} lead time files for each model from {datadir}")



# <codecell> Plot time series comparison

fig, axs = plt.subplots(3, 1, figsize=(14, 11))

ax = axs[0]
ax.plot(combined_for_list[plot_lead_time-1]['time'], 
        combined_for_list[plot_lead_time-1]['vsw'], color=color_insitu, label='InSitu-HUXt', linewidth=1.2, zorder=1)
ax.plot(omni_noicmes['datetime'], omni_noicmes['V'], color=color_obs, label='OMNI', alpha=0.7, linewidth=0.8, zorder=2)
ax.legend(loc='upper right')
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlim((start_time, stop_time))
ax.set_ylim((200, 900))
ax.set_title(f'(a) InSitu-HUXt {plot_lead_time}-day lead-time forecasts', fontsize=14)

ax = axs[1]
ax.plot(combined_for_wsa_list[plot_lead_time-1]['time'], 
        combined_for_wsa_list[plot_lead_time-1]['vsw'], color=color_wsa, label='WSA-HUXt', linewidth=1.2, zorder=1)
ax.plot(omni_noicmes['datetime'], omni_noicmes['V'], color=color_obs, label='OMNI', alpha=0.7, linewidth=0.8, zorder=2)
ax.legend(loc='upper right')
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlim((start_time, stop_time))
ax.set_ylim((200, 900))
ax.set_title(f'(b) WSA-HUXt {plot_lead_time}-day lead-time forecasts', fontsize=14)

# Third panel: first six months of 2017 with all three on the same plot
ax = axs[2]
zoom_year = 2017
zoom_start = datetime.datetime(zoom_year, 1, 1)
zoom_end = datetime.datetime(zoom_year, 6, 30, 23, 59, 59)

# Get data for zoom period
insitu_times = pd.to_datetime(combined_for_list[plot_lead_time-1]['time'])
mask_insitu_zoom = (insitu_times >= zoom_start) & (insitu_times <= zoom_end)

wsa_times = pd.to_datetime(combined_for_wsa_list[plot_lead_time-1]['time'])
mask_wsa_zoom = (wsa_times >= zoom_start) & (wsa_times <= zoom_end)

mask_omni_zoom = (omni_noicmes['datetime'] >= zoom_start) & (omni_noicmes['datetime'] <= zoom_end)

# Plot forecasts first, then observations on top
ax.plot(insitu_times[mask_insitu_zoom], 
        combined_for_list[plot_lead_time-1].loc[mask_insitu_zoom, 'vsw'], 
        color=color_insitu, label='InSitu-HUXt', linewidth=1.5, zorder=1)
ax.plot(wsa_times[mask_wsa_zoom], 
        combined_for_wsa_list[plot_lead_time-1].loc[mask_wsa_zoom, 'vsw'], 
        color=color_wsa, label='WSA-HUXt', linewidth=1.5, zorder=2)
ax.plot(omni_noicmes.loc[mask_omni_zoom, 'datetime'], omni_noicmes.loc[mask_omni_zoom, 'V'], 
        color=color_obs, label='OMNI', alpha=0.8, linewidth=1.2, zorder=3)

ax.legend(loc='upper right')
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlabel(f'Date ({zoom_year})')
ax.set_xlim((zoom_start, zoom_end))
ax.set_ylim((200, 900))
demo_month_range = f"{dtw_demo_start.strftime('%b')}-{dtw_demo_end.strftime('%b %Y')}"
ax.set_title(f'(c) {demo_month_range}: OMNI, InSitu-HUXt, and WSA-HUXt ({plot_lead_time}-day lead time)', fontsize=14)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

fig.tight_layout()
fig.savefig(os.path.join(figdir, f'timeseries_comparison_{plot_lead_time}day.pdf'), dpi=150, bbox_inches='tight')
print(f"Saved time series comparison as timeseries_comparison_{plot_lead_time}day.pdf")


# <codecell> Plot yearly comparison - observations, InSitu-HUXt, and WSA-HUXt on same axes
# =============================================================================
# Create a multi-panel figure with each year as a separate panel, showing
# observations and both forecasts overlaid for direct comparison
# =============================================================================

# Determine the years to plot
years_to_plot = list(range(start_time.year, stop_time.year))
n_years = len(years_to_plot)

# Calculate grid dimensions (aim for roughly 4 columns)
n_cols = 4
n_rows = int(np.ceil(n_years / n_cols))

fig_yearly, axs_yearly = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows), sharex=False, sharey=True)
axs_yearly = axs_yearly.flatten()  # Flatten for easy indexing

# Panel labels
panel_labels = [chr(ord('a') + i) for i in range(n_years)]  # 'a', 'b', 'c', ...

for i, year in enumerate(years_to_plot):
    ax = axs_yearly[i]
    
    # Define time range for this year
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year, 12, 31, 23, 59, 59)
    
    # Get observations for this year
    mask_omni = (omni_noicmes['datetime'] >= year_start) & (omni_noicmes['datetime'] <= year_end)
    
    # Get forecasts for this year
    insitu_times = pd.to_datetime(combined_for_list[plot_lead_time-1]['time'])
    mask_insitu = (insitu_times >= year_start) & (insitu_times <= year_end)
    
    wsa_times = pd.to_datetime(combined_for_wsa_list[plot_lead_time-1]['time'])
    mask_wsa = (wsa_times >= year_start) & (wsa_times <= year_end)
    
    # Plot InSitu-HUXt forecast
    ax.plot(insitu_times[mask_insitu], 
            combined_for_list[plot_lead_time-1].loc[mask_insitu, 'vsw'], 
            color=color_insitu, label='InSitu-HUXt', linewidth=0.8, alpha=0.8, zorder=1)
    
    # Plot WSA-HUXt forecast
    ax.plot(wsa_times[mask_wsa], 
            combined_for_wsa_list[plot_lead_time-1].loc[mask_wsa, 'vsw'], 
            color=color_wsa, label='WSA-HUXt', linewidth=0.8, alpha=0.8, zorder=2)
    
    # Plot observations (on top)
    ax.plot(omni_noicmes.loc[mask_omni, 'datetime'], omni_noicmes.loc[mask_omni, 'V'], 
            color=color_obs, label='OMNI', alpha=0.6, linewidth=0.6, zorder=3)
    
    # Formatting
    ax.set_xlim((year_start, year_end))
    ax.set_ylim((200, 850))
    ax.set_title(f'({panel_labels[i]}) {year}', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.3)
    
    # Only add y-label to leftmost panels
    if i % n_cols == 0:
        ax.set_ylabel('V [km/s]', fontsize=14)

# Hide any unused subplots
for i in range(n_years, len(axs_yearly)):
    axs_yearly[i].set_visible(False)

# Add a shared legend at the bottom
handles, labels = axs_yearly[0].get_legend_handles_labels()
fig_yearly.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, 
                  bbox_to_anchor=(0.5, -0.02), frameon=False)

fig_yearly.suptitle(f'Solar Wind Speed: Observations vs {plot_lead_time}-Day Forecasts by Year', 
                    fontsize=16, fontweight='bold', y=1.02)
fig_yearly.tight_layout()
fig_yearly.savefig(os.path.join(figdir, f'yearly_comparison_{plot_lead_time}day.pdf'), 
                   dpi=150, bbox_inches='tight')
print(f"Saved yearly comparison plot as yearly_comparison_{plot_lead_time}day.pdf")


# <codecell> compute MAE at different lead times

for_insitu_omni = []
for_wsa_omni = []
#interpolate each forecast series onto the OMNI time step
for n in range(0, len(combined_for_list)):
    for_insitu_omni.append( np.interp(omni_noicmes['mjd'], 
                                      combined_for_list[n]['mjd'], combined_for_list[n]['vsw'],
                                      left = np.nan, right = np.nan) )
    for_wsa_omni.append( np.interp(omni_noicmes['mjd'], 
                                      combined_for_wsa_list[n]['mjd'], combined_for_wsa_list[n]['vsw'],
                                      left = np.nan, right = np.nan) )
    
#compute MAE at each lead time (overall and by year)
mae_wsa_lt = []
mae_insitu_lt = []
for n in range(0, len(combined_for_list)):
    # Compute absolute errors
    abs_err_insitu = np.abs(for_insitu_omni[n] - omni_noicmes['V'])
    abs_err_wsa = np.abs(for_wsa_omni[n] - omni_noicmes['V'])
    
    # MAE
    mae_insitu_lt.append(np.nanmean(abs_err_insitu))
    mae_wsa_lt.append(np.nanmean(abs_err_wsa))

# Compute MAE by year for each lead time
years = range(start_time.year, stop_time.year)
mae_insitu_by_year = {year: [] for year in years}
mae_wsa_by_year = {year: [] for year in years}

for year in years:
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year, 12, 31, 23, 59, 59)
    
    mask_year = ((omni_noicmes['datetime'] >= year_start) & 
                 (omni_noicmes['datetime'] <= year_end))
    
    for n in range(0, len(combined_for_list)):
        # Get absolute errors for this year only
        abs_err_insitu = np.abs(for_insitu_omni[n][mask_year.values] - omni_noicmes.loc[mask_year, 'V'].values)
        abs_err_wsa = np.abs(for_wsa_omni[n][mask_year.values] - omni_noicmes.loc[mask_year, 'V'].values)
        
        mae_insitu_by_year[year].append(np.nanmean(abs_err_insitu))
        mae_wsa_by_year[year].append(np.nanmean(abs_err_wsa))


# <codecell> Compute MAE for 27.27-day recurrence forecast baseline
# Forecast source: OMNI (all data), shifted by one synodic solar rotation.
# Verification: OMNI with ICMEs removed.
recurrence_days = 27.27

# Forecast valid at time t is OMNI speed at (t - 27.27 days)
recurrence_mjd = omni_noicmes['mjd'].values - recurrence_days
recurrence_forecast = np.interp(
    recurrence_mjd,
    omni['mjd'].values,
    omni['V'].values,
    left=np.nan,
    right=np.nan
)

abs_err_recurrence = np.abs(recurrence_forecast - omni_noicmes['V'].values)
mae_recurrence = np.nanmean(abs_err_recurrence)

# Recurrence MAE by year for consistency with model-year diagnostics
mae_recurrence_by_year = {year: np.nan for year in years}
for year in years:
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year, 12, 31, 23, 59, 59)
    mask_year = ((omni_noicmes['datetime'] >= year_start) &
                 (omni_noicmes['datetime'] <= year_end)).values
    if np.any(mask_year):
        mae_recurrence_by_year[year] = np.nanmean(abs_err_recurrence[mask_year])

print("\n27.27-day recurrence forecast baseline:")
print(f"  Forecast source: OMNI shifted by {recurrence_days:.2f} days")
print(f"  Verification:    OMNI with ICMEs removed")
print(f"  MAE (overall):   {mae_recurrence:.2f} km/s")

lead_times = np.arange(min_lt, max_lt+1, 1)
# Limit plotting to 26 days maximum
plot_max_lt = 26
lead_times_plot = lead_times[:plot_max_lt]

fig, axs = plt.subplots(1, 1, figsize=(10, 6))

# Convert colors to RGB for tinting
import matplotlib.colors as mcolors
color_insitu_rgb = mcolors.to_rgb(color_insitu)
color_wsa_rgb = mcolors.to_rgb(color_wsa)

# Plot individual years with lighter shades
n_years = len(years)
for i, year in enumerate(years):
    # Create lighter tints of the base colors (blend with white)
    alpha_year = 0.5  # Vary transparency slightly
    
    axs.plot(lead_times_plot, np.array(mae_insitu_by_year[year][:plot_max_lt]),
             color=color_insitu, alpha=alpha_year, linewidth=1, zorder=1)
    axs.plot(lead_times_plot, np.array(mae_wsa_by_year[year][:plot_max_lt]),
             color=color_wsa, alpha=alpha_year, linewidth=1, zorder=1)

# Plot overall average with thicker lines on top
axs.plot(lead_times_plot, np.array(mae_insitu_lt[:plot_max_lt]),
         color=color_insitu, label='InSitu-HUXt (mean)', linewidth=2.5, 
         marker='o', markersize=6, zorder=10)
axs.plot(lead_times_plot, np.array(mae_wsa_lt[:plot_max_lt]),
         color=color_wsa, label='WSA-HUXt (mean)', linewidth=2.5, 
         marker='s', markersize=6, zorder=10)

# Plot 27.27-day recurrence baseline as a horizontal reference line
axs.axhline(mae_recurrence, color='0.2', linestyle='--', linewidth=2.0,
            label=f'27.27-day recurrence (MAE={mae_recurrence:.1f} km/s)', zorder=5)

axs.set_xlabel('Forecast lead time [days]', fontsize=14)
axs.set_ylabel('MAE [km/s]', fontsize=14)
#axs.set_title(f'Mean Absolute Error vs Lead Time\n({start_time.year} to {stop_time.year})', fontsize=14)
axs.legend(fontsize=14)
axs.grid(True, alpha=0.3)
axs.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(figdir, 'mae_vs_leadtime.pdf'), dpi=150, bbox_inches='tight')
print("Saved MAE vs lead time plot as mae_vs_leadtime.pdf")


# <codecell> Example forecast for a a few daya showing full 27-day forecast
# =============================================================================
# Show a single forecast run from the initialization date forward 27 days,
# comparing InSitu-HUXt and WSA-HUXt against observations
# =============================================================================

for forecast_date in example_forecast_dates:
    # Load the actual forecast files for this specific initialization date
    yr_str = f"{forecast_date.year:04d}"
    mn_str = f"{forecast_date.month:02}"
    dy_str = f"{forecast_date.day:02}"

    # Load InSitu-HUXt forecast for example date
    insitu_filename = 'InSitu_Earth_' + yr_str + mn_str + dy_str + '.dat'
    insitu_filepath = os.path.join(ishuxt_dir, yr_str, insitu_filename)
    if os.path.exists(insitu_filepath):
        insitu_forecast_df = pd.read_csv(insitu_filepath, index_col=0, parse_dates=["time"])
        print(f"Loaded InSitu-HUXt forecast from {insitu_filepath}")
    else:
        print(f"Warning: InSitu-HUXt forecast file not found: {insitu_filepath}")
        insitu_forecast_df = pd.DataFrame()

    # Load WSA-HUXt forecast for example date
    wsa_filename = 'HUXt_Earth_' + yr_str + mn_str + dy_str + '.dat'
    wsa_filepath = os.path.join(wsahuxt_dir, yr_str, wsa_filename)
    if os.path.exists(wsa_filepath):
        wsa_forecast_df = pd.read_csv(wsa_filepath, index_col=0, parse_dates=["time"])
        print(f"Loaded WSA-HUXt forecast from {wsa_filepath}")
    else:
        print(f"Warning: WSA-HUXt forecast file not found: {wsa_filepath}")
        wsa_forecast_df = pd.DataFrame()

    fig_example, axs_example = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Define the forecast period (starts at initialization)
    forecast_start = forecast_date
    forecast_end = forecast_date + datetime.timedelta(days=max_lt)

    # Define the full 54-day window (27 days before + 27 days after initialization)
    plot_start = forecast_date - datetime.timedelta(days=27)
    plot_end = forecast_date + datetime.timedelta(days=27)

    # Get observations for the full 54-day window
    mask_omni_full = (omni['datetime'] >= plot_start) & (omni['datetime'] <= plot_end)
    mask_omni_noicmes = (omni_noicmes['datetime'] >= plot_start) & (omni_noicmes['datetime'] <= plot_end)

    # Extract the forecast time series from the loaded files
    # Forecast runs from t+1 to t+27 days after initialization
    # Solar rotation period is 27.27 days (synodic period)
    rotation_period_days = 27.27
    rotation_period_td = datetime.timedelta(days=rotation_period_days)
    
    if len(insitu_forecast_df) > 0:
        # Get the full forecast (starts at t+1 day, ends at t+max_lt)
        mask_insitu = (insitu_forecast_df['time'] > forecast_date) & (insitu_forecast_df['time'] <= forecast_end)
        insitu_forecast_times = insitu_forecast_df.loc[mask_insitu, 'time'].values
        insitu_forecast_vsw = insitu_forecast_df.loc[mask_insitu, 'vsw'].values
        
        # Shift the forecast back by the solar rotation period (27.27 days)
        # to show the recurrence pattern that the model uses
        insitu_forecast_times_shifted = insitu_forecast_times - pd.Timedelta(days=rotation_period_days)
        insitu_forecast_vsw_shifted = insitu_forecast_vsw.copy()
    else:
        insitu_forecast_times = []
        insitu_forecast_vsw = []
        insitu_forecast_times_shifted = []
        insitu_forecast_vsw_shifted = []

    if len(wsa_forecast_df) > 0:
        # Get the full forecast (starts at t+1 day, ends at t+max_lt)
        mask_wsa = (wsa_forecast_df['time'] > forecast_date) & (wsa_forecast_df['time'] <= forecast_end)
        wsa_forecast_times = wsa_forecast_df.loc[mask_wsa, 'time'].values
        wsa_forecast_vsw = wsa_forecast_df.loc[mask_wsa, 'vsw'].values
        
        # Shift back by the solar rotation period
        wsa_forecast_times_shifted = wsa_forecast_times - pd.Timedelta(days=rotation_period_days)
        wsa_forecast_vsw_shifted = wsa_forecast_vsw.copy()
    else:
        wsa_forecast_times = []
        wsa_forecast_vsw = []
        wsa_forecast_times_shifted = []
        wsa_forecast_vsw_shifted = []

    # Plot InSitu-HUXt forecast
    ax = axs_example[0]
    ax.plot(omni.loc[mask_omni_full, 'datetime'], omni.loc[mask_omni_full, 'V'],
            color='#7B7B7B', label='OMNI (all)', linewidth=2, alpha=0.9, zorder=3)
    ax.plot(omni_noicmes.loc[mask_omni_noicmes, 'datetime'], omni_noicmes.loc[mask_omni_noicmes, 'V'],
            color=color_obs, label='OMNI (no ICMEs)', linewidth=2.5, zorder=5)
    
    # Calculate MAE for InSitu-HUXt
    mae_insitu = None
    if len(insitu_forecast_times) > 0 and len(insitu_forecast_vsw) > 0:
        # Match forecast times with observations
        mae_errors = []
        for ft, fv in zip(insitu_forecast_times, insitu_forecast_vsw):
            # Skip NaN values
            if np.isnan(fv):
                continue
            # Convert numpy datetime64 to pandas Timestamp
            ft_ts = pd.Timestamp(ft)
            # Find closest observation time
            omni_mask = (omni_noicmes['datetime'] >= ft_ts - datetime.timedelta(hours=6)) & \
                       (omni_noicmes['datetime'] <= ft_ts + datetime.timedelta(hours=6))
            if np.any(omni_mask):
                obs_vals = omni_noicmes.loc[omni_mask, 'V'].values
                # Exclude NaNs from observations
                obs_vals = obs_vals[~np.isnan(obs_vals)]
                if len(obs_vals) > 0:
                    mae_errors.append(abs(fv - np.mean(obs_vals)))
        if mae_errors:
            mae_insitu = np.mean(mae_errors)
    
    if len(insitu_forecast_times) > 0:
        # Plot the current forecast (solid line)
        ax.plot(insitu_forecast_times, insitu_forecast_vsw,
                color=color_insitu, label='InSitu-HUXt Forecast', linewidth=2.5, zorder=10)
        # Plot the shifted forecast showing previous 27 days (dashed line)
        ax.plot(insitu_forecast_times_shifted, insitu_forecast_vsw_shifted,
                color=color_insitu, label='Forecast (27d prior)', linewidth=2.5, 
                linestyle='--', alpha=0.8, zorder=9)
    ax.axvline(forecast_date, color='gray', linestyle='--', linewidth=2.5, alpha=0.9, label='Forecast initialization')
    ax.set_ylabel('Solar wind speed [km/s]', fontsize=14)
    ax.set_xlim((plot_start, plot_end))
    ax.set_ylim((250, 800))
    
    # Add MAE label
    title_str = f'(a) InSitu-HUXt Forecast (initialized {forecast_date.strftime("%Y-%m-%d")})'
    if mae_insitu is not None:
        title_str += f'\nMAE = {mae_insitu:.1f} km/s'
    ax.set_title(title_str, fontsize=14, fontweight='bold', color=color_insitu)
    # Get handles and labels, keep only first 3 items (OMNI all, OMNI no ICMEs, Forecast)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc='upper left', fontsize=14, facecolor='white', edgecolor='black', framealpha=1.0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=14)

    # Add lead time annotations (0d at forecast start, then 7, 14, 21, 27)
    for lt in [0, 7, 14, 21, 27]:
        lt_date = forecast_start + datetime.timedelta(days=lt)
        if lt_date <= forecast_end:
            ax.axvline(lt_date, color='gray', linestyle=':', linewidth=2, alpha=0.8)
            ax.text(lt_date, ax.get_ylim()[1] * 0.98, f'{lt}d', fontsize=14,
                    ha='center', va='top', color='gray')

    # Plot WSA-HUXt forecast
    ax = axs_example[1]
    ax.plot(omni.loc[mask_omni_full, 'datetime'], omni.loc[mask_omni_full, 'V'],
            color='#7B7B7B', label='OMNI (all)', linewidth=2, alpha=0.9, zorder=3)
    ax.plot(omni_noicmes.loc[mask_omni_noicmes, 'datetime'], omni_noicmes.loc[mask_omni_noicmes, 'V'],
            color=color_obs, label='OMNI (no ICMEs)', linewidth=2.5, zorder=5)
    
    # Calculate MAE for WSA-HUXt
    mae_wsa = None
    if len(wsa_forecast_times) > 0 and len(wsa_forecast_vsw) > 0:
        # Match forecast times with observations
        mae_errors = []
        for ft, fv in zip(wsa_forecast_times, wsa_forecast_vsw):
            # Skip NaN values
            if np.isnan(fv):
                continue
            # Convert numpy datetime64 to pandas Timestamp
            ft_ts = pd.Timestamp(ft)
            # Find closest observation time
            omni_mask = (omni_noicmes['datetime'] >= ft_ts - datetime.timedelta(hours=6)) & \
                       (omni_noicmes['datetime'] <= ft_ts + datetime.timedelta(hours=6))
            if np.any(omni_mask):
                obs_vals = omni_noicmes.loc[omni_mask, 'V'].values
                # Exclude NaNs from observations
                obs_vals = obs_vals[~np.isnan(obs_vals)]
                if len(obs_vals) > 0:
                    mae_errors.append(abs(fv - np.mean(obs_vals)))
        if mae_errors:
            mae_wsa = np.mean(mae_errors)
    
    if len(wsa_forecast_times) > 0:
        # Plot the current forecast (solid line)
        ax.plot(wsa_forecast_times, wsa_forecast_vsw,
                color=color_wsa, label='WSA-HUXt Forecast', linewidth=2.5, zorder=10)
        # Plot the shifted forecast showing previous 27 days (dashed line)
        ax.plot(wsa_forecast_times_shifted, wsa_forecast_vsw_shifted,
                color=color_wsa, label='Forecast (27d prior)', linewidth=2.5,
                linestyle='--', alpha=0.8, zorder=9)
    ax.axvline(forecast_date, color='gray', linestyle='--', linewidth=2.5, alpha=0.9, label='Forecast initialization')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Solar wind speed [km/s]', fontsize=14)
    ax.set_xlim((plot_start, plot_end))
    ax.set_ylim((250, 800))
    
    # Add MAE label
    title_str = f'(b) WSA-HUXt Forecast (initialized {forecast_date.strftime("%Y-%m-%d")})'
    if mae_wsa is not None:
        title_str += f'\nMAE = {mae_wsa:.1f} km/s'
    ax.set_title(title_str, fontsize=14, fontweight='bold', color=color_wsa)
    # Get handles and labels, keep only first 3 items (OMNI all, OMNI no ICMEs, Forecast)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc='upper left', fontsize=14, facecolor='white', edgecolor='black', framealpha=1.0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=14)

    for lt in [0, 7, 14, 21, 27]:
        lt_date = forecast_start + datetime.timedelta(days=lt)
        if lt_date <= forecast_end:
            ax.axvline(lt_date, color='gray', linestyle=':', linewidth=2, alpha=0.8)
            ax.text(lt_date, ax.get_ylim()[1] * 0.98, f'{lt}d', fontsize=14,
                    ha='center', va='top', color='gray')

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_xlabel(f'Date ({forecast_date.year})', fontsize=14)

    fig_example.tight_layout()
    outfile = os.path.join(figdir, f'example_forecast_{forecast_date.strftime("%Y%m%d")}.pdf')
    fig_example.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved example forecast plot as example_forecast_{forecast_date.strftime('%Y%m%d')}.pdf")
    plt.close(fig_example)

# Keep the original 2019-07-15 example for downstream sections (e.g., ecliptic plots)


""" # <codecell> Also plot the ecliptic solutions for this same example period
example_forecast_date = datetime.datetime(2014, 6, 9)


# Create interpolated version for HUXt input
omni_noicmes_interp = omni_noicmes.copy()
omni_noicmes_interp = omni_noicmes_interp.set_index('datetime')
omni_noicmes_interp[['V', 'BX_GSE']] = omni_noicmes_interp[['V', 'BX_GSE']].interpolate(method='time').ffill().bfill()
omni_noicmes_interp = omni_noicmes_interp.reset_index()

model_insitu = His.omniHUXt_forecast(example_forecast_date, simtime = 5*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes_interp, 
                            buffertime = 0*u.day, run_2d = True)
model_insitu.solve([])  
fig_insitu, ax_insitu = HA.plot(model_insitu, time = 0*u.day)
plt.close(fig_insitu)  # Close the individual figure

#find and run the WSA-HUXt for the same date
# Construct the WSA fits file path for the example forecast date
# Files are in YEAR subdirectory with format: vel_YYYYMMDDHHmmR000_gong_z.fits
wsa_fits_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC')
yr_str = str(example_forecast_date.year)

# Search for WSA file matching the date (format: vel_YYYYMMDDHHmmR000_gong_z.fits)
wsa_date_pattern = example_forecast_date.strftime('%Y%m%d')
wsa_search_pattern = os.path.join(wsa_fits_dir, yr_str, f'vel_{wsa_date_pattern}*_gong_z.fits')
matching_wsa_files = glob.glob(wsa_search_pattern)

if matching_wsa_files:
    # Use the first matching file (or closest to start of day)
    matching_wsa_files.sort()
    wsa_fits_filepath = matching_wsa_files[0]
    print(f"Found WSA fits file: {wsa_fits_filepath}")
else:
    print(f"No WSA fits file found for {example_forecast_date.strftime('%Y-%m-%d')}")
    print(f"Searched: {wsa_search_pattern}")
    wsa_fits_filepath = None

cr, cr_lon_init = Hin.datetime2huxtinputs(example_forecast_date)



vr_in = Hin.get_WSA_long_profile(wsa_fits_filepath, lat=4*u.deg)
#deaccelerate the WSA speeds
nlons = len(vr_in)
dlon = 2*np.pi/nlons
vr_longs = np.arange(dlon/2, 2*np.pi, dlon)*u.rad
vr_in, lon_temp = Hin.map_v_inwards(vr_in, 215*u.solRad, vr_longs, rmin)

model_wsa = H.HUXt(v_boundary=vr_in, cr_num=cr, 
                   cr_lon_init=cr_lon_init,
                    r_min = rmin, r_max = rmax, 
                    dt_scale = dt_scale, latitude = 4*u.deg,
                    simtime = 1*u.day)
model_wsa.solve([])  
fig_wsa, ax_wsa = HA.plot(model_wsa, time = 3*u.day)
plt.close(fig_wsa)  # Close the individual figure

# Run InSitu-HUXt with CME
model_insitu_cme = His.omniHUXt_forecast(example_forecast_date, simtime = 5*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes_interp, 
                            buffertime = 0*u.day, run_2d = True)
model_insitu_cme.track_cmes = True
# Add CME: longitude=0, latitude=0, width=30 deg, speed=500 km/s at t=0
cme_list = [H.ConeCME(t_launch=0.0*u.day, longitude=0.0*u.deg, 
                      width=60*u.deg, v=500*(u.km/u.s), latitude=0.0*u.deg,
                      initial_height=rmin)]
model_insitu_cme.solve(cme_list)  
fig_insitu_cme, ax_insitu_cme = HA.plot(model_insitu_cme, time = 3*u.day)
plt.close(fig_insitu_cme)  # Close the individual figure

# Run WSA-HUXt with CME
model_wsa_cme = H.HUXt(v_boundary=vr_in, cr_num=cr, 
                   cr_lon_init=cr_lon_init,
                    r_min = rmin, r_max = rmax, 
                    dt_scale = dt_scale, latitude = 4*u.deg,
                    simtime = 5*u.day)
model_wsa_cme.solve(cme_list)  
fig_wsa_cme, ax_wsa_cme = HA.plot(model_wsa_cme, time = 3*u.day)
plt.close(fig_wsa_cme)  # Close the individual figure

# Create combined figure with 2x2 grid
fig_combined_ecliptic, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20), 
                                                   subplot_kw={"projection": "polar"})

# Top row: No CME
# Re-plot InSitu-HUXt on the top-left subplot (minimalplot=True to suppress colorbar)
HA.plot(model_insitu, time=0*u.day, fighandle=fig_combined_ecliptic, axhandle=ax1, minimalplot=True)
ax1.set_title(f'(a) InSitu-HUXt (no CME)\n({example_forecast_date.strftime("%Y-%m-%d")})', 
              fontsize=14, fontweight='bold', color=color_insitu, pad=20)

# Re-plot WSA-HUXt on the top-right subplot (minimalplot=True to suppress colorbar)
HA.plot(model_wsa, time=0*u.day, fighandle=fig_combined_ecliptic, axhandle=ax2, minimalplot=True)
ax2.set_title(f'(b) WSA-HUXt (no CME)\n({example_forecast_date.strftime("%Y-%m-%d")})', 
              fontsize=14, fontweight='bold', color=color_wsa, pad=20)

# Bottom row: With CME at 2.85 days
# Re-plot InSitu-HUXt with CME on the bottom-left subplot
HA.plot(model_insitu_cme, time=2.85*u.day, fighandle=fig_combined_ecliptic, axhandle=ax3, minimalplot=True)
ax3.set_title(f'(c) InSitu-HUXt with CME (t=+2.85 days)\n(0° lon, 0° lat, 60° width, 500 km/s)', 
              fontsize=14, fontweight='bold', color=color_insitu, pad=20)

# Re-plot WSA-HUXt with CME on the bottom-right subplot
HA.plot(model_wsa_cme, time=2.85*u.day, fighandle=fig_combined_ecliptic, axhandle=ax4, minimalplot=True)
ax4.set_title(f'(d) WSA-HUXt with CME (t=+2.85 days)\n(0° lon, 0° lat, 60° width, 500 km/s)', 
              fontsize=14, fontweight='bold', color=color_wsa, pad=20)

# Add planet positions to all four subplots
def add_observers(ax, model, time=3*u.day):
   #Add observer/planet positions to a polar plot
    id_t = np.argmin(np.abs(model.time_out - time))
    
    # Define which bodies to plot
    plot_observers = [('EARTH', 'ko'), ('STA', 'c*')]
    if model.time_init < datetime.datetime(2016, 8, 21):
        plot_observers.append(('STB', 'y*'))
    if model.r[-1] < 350 * u.solRad:
        plot_observers.append(('VENUS', 'yo'))
        plot_observers.append(('MERCURY', 'mo'))
    if model.r[-1] > 350 * u.solRad:
        plot_observers.append(('MARS', 'ro'))
    
    for body, style in plot_observers:
        obs = model.get_observer(body)
        deltalon = 0.0 * u.rad
        if model.frame == 'sidereal':
            earth_pos = model.get_observer('EARTH')
            deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]
        obslon = H._zerototwopi_(obs.lon[id_t] + deltalon)
        ax.plot(obslon, obs.r[id_t], style, markersize=14, label=body)
    
    # Add legend
    ax.legend(ncol=5, loc='lower center', frameon=False, fontsize=14, 
              handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0.5, -0.15))

add_observers(ax1, model_insitu, time=0*u.day)
add_observers(ax2, model_wsa, time=0*u.day)
add_observers(ax3, model_insitu_cme, time=3*u.day)
add_observers(ax4, model_wsa_cme, time=3*u.day)

# Add a single shared colorbar at the bottom
import matplotlib as mpl
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=200, vmax=810)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Position the colorbar below both plots
cbar_ax = fig_combined_ecliptic.add_axes([0.25, 0.02, 0.5, 0.015])
cbar = fig_combined_ecliptic.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$V_{SW}$ [km/s]', fontsize=14)
cbar.set_ticks(np.arange(200, 850, 100))

fig_combined_ecliptic.suptitle('Ecliptic Solar Wind Solutions', fontsize=16, fontweight='bold', y=0.99)
fig_combined_ecliptic.subplots_adjust(bottom=0.08, top=0.96, hspace=0.25)
fig_combined_ecliptic.savefig(os.path.join(figdir, f'ecliptic_comparison_{example_forecast_date.strftime("%Y%m%d")}.pdf'), 
                               dpi=150, bbox_inches='tight')
print(f"Saved ecliptic comparison plot as ecliptic_comparison_{example_forecast_date.strftime('%Y%m%d')}.pdf")

 """
# <codecell> Compute point-wise statistics for Taylor diagrams

# Taylor diagram for overall period (using plot_lead_time)
# Get the data for the chosen lead time
obs = omni_noicmes['V'].values
pred_insitu = for_insitu_omni[plot_lead_time-1]
pred_wsa = for_wsa_omni[plot_lead_time-1]

# Remove NaN values for statistics calculation
mask = ~(np.isnan(obs) | np.isnan(pred_insitu) | np.isnan(pred_wsa))
obs_clean = obs[mask]
pred_insitu_clean = pred_insitu[mask]
pred_wsa_clean = pred_wsa[mask]

# Calculate regular RMSD for comparison
rmsd_insitu = np.sqrt(np.mean((pred_insitu_clean - obs_clean)**2))
rmsd_wsa = np.sqrt(np.mean((pred_wsa_clean - obs_clean)**2))
mae_insitu = np.mean(np.abs(pred_insitu_clean - obs_clean))
mae_wsa = np.mean(np.abs(pred_wsa_clean - obs_clean))

print(f"\n{plot_lead_time}-day lead time statistics:")
print(f"InSitu-HUXt: MAE = {mae_insitu:.1f} km/s, RMSD = {rmsd_insitu:.1f} km/s")
print(f"WSA-HUXt:    MAE = {mae_wsa:.1f} km/s, RMSD = {rmsd_wsa:.1f} km/s")

# Calculate statistics for target diagram
obs_std = np.std(obs_clean, ddof=1)
obs_mean = np.mean(obs_clean)
bias_insitu = np.mean(pred_insitu_clean - obs_clean)
bias_wsa = np.mean(pred_wsa_clean - obs_clean)

# Calculate unbiased RMSE (centered RMSE)
ubrmse_insitu = np.sqrt(np.mean((pred_insitu_clean - obs_clean - bias_insitu)**2))
ubrmse_wsa = np.sqrt(np.mean((pred_wsa_clean - obs_clean - bias_wsa)**2))

# <codecell> Calculate stats for combined diagrams

# Calculate stats for each year
years = range(start_time.year, stop_time.year)
obs_by_year = []
pred_insitu_by_year = []
pred_wsa_by_year = []
year_labels = []

for year in years:
    # Create mask for this year
    year_start = datetime.datetime(year, 1, 1)
    year_end = datetime.datetime(year, 12, 31, 23, 59, 59)
    
    mask_year = ((omni_noicmes['datetime'] >= year_start) & 
                 (omni_noicmes['datetime'] <= year_end))
    
    obs_year = omni_noicmes.loc[mask_year, 'V'].values
    pred_insitu_year = for_insitu_omni[plot_lead_time-1][mask_year.values]
    pred_wsa_year = for_wsa_omni[plot_lead_time-1][mask_year.values]
    
    # Remove NaN values
    mask_valid = ~(np.isnan(obs_year) | np.isnan(pred_insitu_year) | np.isnan(pred_wsa_year))
    
    if np.sum(mask_valid) > 100:  # Only include years with sufficient data
        obs_by_year.append(obs_year[mask_valid])
        pred_insitu_by_year.append(pred_insitu_year[mask_valid])
        pred_wsa_by_year.append(pred_wsa_year[mask_valid])
        year_labels.append(str(year))

# Create normalized data (across all years)
obs_all_years = np.concatenate(obs_by_year)
pred_insitu_all_years = np.concatenate(pred_insitu_by_year)
pred_wsa_all_years = np.concatenate(pred_wsa_by_year)

# <codecell> Create combined Taylor and Target diagrams

# First pass: collect std values from all years to compute ranges
std_insitu_all_years = []
std_wsa_all_years = []
std_obs_all_years = []

for i, year in enumerate(year_labels):
    obs_year = obs_by_year[i]
    pred_insitu_year = pred_insitu_by_year[i]
    pred_wsa_year = pred_wsa_by_year[i]
    
    stats_insitu = compute_taylor_target_stats(obs_year, pred_insitu_year)
    stats_wsa = compute_taylor_target_stats(obs_year, pred_wsa_year)
    
    std_insitu_all_years.append(stats_insitu['std'])
    std_wsa_all_years.append(stats_wsa['std'])
    std_obs_all_years.append(np.std(obs_year, ddof=1))

# Compute std ranges for each model and for observations
std_range_insitu = (min(std_insitu_all_years), max(std_insitu_all_years))
std_range_wsa = (min(std_wsa_all_years), max(std_wsa_all_years))
std_ranges = [std_range_insitu, std_range_wsa]

obs_std_range = (min(std_obs_all_years), max(std_obs_all_years))

# Combined diagram for overall period (with individual years)
# Use obs_all_years to be consistent with the individual year data
fig_combined, (ax_t, ax_tg) = combined_taylor_target_diagram(
    obs_all_years,
    [pred_insitu_all_years, pred_wsa_all_years],
    labels=['InSitu-HUXt', 'WSA-HUXt'],
    colors=['#2E86AB', '#A23B72'],
    figsize=(16, 7),
    dpi=150,
    units='[km/s]',
    figname=None,  # Don't save yet - will add individual years first
    obs_std_range=obs_std_range  # Pass the observation std range for shading
)

# Add individual years as lighter dots to the diagrams
obs_std_overall = np.std(obs_all_years, ddof=1)
obs_mean_overall = np.mean(obs_all_years)

for i, year in enumerate(year_labels):
    obs_year = obs_by_year[i]
    pred_insitu_year = pred_insitu_by_year[i]
    pred_wsa_year = pred_wsa_by_year[i]
    
    # Compute statistics using the standalone function for consistency
    stats_insitu = compute_taylor_target_stats(obs_year, pred_insitu_year)
    stats_wsa = compute_taylor_target_stats(obs_year, pred_wsa_year)
    
    # Plot on Taylor diagram (polar coordinates: angle=arccos(corr), radius=std)
    theta_insitu = np.arccos(stats_insitu['corr'])
    theta_wsa = np.arccos(stats_wsa['corr'])
    ax_t.plot(theta_insitu, stats_insitu['std'], 'o', color='#2E86AB', markersize=12, alpha=0.3)
    ax_t.plot(theta_wsa, stats_wsa['std'], 'o', color='#A23B72', markersize=12, alpha=0.3)
    
    # Plot on Target diagram (x=bias, y=ubRMSE)
    ax_tg.plot(stats_insitu['bias'], stats_insitu['ubrmse'], 'o', color='#2E86AB', markersize=12, alpha=0.3)
    ax_tg.plot(stats_wsa['bias'], stats_wsa['ubrmse'], 'o', color='#A23B72', markersize=12, alpha=0.3)

# Save the combined diagram
fig_combined.savefig(os.path.join(figdir, f'combined_diagram_{plot_lead_time}day.pdf'), dpi=150, bbox_inches='tight')
print(f"Saved figure as combined_diagram_{plot_lead_time}day.pdf")

# Combined diagram for normalized version (across all years with individual years)
# Normalize obs_std_range by overall observation std
obs_std_all = np.std(obs_all_years, ddof=1)
obs_std_range_norm = (obs_std_range[0] / obs_std_all, obs_std_range[1] / obs_std_all)

fig_combined_norm, (ax_t_norm, ax_tg_norm) = combined_taylor_target_diagram(
    obs_all_years,
    [pred_insitu_all_years, pred_wsa_all_years],
    labels=['InSitu-HUXt', 'WSA-HUXt'],
    colors=['#2E86AB', '#A23B72'],
    figsize=(16, 7),
    dpi=150,
    normalize=True,
    figname=None,  # Don't save yet
    obs_std_range=obs_std_range_norm  # Pass normalized obs std range
)

for i, year in enumerate(year_labels):
    obs_year = obs_by_year[i]
    pred_insitu_year = pred_insitu_by_year[i]
    pred_wsa_year = pred_wsa_by_year[i]
    
    # Compute statistics using the standalone function for consistency
    stats_insitu = compute_taylor_target_stats(obs_year, pred_insitu_year)
    stats_wsa = compute_taylor_target_stats(obs_year, pred_wsa_year)
    
    # Normalize by overall observation std
    stats_insitu_norm = {
        'std': stats_insitu['std'] / obs_std_all,
        'corr': stats_insitu['corr'],
        'crmse': stats_insitu['crmse'] / obs_std_all,
        'bias': stats_insitu['bias'] / obs_std_all,
        'ubrmse': stats_insitu['ubrmse'] / obs_std_all
    }
    
    stats_wsa_norm = {
        'std': stats_wsa['std'] / obs_std_all,
        'corr': stats_wsa['corr'],
        'crmse': stats_wsa['crmse'] / obs_std_all,
        'bias': stats_wsa['bias'] / obs_std_all,
        'ubrmse': stats_wsa['ubrmse'] / obs_std_all
    }
    
    # Plot on Taylor diagram (polar coordinates: angle=arccos(corr), radius=std)
    theta_insitu = np.arccos(stats_insitu_norm['corr'])
    theta_wsa = np.arccos(stats_wsa_norm['corr'])
    ax_t_norm.plot(theta_insitu, stats_insitu_norm['std'], 'o', color='#2E86AB', markersize=12, alpha=0.3)
    ax_t_norm.plot(theta_wsa, stats_wsa_norm['std'], 'o', color='#A23B72', markersize=12, alpha=0.3)
    
    # Plot on Target diagram (x=bias, y=ubRMSE)
    ax_tg_norm.plot(stats_insitu_norm['bias'], stats_insitu_norm['ubrmse'], 'o', color='#2E86AB', markersize=12, alpha=0.3)
    ax_tg_norm.plot(stats_wsa_norm['bias'], stats_wsa_norm['ubrmse'], 'o', color='#A23B72', markersize=12, alpha=0.3)

# Save the normalized combined diagram
fig_combined_norm.savefig(os.path.join(figdir, f'combined_diagram_{plot_lead_time}day_normalized.pdf'), dpi=150, bbox_inches='tight')
print(f"Saved figure as combined_diagram_{plot_lead_time}day_normalized.pdf")


# <codecell> High-Speed Enhancement (HSE) Event-Based Analysis
# =============================================================
# Detect HSE events in observations and forecasts, then compute
# timing errors and peak speed errors following Owens (2005) method
# =============================================================

print("\n" + "="*60)
print(f"HIGH-SPEED ENHANCEMENT ANALYSIS ({plot_lead_time}-day lead time)")
print("="*60)

# Prepare dataframes for HSE detection
# Need datetime index and speed columns

# Create observation dataframe
obs_df = pd.DataFrame({
    'datetime': omni_noicmes['datetime'].values,
    'V': omni_noicmes['V'].values
})
obs_df = obs_df.set_index('datetime')

# Create InSitu-HUXt forecast dataframe for the selected lead time
insitu_times = pd.to_datetime(combined_for_list[plot_lead_time-1]['time'].values)
insitu_df = pd.DataFrame({
    'datetime': insitu_times.round('h'),  # Round to nearest hour for alignment
    'V': combined_for_list[plot_lead_time-1]['vsw'].values
})
# Group by datetime and take mean to handle duplicate timestamps after rounding
insitu_df = insitu_df.groupby('datetime').mean().reset_index()
insitu_df = insitu_df.set_index('datetime')

# Create WSA-HUXt forecast dataframe for the selected lead time
wsa_times = pd.to_datetime(combined_for_wsa_list[plot_lead_time-1]['time'].values)
wsa_df = pd.DataFrame({
    'datetime': wsa_times.round('h'),  # Round to nearest hour for alignment
    'V': combined_for_wsa_list[plot_lead_time-1]['vsw'].values
})
# Group by datetime and take mean to handle duplicate timestamps after rounding
wsa_df = wsa_df.groupby('datetime').mean().reset_index()
wsa_df = wsa_df.set_index('datetime')

# Apply smoothing using configured window
obs_smoothed = hse.smooth_8h(obs_df.reset_index(), date_col='datetime', window_hours=hse_smoothing_window)
insitu_smoothed = hse.smooth_8h(insitu_df.reset_index(), date_col='datetime', window_hours=hse_smoothing_window)
wsa_smoothed = hse.smooth_8h(wsa_df.reset_index(), date_col='datetime', window_hours=hse_smoothing_window)

# Detect HSE events in each time series using configured parameters
# Parameters from configuration: rise_threshold, rise_hours, min_separation
events_obs = hse.detect_hss_events(obs_smoothed, speed_col='V', rise_threshold=hse_rise_threshold, 
                                    rise_hours=hse_rise_hours, min_separation_hours=hse_min_separation)
events_insitu = hse.detect_hss_events(insitu_smoothed, speed_col='V', rise_threshold=hse_rise_threshold,
                                      rise_hours=hse_rise_hours, min_separation_hours=hse_min_separation)
events_wsa = hse.detect_hss_events(wsa_smoothed, speed_col='V', rise_threshold=hse_rise_threshold,
                                    rise_hours=hse_rise_hours, min_separation_hours=hse_min_separation)

print(f"\nHSE Events Detected:")
print(f"  Observations:  {len(events_obs)} events")
print(f"  InSitu-HUXt:   {len(events_insitu)} events")
print(f"  WSA-HUXt:      {len(events_wsa)} events")

# Match events between observations and forecasts
# Using Owens (2005) method with configured maximum timing difference
matches_insitu, unmatched_obs_insitu, unmatched_insitu = hse.match_events(
    events_obs, events_insitu, delta_t_max=hse_max_matching_dt)
matches_wsa, unmatched_obs_wsa, unmatched_wsa = hse.match_events(
    events_obs, events_wsa, delta_t_max=hse_max_matching_dt)

print(f"\nEvent Matching Results (max dt = {hse_max_matching_dt} hours):")
print(f"  InSitu-HUXt: {len(matches_insitu)} matched, "
      f"{len(unmatched_obs_insitu)} missed, {len(unmatched_insitu)} false alarms")
print(f"  WSA-HUXt:    {len(matches_wsa)} matched, "
      f"{len(unmatched_obs_wsa)} missed, {len(unmatched_wsa)} false alarms")

# Compute timing errors for matched events (using max gradient time)
if len(matches_insitu) > 0:
    timing_errors_insitu = hse.compute_timing_errors(matches_insitu, use='max_gradient')
    timing_stats_insitu = hse.summary_error_stats(timing_errors_insitu)
    speed_errors_insitu, speed_stats_insitu = hse.peak_speed_errors(matches_insitu)
else:
    timing_errors_insitu = []
    timing_stats_insitu = {'mean_error': np.nan, 'mean_absolute_error': np.nan, 
                           'std_error': np.nan, 'n_events': 0}
    speed_errors_insitu = []
    speed_stats_insitu = {'mean_error': np.nan, 'mean_absolute_error': np.nan, 
                          'std_error': np.nan, 'n_events': 0}

if len(matches_wsa) > 0:
    timing_errors_wsa = hse.compute_timing_errors(matches_wsa, use='max_gradient')
    timing_stats_wsa = hse.summary_error_stats(timing_errors_wsa)
    speed_errors_wsa, speed_stats_wsa = hse.peak_speed_errors(matches_wsa)
else:
    timing_errors_wsa = []
    timing_stats_wsa = {'mean_error': np.nan, 'mean_absolute_error': np.nan, 
                        'std_error': np.nan, 'n_events': 0}
    speed_errors_wsa = []
    speed_stats_wsa = {'mean_error': np.nan, 'mean_absolute_error': np.nan, 
                       'std_error': np.nan, 'n_events': 0}

# Print timing error statistics
print(f"\nHSE Max Gradient Timing Errors (hours):")
print(f"  InSitu-HUXt:")
print(f"    Mean Error:     {timing_stats_insitu['mean_error']:+.1f} h "
      f"({'early' if timing_stats_insitu['mean_error'] < 0 else 'late'} bias)")
print(f"    Mean Abs Error: {timing_stats_insitu['mean_absolute_error']:.1f} h")
print(f"    Std Dev:        {timing_stats_insitu['std_error']:.1f} h")
print(f"    N events:       {timing_stats_insitu['n_events']}")

print(f"  WSA-HUXt:")
print(f"    Mean Error:     {timing_stats_wsa['mean_error']:+.1f} h "
      f"({'early' if timing_stats_wsa['mean_error'] < 0 else 'late'} bias)")
print(f"    Mean Abs Error: {timing_stats_wsa['mean_absolute_error']:.1f} h")
print(f"    Std Dev:        {timing_stats_wsa['std_error']:.1f} h")
print(f"    N events:       {timing_stats_wsa['n_events']}")

# Print peak speed error statistics
print(f"\nHSE Peak Speed Errors (km/s):")
print(f"  InSitu-HUXt:")
print(f"    Mean Error:     {speed_stats_insitu['mean_error']:+.1f} km/s")
print(f"    Mean Abs Error: {speed_stats_insitu['mean_absolute_error']:.1f} km/s")
print(f"    Std Dev:        {speed_stats_insitu['std_error']:.1f} km/s")

print(f"  WSA-HUXt:")
print(f"    Mean Error:     {speed_stats_wsa['mean_error']:+.1f} km/s")
print(f"    Mean Abs Error: {speed_stats_wsa['mean_absolute_error']:.1f} km/s")
print(f"    Std Dev:        {speed_stats_wsa['std_error']:.1f} km/s")

# Calculate contingency table metrics
# Hit Rate (Probability of Detection): matched / (matched + missed)
# False Alarm Rate: false_alarms / (matched + false_alarms)
# Critical Success Index: matched / (matched + missed + false_alarms)

def contingency_metrics(n_matched, n_missed, n_false_alarms):
    """Calculate contingency table metrics for event detection."""
    total_obs = n_matched + n_missed
    total_forecast = n_matched + n_false_alarms
    
    hit_rate = n_matched / total_obs if total_obs > 0 else np.nan
    false_alarm_ratio = n_false_alarms / total_forecast if total_forecast > 0 else np.nan
    csi = n_matched / (n_matched + n_missed + n_false_alarms) if (n_matched + n_missed + n_false_alarms) > 0 else np.nan
    bias = total_forecast / total_obs if total_obs > 0 else np.nan
    
    return {
        'hit_rate': hit_rate,
        'false_alarm_ratio': false_alarm_ratio,
        'critical_success_index': csi,
        'frequency_bias': bias
    }

metrics_insitu = contingency_metrics(len(matches_insitu), len(unmatched_obs_insitu), len(unmatched_insitu))
metrics_wsa = contingency_metrics(len(matches_wsa), len(unmatched_obs_wsa), len(unmatched_wsa))

print(f"\nHSE Contingency Metrics:")
print(f"  InSitu-HUXt:")
print(f"    Hit Rate (POD):          {metrics_insitu['hit_rate']:.2%}")
print(f"    False Alarm Ratio:       {metrics_insitu['false_alarm_ratio']:.2%}")
print(f"    Critical Success Index:  {metrics_insitu['critical_success_index']:.2%}")
print(f"    Frequency Bias:          {metrics_insitu['frequency_bias']:.2f}")

print(f"  WSA-HUXt:")
print(f"    Hit Rate (POD):          {metrics_wsa['hit_rate']:.2%}")
print(f"    False Alarm Ratio:       {metrics_wsa['false_alarm_ratio']:.2%}")
print(f"    Critical Success Index:  {metrics_wsa['critical_success_index']:.2%}")
print(f"    Frequency Bias:          {metrics_wsa['frequency_bias']:.2f}")


# <codecell> Plot HSE error distributions

fig_hse, axs_hse = plt.subplots(2, 2, figsize=(12, 10))

# Timing error histograms (expanded to full association time window)
timing_bins = np.arange(-80, 85, 10)

ax = axs_hse[0, 0]
if len(timing_errors_insitu) > 0:
    ax.hist(timing_errors_insitu, bins=timing_bins, alpha=0.7, color='#2E86AB', 
            edgecolor='black', linewidth=1.2, label='InSitu-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('HSE timing error [hours]', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'(a) MAE = {timing_stats_insitu["mean_absolute_error"]:.1f} h ; Bias = {timing_stats_insitu["mean_error"]:+.1f} h', 
             fontsize=14)
ax.tick_params(labelsize=13)
ax.legend(fontsize=14)

ax = axs_hse[0, 1]
if len(timing_errors_wsa) > 0:
    ax.hist(timing_errors_wsa, bins=timing_bins, alpha=0.7, color='#A23B72',
            edgecolor='black', linewidth=1.2, label='WSA-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('HSE timing error [hours]', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'(b) MAE = {timing_stats_wsa["mean_absolute_error"]:.1f} h ; Bias = {timing_stats_wsa["mean_error"]:+.1f} h', 
             fontsize=14)
ax.tick_params(labelsize=13)
ax.legend(fontsize=14)

# Peak speed error histograms
speed_bins = np.arange(-300, 350, 50)

ax = axs_hse[1, 0]
if len(speed_errors_insitu) > 0:
    ax.hist(speed_errors_insitu, bins=speed_bins, alpha=0.7, color='#2E86AB',
            edgecolor='black', linewidth=1.2, label='InSitu-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('HSE peak speed error [km/s]', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'(c) MAE = {speed_stats_insitu["mean_absolute_error"]:.1f} km/s ; Bias = {speed_stats_insitu["mean_error"]:+.1f} km/s', 
             fontsize=14)
ax.tick_params(labelsize=13)
ax.legend(fontsize=14)

ax = axs_hse[1, 1]
if len(speed_errors_wsa) > 0:
    ax.hist(speed_errors_wsa, bins=speed_bins, alpha=0.7, color='#A23B72',
            edgecolor='black', linewidth=1.2, label='WSA-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('HSE peak speed error [km/s]', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'(d) MAE = {speed_stats_wsa["mean_absolute_error"]:.1f} km/s ; Bias = {speed_stats_wsa["mean_error"]:+.1f} km/s', 
             fontsize=14)
ax.tick_params(labelsize=13)
ax.legend(fontsize=14)
fig_hse.tight_layout()
fig_hse.savefig(os.path.join(figdir, f'hse_errors_{plot_lead_time}day.pdf'), dpi=150, bbox_inches='tight')
print(f"\nSaved HSE error distributions as hse_errors_{plot_lead_time}day.pdf")


# <codecell> Create HSE demonstration figure for first 6 months of 2017

demo_period_str = f"{dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}"
print("\n" + "="*60)
print(f"CREATING HSE DEMONSTRATION FIGURE ({demo_period_str})")
print("="*60)

# Define demonstration period
demo_start = datetime.datetime(2017, 1, 1)
demo_end = datetime.datetime(2017, 7, 1)

# Filter data for demonstration period
demo_mask_obs = ((obs_smoothed.index >= demo_start) & (obs_smoothed.index < demo_end))
demo_mask_insitu = ((insitu_smoothed.index >= demo_start) & (insitu_smoothed.index < demo_end))
demo_mask_wsa = ((wsa_smoothed.index >= demo_start) & (wsa_smoothed.index < demo_end))

demo_obs = obs_smoothed[demo_mask_obs]
demo_insitu = insitu_smoothed[demo_mask_insitu]
demo_wsa = wsa_smoothed[demo_mask_wsa]

# Filter events for demonstration period
demo_events_obs = [e for e in events_obs if demo_start <= e['start'] < demo_end]
demo_events_insitu = [e for e in events_insitu if demo_start <= e['start'] < demo_end]
demo_events_wsa = [e for e in events_wsa if demo_start <= e['start'] < demo_end]

# Get matches for demonstration period
demo_matches_insitu = [m for m in matches_insitu if demo_start <= m[0]['start'] < demo_end]
demo_matches_wsa = [m for m in matches_wsa if demo_start <= m[0]['start'] < demo_end]

print(f"Demo period events:")
print(f"  Observations: {len(demo_events_obs)}")
print(f"  InSitu-HUXt:  {len(demo_events_insitu)} ({len(demo_matches_insitu)} matched)")
print(f"  WSA-HUXt:     {len(demo_events_wsa)} ({len(demo_matches_wsa)} matched)")

# Create figure with 3 panels
fig_demo, axs_demo = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Helper function to find HSE period indices
def find_hse_period(event, df, rise_hours=48):
    """Find the HSE period: from start to at least 48 hours after start"""
    start_time = event['start']
    peak_time = event['peak']
    
    # HSE period should span at least the rise_hours (48h) from start
    min_end_time = start_time + pd.Timedelta(hours=rise_hours)
    
    # Use whichever is later: 48h after start or the peak time
    end_time = max(min_end_time, peak_time)
    
    # Get indices for the HSE period
    mask = (df.index >= start_time) & (df.index <= end_time)
    return mask

# Color scheme
color_wsa_plot = '#A23B72'
color_obs_plot = 'k'
color_insitu_plot = '#2E86AB'

# Panel 1: WSA-HUXt
ax = axs_demo[0]
ax.plot(demo_wsa.index, demo_wsa['V'], color=color_wsa_plot, linewidth=1.5, label='WSA-HUXt')

# Shade HSE periods and mark max gradient times
for event in demo_events_wsa:
    mask = find_hse_period(event, demo_wsa)
    if mask.sum() > 0:
        ax.fill_between(demo_wsa.index, 0, demo_wsa['V'], where=mask, 
                        alpha=0.3, color=color_wsa_plot, linewidth=0)
    ax.axvline(event['max_gradient_time'], color=color_wsa_plot, linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_ylabel('Speed [km/s]', fontsize=14, fontweight='bold')
ax.set_ylim(250, 750)
ax.tick_params(labelsize=12)
ax.legend(loc='upper right', fontsize=14)
ax.grid(True, alpha=0.3)
ax.text(0.01, 0.95, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold',
        verticalalignment='top')

# Panel 2: Observations
ax = axs_demo[1]
ax.plot(demo_obs.index, demo_obs['V'], color=color_obs_plot, linewidth=1.5, label='Observations')

# Shade HSE periods and mark max gradient times
for event in demo_events_obs:
    mask = find_hse_period(event, demo_obs)
    if mask.sum() > 0:
        ax.fill_between(demo_obs.index, 0, demo_obs['V'], where=mask, 
                        alpha=0.2, color='gray', linewidth=0)
    ax.axvline(event['max_gradient_time'], color=color_obs_plot, linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_ylabel('Speed [km/s]', fontsize=14, fontweight='bold')
ax.set_ylim(250, 750)
ax.tick_params(labelsize=12)
ax.legend(loc='upper right', fontsize=14)
ax.grid(True, alpha=0.3)
ax.text(0.01, 0.95, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold',
        verticalalignment='top')

# Panel 3: InSitu-HUXt
ax = axs_demo[2]
ax.plot(demo_insitu.index, demo_insitu['V'], color=color_insitu_plot, linewidth=1.5, label='InSitu-HUXt')

# Shade HSE periods and mark max gradient times
for event in demo_events_insitu:
    mask = find_hse_period(event, demo_insitu)
    if mask.sum() > 0:
        ax.fill_between(demo_insitu.index, 0, demo_insitu['V'], where=mask, 
                        alpha=0.3, color=color_insitu_plot, linewidth=0)
    ax.axvline(event['max_gradient_time'], color=color_insitu_plot, linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_ylabel('Speed [km/s]', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylim(250, 750)
ax.tick_params(labelsize=12)
ax.legend(loc='upper right', fontsize=14)
ax.grid(True, alpha=0.3)
ax.text(0.01, 0.95, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold',
        verticalalignment='top')

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Draw connecting lines for matched events using max_gradient_time
# This aligns with the dashed vertical lines on each panel
for match in demo_matches_wsa:
    obs_event = match[0]
    wsa_event = match[1]
    
    # Draw line from WSA panel to Obs panel
    # Convert times to x-coordinates
    obs_time = mdates.date2num(obs_event['max_gradient_time'])
    wsa_time = mdates.date2num(wsa_event['max_gradient_time'])
    
    # Create connection between top and middle panels
    con = plt.matplotlib.patches.ConnectionPatch(
        xyA=(wsa_time, 250), coordsA=axs_demo[0].transData,
        xyB=(obs_time, 750), coordsB=axs_demo[1].transData,
        color=color_wsa_plot, alpha=0.4, linewidth=1.5, linestyle=':')
    fig_demo.add_artist(con)

for match in demo_matches_insitu:
    obs_event = match[0]
    insitu_event = match[1]
    
    # Draw line from Obs panel to InSitu panel
    obs_time = mdates.date2num(obs_event['max_gradient_time'])
    insitu_time = mdates.date2num(insitu_event['max_gradient_time'])
    
    # Create connection between middle and bottom panels
    con = plt.matplotlib.patches.ConnectionPatch(
        xyA=(obs_time, 250), coordsA=axs_demo[1].transData,
        xyB=(insitu_time, 750), coordsB=axs_demo[2].transData,
        color=color_insitu_plot, alpha=0.4, linewidth=1.5, linestyle=':')
    fig_demo.add_artist(con)

fig_demo.tight_layout()
fig_demo.savefig(os.path.join(figdir, f'hse_demonstration_2017_H1.pdf'), 
                dpi=150, bbox_inches='tight')
print(f"\nSaved HSE demonstration figure as hse_demonstration_2017_H1.pdf")


# <codecell> Create HSE summary table

print("\n" + "="*60)
print("HSE ANALYSIS SUMMARY TABLE")
print("="*60)

summary_table = pd.DataFrame({
    'Metric': [
        'Events Detected',
        'Matched Events',
        'Missed Events', 
        'False Alarms',
        'Hit Rate (POD)',
        'False Alarm Ratio',
        'Critical Success Index',
        'Timing MAE (hours)',
        'Timing Bias (hours)',
        'Peak Speed MAE (km/s)',
        'Peak Speed Bias (km/s)'
    ],
    'InSitu-HUXt': [
        len(events_insitu),
        len(matches_insitu),
        len(unmatched_obs_insitu),
        len(unmatched_insitu),
        f"{metrics_insitu['hit_rate']:.1%}",
        f"{metrics_insitu['false_alarm_ratio']:.1%}",
        f"{metrics_insitu['critical_success_index']:.1%}",
        f"{timing_stats_insitu['mean_absolute_error']:.1f}",
        f"{timing_stats_insitu['mean_error']:+.1f}",
        f"{speed_stats_insitu['mean_absolute_error']:.1f}",
        f"{speed_stats_insitu['mean_error']:+.1f}"
    ],
    'WSA-HUXt': [
        len(events_wsa),
        len(matches_wsa),
        len(unmatched_obs_wsa),
        len(unmatched_wsa),
        f"{metrics_wsa['hit_rate']:.1%}",
        f"{metrics_wsa['false_alarm_ratio']:.1%}",
        f"{metrics_wsa['critical_success_index']:.1%}",
        f"{timing_stats_wsa['mean_absolute_error']:.1f}",
        f"{timing_stats_wsa['mean_error']:+.1f}",
        f"{speed_stats_wsa['mean_absolute_error']:.1f}",
        f"{speed_stats_wsa['mean_error']:+.1f}"
    ]
})

# Add observation count for reference
summary_table.loc[-1] = ['Observed Events', len(events_obs), len(events_obs)]
summary_table.index = summary_table.index + 1
summary_table = summary_table.sort_index()

print(summary_table.to_string(index=False))

# Save summary table to LaTeX with proper escaping
latex_output = summary_table.to_latex(index=False, escape=False, column_format='lcc')
# Replace booktabs commands with basic LaTeX and escape % signs
latex_output = latex_output.replace('\\toprule', '\\hline')
latex_output = latex_output.replace('\\midrule', '\\hline')
latex_output = latex_output.replace('\\bottomrule', '\\hline')
# Escape percentage signs (% is a comment in LaTeX)
latex_output = latex_output.replace('%', '\\%')
with open(os.path.join(figdir, f'hse_summary_{plot_lead_time}day.tex'), 'w') as f:
    f.write(latex_output)
print(f"\nSaved HSE summary table as hse_summary_{plot_lead_time}day.tex")


# <codecell> Final HSE count analysis using individual forecast files
# =============================================================================
# For each forecast initialization date, load the full forecast time series,
# make it periodic by repeating V three times, then detect HSEs on that wrapped
# series. Report total counts and normalized totals.
# =============================================================================

print("\n" + "="*60)
print("FINAL PER-FORECAST HSE COUNT ANALYSIS")
print("="*60)


def _load_single_forecast(filepath):
    """Load one forecast file and return a dataframe with datetime index and V."""
    ts = pd.read_csv(filepath, index_col=0, parse_dates=["time"])
    # Ensure true hourly bins before smoothing; raw files include sub-second offsets.
    time_hourly = pd.to_datetime(ts['time'], errors='coerce').dt.round('h')
    df = pd.DataFrame({
        'datetime': time_hourly,
        'V': ts['vsw'].values
    })
    df = df.dropna(subset=['datetime'])
    df = df.groupby('datetime').mean().sort_index()
    return df


def _repeat_series_three_times(df):
    """Repeat a forecast time series 3x with constant cadence for periodic HSE detection."""
    if len(df) < 2:
        return None

    times = df.index.to_numpy()
    vals = df['V'].to_numpy()

    dt_seconds = np.median(np.diff(times).astype('timedelta64[s]').astype(float))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        dt_seconds = 3600.0

    dt = pd.to_timedelta(dt_seconds, unit='s')
    n = len(vals)

    wrapped_times = pd.to_datetime(times[0]) + np.arange(3 * n) * dt
    wrapped_vals = np.tile(vals, 3)

    wrapped_df = pd.DataFrame({'datetime': wrapped_times, 'V': wrapped_vals})
    wrapped_df = wrapped_df.set_index('datetime')
    return wrapped_df


def _count_hse_in_forecast_file(filepath):
    """Count HSEs in one forecast file after 3x periodic wrapping."""
    try:
        single_df = _load_single_forecast(filepath)
        wrapped_df = _repeat_series_three_times(single_df)
        if wrapped_df is None:
            return np.nan

        smoothed = hse.smooth_8h(wrapped_df.reset_index(), date_col='datetime',
                                 window_hours=hse_smoothing_window)
        events = hse.detect_hss_events(
            smoothed,
            speed_col='V',
            rise_threshold=hse_rise_threshold,
            rise_hours=hse_rise_hours,
            min_separation_hours=hse_min_separation
        )
        return len(events)
    except Exception:
        return np.nan


def _collect_hse_counts(leadtime_dir, file_prefix, progress_label=''):
    """Collect per-forecast HSE counts over all available forecast files."""
    rows = []
    file_pattern = os.path.join(leadtime_dir, '*', f'{file_prefix}_*.dat')
    filepaths = sorted(glob.glob(file_pattern))
    total_files = len(filepaths)

    if total_files == 0:
        print(f"No files found for pattern: {file_pattern}")
        return pd.DataFrame(rows)

    label = progress_label if progress_label else file_prefix
    print(f"Processing {total_files} {label} forecast files...")

    for i, filepath in enumerate(filepaths, start=1):
        basename = os.path.basename(filepath)
        date_token = basename.replace(f'{file_prefix}_', '').replace('.dat', '')

        try:
            ftime = pd.to_datetime(date_token, format='%Y%m%d')
        except Exception:
            ftime = pd.NaT

        hse_count = _count_hse_in_forecast_file(filepath)
        rows.append({'init_time': ftime, 'file': filepath, 'hse_count': hse_count})

        if (per_forecast_hse_progress_every > 0 and
                (i % per_forecast_hse_progress_every == 0 or i == total_files)):
            print(f"  {label}: processed {i}/{total_files}")

    return pd.DataFrame(rows)


def _load_cached_hse_counts(csv_path, label):
    """Load cached per-forecast HSE counts, returning None if unavailable/invalid."""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=['init_time'])
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return None

    if 'hse_count' not in df.columns:
        return None

    print(f"Loaded cached {label} HSE counts from {csv_path}")
    return df


# Count HSEs forecast-by-forecast for both models
insitu_counts_path = os.path.join(datadir, 'hse_counts_per_forecast_insitu.csv')
wsa_counts_path = os.path.join(datadir, 'hse_counts_per_forecast_wsa.csv')

if recompute_per_forecast_hse_counts_now:
    hse_counts_insitu = _collect_hse_counts(ishuxt_dir, 'InSitu_Earth', progress_label='InSitu-HUXt')
    hse_counts_wsa = _collect_hse_counts(wsahuxt_dir, 'HUXt_Earth', progress_label='WSA-HUXt')
else:
    hse_counts_insitu = _load_cached_hse_counts(insitu_counts_path, 'InSitu-HUXt')
    hse_counts_wsa = _load_cached_hse_counts(wsa_counts_path, 'WSA-HUXt')

    if hse_counts_insitu is None or hse_counts_wsa is None:
        print("Cached HSE count files missing/invalid. Recomputing now...")
        hse_counts_insitu = _collect_hse_counts(ishuxt_dir, 'InSitu_Earth', progress_label='InSitu-HUXt')
        hse_counts_wsa = _collect_hse_counts(wsahuxt_dir, 'HUXt_Earth', progress_label='WSA-HUXt')
    else:
        print("Using cached per-forecast HSE count files (set recompute_per_forecast_hse_counts_now=True to refresh).")

# Total counts over all individual forecasts
total_hse_insitu_raw = np.nansum(hse_counts_insitu['hse_count'].values)
total_hse_wsa_raw = np.nansum(hse_counts_wsa['hse_count'].values)

# Requested normalization: divide by 3 for wrapping and 27 for overlapping forecasts
normalization_factor = 3 * 27
total_hse_insitu_norm = total_hse_insitu_raw / normalization_factor
total_hse_wsa_norm = total_hse_wsa_raw / normalization_factor

print(f"InSitu-HUXt forecasts processed: {len(hse_counts_insitu)}")
print(f"WSA-HUXt forecasts processed:    {len(hse_counts_wsa)}")

print("\nTotal HSE counts from individual forecast series (after 3x wrapping per forecast):")
print(f"  InSitu-HUXt raw total: {total_hse_insitu_raw:.0f}")
print(f"  WSA-HUXt raw total:    {total_hse_wsa_raw:.0f}")

print("\nNormalized totals (divide by 3 for wrapping and 27 for overlap):")
print(f"  InSitu-HUXt: {total_hse_insitu_norm:.2f}")
print(f"  WSA-HUXt:    {total_hse_wsa_norm:.2f}")

# Save per-forecast counts for traceability
hse_counts_insitu.to_csv(insitu_counts_path, index=False)
hse_counts_wsa.to_csv(wsa_counts_path, index=False)
print("\nSaved per-forecast HSE counts to data directory")


# <codecell> Dynamic Time Warping Analysis
# =============================================================================
# Use DTW to measure similarity between forecast and observation time series
# DTW allows for temporal distortions and is useful for quantifying timing errors
# =============================================================================


print("\n" + "="*60)
print(f"DYNAMIC TIME WARPING ANALYSIS ({plot_lead_time}-day lead time)")
print("="*60)


try:
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    has_fastdtw = True
except ImportError:
    has_fastdtw = False


try:
    from dtaidistance import dtw
    has_dtaidistance = True
except ImportError:
    has_dtaidistance = False


if not has_fastdtw and not has_dtaidistance:
    print("Warning: No DTW library available. Install fastdtw or dtaidistance:")
    print("  pip install fastdtw")
    print("  or")
    print("  pip install dtaidistance")


if has_fastdtw or has_dtaidistance:
    # Start with a shorter test period (first 6 months of 2017) for visualization
    print("\nTesting DTW on first 6 months of 2017...")
    demo_start_dtw = datetime.datetime(2017, 1, 1)
    demo_end_dtw = datetime.datetime(2017, 7, 1)
    
    # For forecasts, need to align with observation times
    insitu_times = pd.to_datetime(combined_for_list[plot_lead_time-1]['time'].values)
    wsa_times = pd.to_datetime(combined_for_wsa_list[plot_lead_time-1]['time'].values)
    
    # Resample to hourly to match observation cadence
    insitu_df_dtw = pd.DataFrame({
        'time': insitu_times.round('h'),
        'V': combined_for_list[plot_lead_time-1]['vsw'].values
    }).groupby('time').mean().reset_index()
    
    wsa_df_dtw = pd.DataFrame({
        'time': wsa_times.round('h'),
        'V': combined_for_wsa_list[plot_lead_time-1]['vsw'].values
    }).groupby('time').mean().reset_index()
    
    # Create common hourly time grid for demo period
    common_times_hourly = pd.date_range(start=demo_start_dtw, end=demo_end_dtw, freq='h', inclusive='left')
    
    # Merge all datasets onto hourly grid
    obs_demo_df = pd.DataFrame({'datetime': common_times_hourly})
    obs_demo_df = obs_demo_df.merge(omni_noicmes[['datetime', 'V']], on='datetime', how='left')
    
    insitu_demo_df = pd.DataFrame({'datetime': common_times_hourly})
    insitu_demo_df = insitu_demo_df.merge(insitu_df_dtw.rename(columns={'time': 'datetime'})[['datetime', 'V']], 
                                          on='datetime', how='left')
    
    wsa_demo_df = pd.DataFrame({'datetime': common_times_hourly})
    wsa_demo_df = wsa_demo_df.merge(wsa_df_dtw.rename(columns={'time': 'datetime'})[['datetime', 'V']], 
                                     on='datetime', how='left')
    
    # Linearly interpolate through any data gaps (NaNs) before filtering
    print(f"\nInterpolating data gaps in demo period time series:")
    print(f"  Observations: {obs_demo_df['V'].isna().sum()} NaN values before interpolation")
    print(f"  InSitu-HUXt:  {insitu_demo_df['V'].isna().sum()} NaN values before interpolation")
    print(f"  WSA-HUXt:     {wsa_demo_df['V'].isna().sum()} NaN values before interpolation")
    
    obs_demo_df['V'] = obs_demo_df['V'].interpolate(method='linear', limit_direction='both')
    insitu_demo_df['V'] = insitu_demo_df['V'].interpolate(method='linear', limit_direction='both')
    wsa_demo_df['V'] = wsa_demo_df['V'].interpolate(method='linear', limit_direction='both')
    
    print(f"  Observations: {obs_demo_df['V'].isna().sum()} NaN values after interpolation")
    print(f"  InSitu-HUXt:  {insitu_demo_df['V'].isna().sum()} NaN values after interpolation")
    print(f"  WSA-HUXt:     {wsa_demo_df['V'].isna().sum()} NaN values after interpolation")
    
    # Forward/backward fill any remaining NaNs at edges
    obs_demo_df['V'] = obs_demo_df['V'].ffill().bfill()
    insitu_demo_df['V'] = insitu_demo_df['V'].ffill().bfill()
    wsa_demo_df['V'] = wsa_demo_df['V'].ffill().bfill()
    
    # Resample to 8-hourly resolution
    obs_demo_df = obs_demo_df.set_index('datetime').resample('8h').mean().reset_index()
    insitu_demo_df = insitu_demo_df.set_index('datetime').resample('8h').mean().reset_index()
    wsa_demo_df = wsa_demo_df.set_index('datetime').resample('8h').mean().reset_index()
    
    # Extract 8-hourly arrays for DTW
    obs_dtw_demo = obs_demo_df['V'].values
    obs_times_demo = obs_demo_df['datetime'].values
    insitu_dtw_demo = insitu_demo_df['V'].values
    insitu_times_demo = insitu_demo_df['datetime'].values
    wsa_dtw_demo = wsa_demo_df['V'].values
    wsa_times_demo = wsa_demo_df['datetime'].values
    
    print(f"\nDemo period time series lengths ({dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}, 8h resolution):")
    print(f"  Observations: {len(obs_dtw_demo)} ({np.isnan(obs_dtw_demo).sum()} NaN values)")
    print(f"  InSitu-HUXt:  {len(insitu_dtw_demo)} ({np.isnan(insitu_dtw_demo).sum()} NaN values)")
    print(f"  WSA-HUXt:     {len(wsa_dtw_demo)} ({np.isnan(wsa_dtw_demo).sum()} NaN values)")
    print(f"  (All series on common 8-hourly grid with gaps interpolated)")
    
    # Compute DTW distance and alignment on demo period
    # Use dtaidistance for proper Sakoe-Chiba band constraint if available, else fastdtw
    if has_dtaidistance:
        print("\nUsing dtaidistance library with Sakoe-Chiba band constraint...")
        
        # Normalize series for fair comparison and ensure 1-D arrays
        obs_norm_demo = ((obs_dtw_demo - np.mean(obs_dtw_demo)) / np.std(obs_dtw_demo)).ravel()
        insitu_norm_demo = ((insitu_dtw_demo - np.mean(insitu_dtw_demo)) / np.std(insitu_dtw_demo)).ravel()
        wsa_norm_demo = ((wsa_dtw_demo - np.mean(wsa_dtw_demo)) / np.std(wsa_dtw_demo)).ravel()
        
        # Compute DTW distance with configured window constraint
        # This enforces strict Sakoe-Chiba band: |i-j| <= dtw_window_points
        print("Computing DTW for InSitu-HUXt...")
        dtw_dist_insitu_demo = dtw.distance(obs_norm_demo, insitu_norm_demo, window=dtw_window_points)
        print("Computing DTW path for InSitu-HUXt...")
        dtw_path_insitu = dtw.warping_path(obs_norm_demo, insitu_norm_demo, window=dtw_window_points)
        
        print("Computing DTW for WSA-HUXt...")
        dtw_dist_wsa_demo = dtw.distance(obs_norm_demo, wsa_norm_demo, window=dtw_window_points)
        print("Computing DTW path for WSA-HUXt...")
        dtw_path_wsa = dtw.warping_path(obs_norm_demo, wsa_norm_demo, window=dtw_window_points)
        
        # Normalize by sequence length for interpretability
        dtw_dist_insitu_norm_demo = dtw_dist_insitu_demo / len(obs_norm_demo)
        dtw_dist_wsa_norm_demo = dtw_dist_wsa_demo / len(obs_norm_demo)
        
        demo_period_str = f"{dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}"
        print(f"\nDTW Distance (normalized, {demo_period_str}):")
        print(f"  InSitu-HUXt vs Obs: {dtw_dist_insitu_norm_demo:.4f}")
        print(f"  WSA-HUXt vs Obs:    {dtw_dist_wsa_norm_demo:.4f}")
        
        # Compute average time offset from warping path
        dtw_path_insitu_arr = np.array(dtw_path_insitu)
        dtw_path_wsa_arr = np.array(dtw_path_wsa)
        
        # Calculate time offsets: (forecast_idx - obs_idx) * 8 hours
        # Positive = forecast late, Negative = forecast early
        time_offset_insitu = (dtw_path_insitu_arr[:, 1] - dtw_path_insitu_arr[:, 0]) * 8.0
        time_offset_wsa = (dtw_path_wsa_arr[:, 1] - dtw_path_wsa_arr[:, 0]) * 8.0
        
        avg_offset_insitu = np.mean(time_offset_insitu)
        avg_offset_wsa = np.mean(time_offset_wsa)
        avg_offset_insitu_mag = np.mean(np.abs(time_offset_insitu))
        avg_offset_wsa_mag = np.mean(np.abs(time_offset_wsa))
        max_offset_insitu = np.max(np.abs(time_offset_insitu))
        max_offset_wsa = np.max(np.abs(time_offset_wsa))
        
        demo_period_str = f"{dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}"
        print(f"\nAverage Time Offset ({demo_period_str}):")
        print(f"  InSitu-HUXt: {avg_offset_insitu:.1f} hours (magnitude: {avg_offset_insitu_mag:.1f} hours, max: {max_offset_insitu:.0f} hours)")
        print(f"  WSA-HUXt:    {avg_offset_wsa:.1f} hours (magnitude: {avg_offset_wsa_mag:.1f} hours, max: {max_offset_wsa:.0f} hours)")
        
        # Extract aligned speed values using the warping path
        obs_aligned_insitu = obs_dtw_demo[dtw_path_insitu_arr[:, 0]]
        insitu_aligned = insitu_dtw_demo[dtw_path_insitu_arr[:, 1]]
        obs_aligned_wsa = obs_dtw_demo[dtw_path_wsa_arr[:, 0]]
        wsa_aligned = wsa_dtw_demo[dtw_path_wsa_arr[:, 1]]
        
        # Compute velocity differences for demo period (absolute speed errors after optimal warping)
        # Error = model - obs (positive means model overestimates)
        dv_insitu_aligned_demo = insitu_aligned - obs_aligned_insitu
        dv_wsa_aligned_demo = wsa_aligned - obs_aligned_wsa
        
        # Compute metrics on aligned data (actual km/s differences after optimal warping)
        # Error = model - obs (positive means model overestimates)
        rmse_aligned_insitu = np.sqrt(np.mean((insitu_aligned - obs_aligned_insitu)**2))
        mae_aligned_insitu = np.mean(np.abs(insitu_aligned - obs_aligned_insitu))
        corr_aligned_insitu = np.corrcoef(obs_aligned_insitu, insitu_aligned)[0, 1]
        
        rmse_aligned_wsa = np.sqrt(np.mean((wsa_aligned - obs_aligned_wsa)**2))
        mae_aligned_wsa = np.mean(np.abs(wsa_aligned - obs_aligned_wsa))
        corr_aligned_wsa = np.corrcoef(obs_aligned_wsa, wsa_aligned)[0, 1]
        
        # Compare to non-aligned metrics (point-by-point)
        rmse_noalign_insitu = np.sqrt(np.mean((insitu_dtw_demo - obs_dtw_demo)**2))
        mae_noalign_insitu = np.mean(np.abs(insitu_dtw_demo - obs_dtw_demo))
        corr_noalign_insitu = np.corrcoef(obs_dtw_demo, insitu_dtw_demo)[0, 1]
        
        rmse_noalign_wsa = np.sqrt(np.mean((wsa_dtw_demo - obs_dtw_demo)**2))
        mae_noalign_wsa = np.mean(np.abs(wsa_dtw_demo - obs_dtw_demo))
        corr_noalign_wsa = np.corrcoef(obs_dtw_demo, wsa_dtw_demo)[0, 1]
        
        print(f"\nSpeed Errors After Optimal DTW Alignment ({dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}):")
        print(f"  InSitu-HUXt: RMSE={rmse_aligned_insitu:.1f} km/s, MAE={mae_aligned_insitu:.1f} km/s, r={corr_aligned_insitu:.3f}")
        print(f"  WSA-HUXt:    RMSE={rmse_aligned_wsa:.1f} km/s, MAE={mae_aligned_wsa:.1f} km/s, r={corr_aligned_wsa:.3f}")
        
        print(f"\nSpeed Errors Without Alignment (point-by-point, {dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}):")
        print(f"  InSitu-HUXt: RMSE={rmse_noalign_insitu:.1f} km/s, MAE={mae_noalign_insitu:.1f} km/s, r={corr_noalign_insitu:.3f}")
        print(f"  WSA-HUXt:    RMSE={rmse_noalign_wsa:.1f} km/s, MAE={mae_noalign_wsa:.1f} km/s, r={corr_noalign_wsa:.3f}")
        
        print(f"\nImprovement from DTW Alignment:")
        print(f"  InSitu-HUXt: RMSE reduced by {(1-rmse_aligned_insitu/rmse_noalign_insitu)*100:.1f}%, MAE by {(1-mae_aligned_insitu/mae_noalign_insitu)*100:.1f}%")
        print(f"  WSA-HUXt:    RMSE reduced by {(1-rmse_aligned_wsa/rmse_noalign_wsa)*100:.1f}%, MAE by {(1-mae_aligned_wsa/mae_noalign_wsa)*100:.1f}%")
    
    elif has_fastdtw:
        print("\nUsing FastDTW algorithm...")
        print("Warning: FastDTW radius constraint may not strictly enforce Sakoe-Chiba band")
        
        # Normalize series for fair comparison and ensure 1-D arrays
        obs_norm_demo = ((obs_dtw_demo - np.mean(obs_dtw_demo)) / np.std(obs_dtw_demo)).ravel()
        insitu_norm_demo = ((insitu_dtw_demo - np.mean(insitu_dtw_demo)) / np.std(insitu_dtw_demo)).ravel()
        wsa_norm_demo = ((wsa_dtw_demo - np.mean(wsa_dtw_demo)) / np.std(wsa_dtw_demo)).ravel()
        
        # Compute DTW distance with configured radius constraint
        print("Computing DTW for InSitu-HUXt...")
        dtw_dist_insitu_demo, dtw_path_insitu = fastdtw(obs_norm_demo, insitu_norm_demo, 
                                                         radius=dtw_window_points)
        print("Computing DTW for WSA-HUXt...")
        dtw_dist_wsa_demo, dtw_path_wsa = fastdtw(obs_norm_demo, wsa_norm_demo, 
                                                   radius=dtw_window_points)
        
        # Normalize by sequence length for interpretability
        dtw_dist_insitu_norm_demo = dtw_dist_insitu_demo / len(obs_norm_demo)
        dtw_dist_wsa_norm_demo = dtw_dist_wsa_demo / len(obs_norm_demo)
        
        demo_period_str = f"{dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}"
        print(f"\nDTW Distance (normalized, {demo_period_str}):")
        print(f"  InSitu-HUXt vs Obs: {dtw_dist_insitu_norm_demo:.4f}")
        print(f"  WSA-HUXt vs Obs:    {dtw_dist_wsa_norm_demo:.4f}")
        
        # Compute average time offset from warping path
        dtw_path_insitu_arr = np.array(dtw_path_insitu)
        dtw_path_wsa_arr = np.array(dtw_path_wsa)
        
        # Calculate time offsets: (forecast_idx - obs_idx) * 8 hours
        # Positive = forecast late, Negative = forecast early
        time_offset_insitu = (dtw_path_insitu_arr[:, 1] - dtw_path_insitu_arr[:, 0]) * 8.0
        time_offset_wsa = (dtw_path_wsa_arr[:, 1] - dtw_path_wsa_arr[:, 0]) * 8.0
        
        avg_offset_insitu = np.mean(time_offset_insitu)
        avg_offset_wsa = np.mean(time_offset_wsa)
        avg_offset_insitu_mag = np.mean(np.abs(time_offset_insitu))
        avg_offset_wsa_mag = np.mean(np.abs(time_offset_wsa))
        max_offset_insitu = np.max(np.abs(time_offset_insitu))
        max_offset_wsa = np.max(np.abs(time_offset_wsa))
        
        demo_period_str = f"{dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}"
        print(f"\nAverage Time Offset ({demo_period_str}):")
        print(f"  InSitu-HUXt: {avg_offset_insitu:.1f} hours (magnitude: {avg_offset_insitu_mag:.1f} hours, max: {max_offset_insitu:.0f} hours)")
        print(f"  WSA-HUXt:    {avg_offset_wsa:.1f} hours (magnitude: {avg_offset_wsa_mag:.1f} hours, max: {max_offset_wsa:.0f} hours)")
        
        # Extract aligned speed values using the warping path
        obs_aligned_insitu = obs_dtw_demo[dtw_path_insitu_arr[:, 0]]
        insitu_aligned = insitu_dtw_demo[dtw_path_insitu_arr[:, 1]]
        obs_aligned_wsa = obs_dtw_demo[dtw_path_wsa_arr[:, 0]]
        wsa_aligned = wsa_dtw_demo[dtw_path_wsa_arr[:, 1]]
        
        # Compute velocity differences for demo period (absolute speed errors after optimal warping)
        # Error = model - obs (positive means model overestimates)
        dv_insitu_aligned_demo = insitu_aligned - obs_aligned_insitu
        dv_wsa_aligned_demo = wsa_aligned - obs_aligned_wsa
        
        # Compute metrics on aligned data (actual km/s differences after optimal warping)
        rmse_aligned_insitu = np.sqrt(np.mean((insitu_aligned - obs_aligned_insitu)**2))
        mae_aligned_insitu = np.mean(np.abs(insitu_aligned - obs_aligned_insitu))
        corr_aligned_insitu = np.corrcoef(obs_aligned_insitu, insitu_aligned)[0, 1]
        
        rmse_aligned_wsa = np.sqrt(np.mean((wsa_aligned - obs_aligned_wsa)**2))
        mae_aligned_wsa = np.mean(np.abs(wsa_aligned - obs_aligned_wsa))
        corr_aligned_wsa = np.corrcoef(obs_aligned_wsa, wsa_aligned)[0, 1]
        
        # Compare to non-aligned metrics (point-by-point)
        rmse_noalign_insitu = np.sqrt(np.mean((insitu_dtw_demo - obs_dtw_demo)**2))
        mae_noalign_insitu = np.mean(np.abs(insitu_dtw_demo - obs_dtw_demo))
        corr_noalign_insitu = np.corrcoef(obs_dtw_demo, insitu_dtw_demo)[0, 1]
        
        rmse_noalign_wsa = np.sqrt(np.mean((wsa_dtw_demo - obs_dtw_demo)**2))
        mae_noalign_wsa = np.mean(np.abs(wsa_dtw_demo - obs_dtw_demo))
        corr_noalign_wsa = np.corrcoef(obs_dtw_demo, wsa_dtw_demo)[0, 1]
        
        print(f"\nSpeed Errors After Optimal DTW Alignment ({dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}):")
        print(f"  InSitu-HUXt: RMSE={rmse_aligned_insitu:.1f} km/s, MAE={mae_aligned_insitu:.1f} km/s, r={corr_aligned_insitu:.3f}")
        print(f"  WSA-HUXt:    RMSE={rmse_aligned_wsa:.1f} km/s, MAE={mae_aligned_wsa:.1f} km/s, r={corr_aligned_wsa:.3f}")
        
        print(f"\nSpeed Errors Without Alignment (point-by-point, {dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}):")
        print(f"  InSitu-HUXt: RMSE={rmse_noalign_insitu:.1f} km/s, MAE={mae_noalign_insitu:.1f} km/s, r={corr_noalign_insitu:.3f}")
        print(f"  WSA-HUXt:    RMSE={rmse_noalign_wsa:.1f} km/s, MAE={mae_noalign_wsa:.1f} km/s, r={corr_noalign_wsa:.3f}")
        
        print(f"\nImprovement from DTW Alignment:")
        print(f"  InSitu-HUXt: RMSE reduced by {(1-rmse_aligned_insitu/rmse_noalign_insitu)*100:.1f}%, MAE by {(1-mae_aligned_insitu/mae_noalign_insitu)*100:.1f}%")
        print(f"  WSA-HUXt:    RMSE reduced by {(1-rmse_aligned_wsa/rmse_noalign_wsa)*100:.1f}%, MAE by {(1-mae_aligned_wsa/mae_noalign_wsa)*100:.1f}%")
    
    # Create visualization showing DTW alignment (works for both dtaidistance and fastdtw)
    if has_dtaidistance or has_fastdtw:
        print("\nCreating DTW alignment visualization...")
        fig_dtw = plt.figure(figsize=(16, 12))
        gs = fig_dtw.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Row 1: InSitu-HUXt vs Obs time series (full width)
        ax1 = fig_dtw.add_subplot(gs[0, :])
        ax1.plot(obs_times_demo, obs_dtw_demo, 'k-', linewidth=2.5, alpha=0.7, label='Observations')
        ax1.plot(insitu_times_demo, insitu_dtw_demo, color=color_insitu, linewidth=2.5, alpha=0.7, 
                label='InSitu-HUXt')
        
        # Add mapping lines between matched points (show all paths)
        for i in range(len(dtw_path_insitu_arr)):
            obs_idx = dtw_path_insitu_arr[i, 0]
            insitu_idx = dtw_path_insitu_arr[i, 1]
            ax1.plot([obs_times_demo[obs_idx], insitu_times_demo[insitu_idx]], 
                    [obs_dtw_demo[obs_idx], insitu_dtw_demo[insitu_idx]], 
                    'gray', linewidth=0.5, alpha=0.4)
        
        ax1.set_ylabel('Speed [km/s]', fontsize=16)
        ax1.set_title(f'(a) InSitu-HUXt vs Observations (DTW aligned, avg offset={avg_offset_insitu:.1f}h)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=14, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Row 2: WSA-HUXt vs Obs time series (full width)
        ax2 = fig_dtw.add_subplot(gs[1, :])
        ax2.plot(obs_times_demo, obs_dtw_demo, 'k-', linewidth=2.5, alpha=0.7, label='Observations')
        ax2.plot(wsa_times_demo, wsa_dtw_demo, color=color_wsa, linewidth=2.5, alpha=0.7, 
                label='WSA-HUXt')
        
        # Add mapping lines for WSA-HUXt (show all paths)
        for i in range(len(dtw_path_wsa_arr)):
            obs_idx = dtw_path_wsa_arr[i, 0]
            wsa_idx = dtw_path_wsa_arr[i, 1]
            ax2.plot([obs_times_demo[obs_idx], wsa_times_demo[wsa_idx]], 
                    [obs_dtw_demo[obs_idx], wsa_dtw_demo[wsa_idx]], 
                    'gray', linewidth=0.5, alpha=0.4)
        
        ax2.set_ylabel('Speed [km/s]', fontsize=16)
        ax2.set_title(f'(b) WSA-HUXt vs Observations (DTW aligned, avg offset={avg_offset_wsa:.1f}h)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=14, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Row 3 left: DTW alignment path for InSitu-HUXt
        ax3 = fig_dtw.add_subplot(gs[2, 0])
        ax3.plot(dtw_path_insitu_arr[:, 0] * 8, dtw_path_insitu_arr[:, 1] * 8, 'o-', 
                color=color_insitu, markersize=1.0, linewidth=1.5, alpha=0.3, label='InSitu-HUXt')
        ax3.plot([0, len(obs_norm_demo) * 8], [0, len(insitu_norm_demo) * 8], 'k--', 
                linewidth=2.0, alpha=0.5, label='Perfect alignment (no offset)')
        ax3.set_xlabel('Observation Time [hours]', fontsize=16)
        ax3.set_ylabel('InSitu-HUXt Time [hours]', fontsize=16)
        ax3.set_title(f'(c) avg magnitude={avg_offset_insitu_mag:.1f}h, max={max_offset_insitu:.0f}h', 
                    fontsize=14, fontweight='bold')
        ax3.legend(fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Row 3 right: DTW alignment path for WSA-HUXt
        ax4 = fig_dtw.add_subplot(gs[2, 1])
        ax4.plot(dtw_path_wsa_arr[:, 0] * 8, dtw_path_wsa_arr[:, 1] * 8, 'o-', 
                color=color_wsa, markersize=1.0, linewidth=1.5, alpha=0.3, label='WSA-HUXt')
        ax4.plot([0, len(obs_norm_demo) * 8], [0, len(wsa_norm_demo) * 8], 'k--', 
                linewidth=2.0, alpha=0.5, label='Perfect alignment (no offset)')
        ax4.set_xlabel('Observation Time [hours]', fontsize=16)
        ax4.set_ylabel('WSA-HUXt Time [hours]', fontsize=16)
        ax4.set_title(f'(d) avg magnitude={avg_offset_wsa_mag:.1f}h, max={max_offset_wsa:.0f}h', 
                    fontsize=14, fontweight='bold')
        ax4.legend(fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        fig_dtw.tight_layout()
        fig_dtw.savefig(os.path.join(figdir, f'dtw_alignment_{plot_lead_time}day_2017H1.pdf'), 
                       dpi=150, bbox_inches='tight')
        print(f"Saved DTW alignment visualization to dtw_alignment_{plot_lead_time}day_2017H1.pdf")
    
    # Save demo results
    dtw_summary_demo = pd.DataFrame({
        'Model': ['InSitu-HUXt', 'WSA-HUXt'],
        'DTW_Distance_2017H1': [dtw_dist_insitu_norm_demo, dtw_dist_wsa_norm_demo],
        'Avg_Time_Offset_hours': [avg_offset_insitu, avg_offset_wsa],
        'Max_Time_Offset_hours': [max_offset_insitu, max_offset_wsa],
        'RMSE_Aligned_kms': [rmse_aligned_insitu, rmse_aligned_wsa],
        'MAE_Aligned_kms': [mae_aligned_insitu, mae_aligned_wsa],
        'Corr_Aligned': [corr_aligned_insitu, corr_aligned_wsa],
        'RMSE_NoAlign_kms': [rmse_noalign_insitu, rmse_noalign_wsa],
        'MAE_NoAlign_kms': [mae_noalign_insitu, mae_noalign_wsa],
        'Corr_NoAlign': [corr_noalign_insitu, corr_noalign_wsa]
    })
    
    dtw_summary_demo.to_csv(os.path.join(figdir, f'dtw_summary_{plot_lead_time}day_2017H1.csv'), 
                            index=False)
    print(f"\nSaved DTW summary as dtw_summary_{plot_lead_time}day_2017H1.csv")
    
    # Save as LaTeX table
    latex_output = dtw_summary_demo.to_latex(index=False, escape=False, 
                                              float_format=lambda x: f'{x:.3f}' if abs(x) < 100 else f'{x:.1f}')
    # Replace booktabs commands with basic LaTeX
    latex_output = latex_output.replace('\\toprule', '\\hline')
    latex_output = latex_output.replace('\\midrule', '\\hline')
    latex_output = latex_output.replace('\\bottomrule', '\\hline')
    with open(os.path.join(figdir, f'dtw_summary_{plot_lead_time}day_2017H1.tex'), 'w') as f:
        f.write(latex_output)
    print(f"Saved DTW summary as dtw_summary_{plot_lead_time}day_2017H1.tex")
    
    # Interpretation
    print("\n" + "-"*60)
    print(f"DTW Interpretation ({dtw_demo_start.strftime('%b-%Y')} to {dtw_demo_end.strftime('%b-%Y')}):")
    print("-"*60)
    print("Lower DTW distance indicates better alignment between forecast and observations.")
    print("DTW accounts for temporal shifts, allowing for timing errors in the forecast.")
    print("\nThe alignment path plot shows how DTW matches points between the two series.")
    print("Deviations from the diagonal indicate timing offsets where the forecast is")
    print("early (below diagonal) or late (above diagonal) relative to observations.")
    
    # Compute DTW for full time period
    print("\n" + "="*60)
    print("Computing DTW for full dataset (2008-2022)...")
    print("="*60)
    
    # Get full overlapping time range and normalize to hour boundaries
    common_start = max(omni_noicmes['datetime'].min(), 
                      combined_for_list[plot_lead_time-1]['time'].min(),
                      combined_for_wsa_list[plot_lead_time-1]['time'].min())
    common_end = min(omni_noicmes['datetime'].max(), 
                    combined_for_list[plot_lead_time-1]['time'].max(),
                    combined_for_wsa_list[plot_lead_time-1]['time'].max())
    
    # Round to nearest hour to ensure alignment with hourly grid
    common_start = common_start.ceil('h')
    common_end = common_end.floor('h')
    
    # Create common hourly time grid for full period
    common_times_full = pd.date_range(start=common_start, end=common_end, freq='h', inclusive='left')
    
    # Merge all datasets onto hourly grid
    obs_full_df = pd.DataFrame({'datetime': common_times_full})
    obs_full_df = obs_full_df.merge(omni_noicmes[['datetime', 'V']], on='datetime', how='left')
    
    insitu_full_df = pd.DataFrame({'datetime': common_times_full})
    insitu_full_df = insitu_full_df.merge(insitu_df_dtw.rename(columns={'time': 'datetime'})[['datetime', 'V']], 
                                          on='datetime', how='left')
    
    wsa_full_df = pd.DataFrame({'datetime': common_times_full})
    wsa_full_df = wsa_full_df.merge(wsa_df_dtw.rename(columns={'time': 'datetime'})[['datetime', 'V']], 
                                     on='datetime', how='left')
    
    # Linearly interpolate through any data gaps (NaNs) before filtering
    print(f"\nInterpolating data gaps in full time series:")
    print(f"  Observations: {obs_full_df['V'].isna().sum()} NaN values before interpolation")
    print(f"  InSitu-HUXt:  {insitu_full_df['V'].isna().sum()} NaN values before interpolation")
    print(f"  WSA-HUXt:     {wsa_full_df['V'].isna().sum()} NaN values before interpolation")
    
    obs_full_df['V'] = obs_full_df['V'].interpolate(method='linear', limit_direction='both')
    insitu_full_df['V'] = insitu_full_df['V'].interpolate(method='linear', limit_direction='both')
    wsa_full_df['V'] = wsa_full_df['V'].interpolate(method='linear', limit_direction='both')
    
    print(f"  Observations: {obs_full_df['V'].isna().sum()} NaN values after interpolation")
    print(f"  InSitu-HUXt:  {insitu_full_df['V'].isna().sum()} NaN values after interpolation")
    print(f"  WSA-HUXt:     {wsa_full_df['V'].isna().sum()} NaN values after interpolation")
    
    # Forward/backward fill any remaining NaNs at edges
    obs_full_df['V'] = obs_full_df['V'].ffill().bfill()
    insitu_full_df['V'] = insitu_full_df['V'].ffill().bfill()
    wsa_full_df['V'] = wsa_full_df['V'].ffill().bfill()
    
    # Resample to 8-hourly resolution
    obs_full_df = obs_full_df.set_index('datetime').resample('8h').mean().reset_index()
    insitu_full_df = insitu_full_df.set_index('datetime').resample('8h').mean().reset_index()
    wsa_full_df = wsa_full_df.set_index('datetime').resample('8h').mean().reset_index()
    
    # Extract 8-hourly arrays for DTW
    obs_dtw_full = obs_full_df['V'].values
    insitu_dtw_full = insitu_full_df['V'].values
    wsa_dtw_full = wsa_full_df['V'].values
    
    print(f"\nFull time series lengths (2008-2022, 8h resolution):")
    print(f"  Observations: {len(obs_dtw_full)}")
    print(f"  InSitu-HUXt:  {len(insitu_dtw_full)}")
    print(f"  WSA-HUXt:     {len(wsa_dtw_full)}")
    print(f"  Observations: {np.isnan(obs_dtw_full).sum()} NaN values")
    print(f"  InSitu-HUXt:  {np.isnan(insitu_dtw_full).sum()} NaN values")
    print(f"  WSA-HUXt:     {np.isnan(wsa_dtw_full).sum()} NaN values")
    
    # Check if all values are NaN (indicating a data merge problem)
    if np.all(np.isnan(obs_dtw_full)):
        print("\nWARNING: All observation values are NaN in the full dataset!")
        print("This suggests the merge operation found no matching timestamps.")
        print(f"Common time range: {common_start} to {common_end}")
        print(f"OMNI time range: {omni_noicmes['datetime'].min()} to {omni_noicmes['datetime'].max()}")
        print(f"InSitu-HUXt time range: {combined_for_list[plot_lead_time-1]['time'].min()} to {combined_for_list[plot_lead_time-1]['time'].max()}")
    
    # Normalize series for fair comparison (should have no NaNs at this point)
    obs_norm_full = ((obs_dtw_full - np.mean(obs_dtw_full)) / np.std(obs_dtw_full)).ravel()
    insitu_norm_full = ((insitu_dtw_full - np.mean(insitu_dtw_full)) / np.std(insitu_dtw_full)).ravel()
    wsa_norm_full = ((wsa_dtw_full - np.mean(wsa_dtw_full)) / np.std(wsa_dtw_full)).ravel()
    
    # Compute DTW on full dataset
    if has_dtaidistance:
        print(f"\nUsing dtaidistance with Sakoe-Chiba band constraint (window={dtw_window_points} points)...")
        print("Computing DTW for InSitu-HUXt (full dataset)...")
        dtw_dist_insitu_full = dtw.distance(obs_norm_full, insitu_norm_full, window=dtw_window_points)
        print("Computing DTW path for InSitu-HUXt (full dataset)...")
        dtw_path_insitu_full = dtw.warping_path(obs_norm_full, insitu_norm_full, window=dtw_window_points)
        
        print("Computing DTW for WSA-HUXt (full dataset)...")
        dtw_dist_wsa_full = dtw.distance(obs_norm_full, wsa_norm_full, window=dtw_window_points)
        print("Computing DTW path for WSA-HUXt (full dataset)...")
        dtw_path_wsa_full = dtw.warping_path(obs_norm_full, wsa_norm_full, window=dtw_window_points)
    elif has_fastdtw:
        print(f"\nUsing FastDTW algorithm (radius={dtw_window_points} points)...")
        print("Computing DTW for InSitu-HUXt (full dataset)...")
        dtw_dist_insitu_full, dtw_path_insitu_full = fastdtw(obs_norm_full, insitu_norm_full, radius=dtw_window_points)
        print("Computing DTW for WSA-HUXt (full dataset)...")
        dtw_dist_wsa_full, dtw_path_wsa_full = fastdtw(obs_norm_full, wsa_norm_full, radius=dtw_window_points)
    
    # Normalize by sequence length
    dtw_dist_insitu_norm_full = dtw_dist_insitu_full / len(obs_norm_full)
    dtw_dist_wsa_norm_full = dtw_dist_wsa_full / len(obs_norm_full)
    
    # Compute average time offset from warping path
    dtw_path_insitu_full_arr = np.array(dtw_path_insitu_full)
    dtw_path_wsa_full_arr = np.array(dtw_path_wsa_full)
    
    # Check that both paths have the same length
    print(f"\nDTW Path Lengths:")
    print(f"  InSitu-HUXt path: {len(dtw_path_insitu_full_arr)} warps")
    print(f"  WSA-HUXt path:    {len(dtw_path_wsa_full_arr)} warps")
    if len(dtw_path_insitu_full_arr) != len(dtw_path_wsa_full_arr):
        print(f"  WARNING: Path lengths differ by {abs(len(dtw_path_insitu_full_arr) - len(dtw_path_wsa_full_arr))} warps!")
    else:
        print(f"  ✓ Path lengths match")
    
    # Calculate time offsets: (forecast_idx - obs_idx) * 8 hours
    # Positive = forecast late, Negative = forecast early
    time_offset_insitu_full = (dtw_path_insitu_full_arr[:, 1] - dtw_path_insitu_full_arr[:, 0]) * 8.0
    time_offset_wsa_full = (dtw_path_wsa_full_arr[:, 1] - dtw_path_wsa_full_arr[:, 0]) * 8.0
    
    avg_offset_insitu_full = np.mean(time_offset_insitu_full)
    avg_offset_wsa_full = np.mean(time_offset_wsa_full)
    max_offset_insitu_full = np.max(np.abs(time_offset_insitu_full))
    max_offset_wsa_full = np.max(np.abs(time_offset_wsa_full))
    
    print(f"\nDTW Distance (normalized, full 2008-2022):")
    print(f"  InSitu-HUXt vs Obs: {dtw_dist_insitu_norm_full:.4f}")
    print(f"  WSA-HUXt vs Obs:    {dtw_dist_wsa_norm_full:.4f}")
    
    print(f"\nAverage Time Offset (full 2008-2022):")
    print(f"  InSitu-HUXt: {avg_offset_insitu_full:.1f} hours (max: {max_offset_insitu_full:.0f} hours)")
    print(f"  WSA-HUXt:    {avg_offset_wsa_full:.1f} hours (max: {max_offset_wsa_full:.0f} hours)")
    
    # Extract aligned speed values using the warping path
    obs_aligned_insitu_full = obs_dtw_full[dtw_path_insitu_full_arr[:, 0]]
    insitu_aligned_full = insitu_dtw_full[dtw_path_insitu_full_arr[:, 1]]
    obs_aligned_wsa_full = obs_dtw_full[dtw_path_wsa_full_arr[:, 0]]
    wsa_aligned_full = wsa_dtw_full[dtw_path_wsa_full_arr[:, 1]]
    
    # Compute velocity differences for histograms (do this before other metrics)
    # Error = model - obs (positive means model overestimates)
    dv_insitu_aligned = insitu_aligned_full - obs_aligned_insitu_full
    dv_wsa_aligned = wsa_aligned_full - obs_aligned_wsa_full
    
    # Compute metrics on aligned data
    rmse_aligned_insitu_full = np.sqrt(np.mean((insitu_aligned_full - obs_aligned_insitu_full)**2))
    mae_aligned_insitu_full = np.mean(np.abs(insitu_aligned_full - obs_aligned_insitu_full))
    corr_aligned_insitu_full = np.corrcoef(obs_aligned_insitu_full, insitu_aligned_full)[0, 1]
    
    rmse_aligned_wsa_full = np.sqrt(np.mean((wsa_aligned_full - obs_aligned_wsa_full)**2))
    mae_aligned_wsa_full = np.mean(np.abs(wsa_aligned_full - obs_aligned_wsa_full))
    corr_aligned_wsa_full = np.corrcoef(obs_aligned_wsa_full, wsa_aligned_full)[0, 1]
    
    # Compare to non-aligned metrics
    rmse_noalign_insitu_full = np.sqrt(np.mean((insitu_dtw_full - obs_dtw_full)**2))
    mae_noalign_insitu_full = np.mean(np.abs(insitu_dtw_full - obs_dtw_full))
    corr_noalign_insitu_full = np.corrcoef(obs_dtw_full, insitu_dtw_full)[0, 1]
    
    rmse_noalign_wsa_full = np.sqrt(np.mean((wsa_dtw_full - obs_dtw_full)**2))
    mae_noalign_wsa_full = np.mean(np.abs(wsa_dtw_full - obs_dtw_full))
    corr_noalign_wsa_full = np.corrcoef(obs_dtw_full, wsa_dtw_full)[0, 1]
    
    print(f"\nSpeed Errors After Optimal DTW Alignment (full 2008-2022):")
    print(f"  InSitu-HUXt: RMSE={rmse_aligned_insitu_full:.1f} km/s, MAE={mae_aligned_insitu_full:.1f} km/s, r={corr_aligned_insitu_full:.3f}")
    print(f"  WSA-HUXt:    RMSE={rmse_aligned_wsa_full:.1f} km/s, MAE={mae_aligned_wsa_full:.1f} km/s, r={corr_aligned_wsa_full:.3f}")
    
    print(f"\nSpeed Errors Without Alignment (point-by-point, full 2008-2022):")
    print(f"  InSitu-HUXt: RMSE={rmse_noalign_insitu_full:.1f} km/s, MAE={mae_noalign_insitu_full:.1f} km/s, r={corr_noalign_insitu_full:.3f}")
    print(f"  WSA-HUXt:    RMSE={rmse_noalign_wsa_full:.1f} km/s, MAE={mae_noalign_wsa_full:.1f} km/s, r={corr_noalign_wsa_full:.3f}")
    
    print(f"\nImprovement from DTW Alignment (full 2008-2022):")
    print(f"  InSitu-HUXt: RMSE reduced by {(1-rmse_aligned_insitu_full/rmse_noalign_insitu_full)*100:.1f}%, MAE by {(1-mae_aligned_insitu_full/mae_noalign_insitu_full)*100:.1f}%")
    print(f"  WSA-HUXt:    RMSE reduced by {(1-rmse_aligned_wsa_full/rmse_noalign_wsa_full)*100:.1f}%, MAE by {(1-mae_aligned_wsa_full/mae_noalign_wsa_full)*100:.1f}%")
    
    # Save full results
    dtw_summary_full = pd.DataFrame({
        'Model': ['InSitu-HUXt', 'WSA-HUXt'],
        'DTW_Distance_2017H1': [dtw_dist_insitu_norm_demo, dtw_dist_wsa_norm_demo],
        'DTW_Distance_Full': [dtw_dist_insitu_norm_full, dtw_dist_wsa_norm_full],
        'Avg_Time_Offset_2017H1_hours': [avg_offset_insitu, avg_offset_wsa],
        'Avg_Time_Offset_Full_hours': [avg_offset_insitu_full, avg_offset_wsa_full],
        'Max_Time_Offset_2017H1_hours': [max_offset_insitu, max_offset_wsa],
        'Max_Time_Offset_Full_hours': [max_offset_insitu_full, max_offset_wsa_full],
        'RMSE_Aligned_2017H1_kms': [rmse_aligned_insitu, rmse_aligned_wsa],
        'RMSE_Aligned_Full_kms': [rmse_aligned_insitu_full, rmse_aligned_wsa_full],
        'MAE_Aligned_2017H1_kms': [mae_aligned_insitu, mae_aligned_wsa],
        'MAE_Aligned_Full_kms': [mae_aligned_insitu_full, mae_aligned_wsa_full],
        'Corr_Aligned_2017H1': [corr_aligned_insitu, corr_aligned_wsa],
        'Corr_Aligned_Full': [corr_aligned_insitu_full, corr_aligned_wsa_full]
    })
    
    dtw_summary_full.to_csv(os.path.join(figdir, f'dtw_summary_{plot_lead_time}day.csv'), index=False)
    print(f"\nSaved full DTW summary as dtw_summary_{plot_lead_time}day.csv")
    
    # Save as LaTeX table
    latex_output = dtw_summary_full.to_latex(index=False, escape=False,
                                              float_format=lambda x: f'{x:.3f}' if abs(x) < 100 else f'{x:.1f}')
    # Replace booktabs commands with basic LaTeX
    latex_output = latex_output.replace('\\toprule', '\\hline')
    latex_output = latex_output.replace('\\midrule', '\\hline')
    latex_output = latex_output.replace('\\bottomrule', '\\hline')
    with open(os.path.join(figdir, f'dtw_summary_{plot_lead_time}day.tex'), 'w') as f:
        f.write(latex_output)
    print(f"Saved full DTW summary as dtw_summary_{plot_lead_time}day.tex")
    
    print("\n" + "-"*60)
    print("Full Dataset DTW Summary:")
    print("-"*60)
    print(dtw_summary_full.to_string(index=False))
    
    # Create histograms of warping times and velocity differences
    print("\n" + "="*60)
    print("Creating histograms of warping times and velocity differences...")
    print("="*60)
    
    # Check if we have valid data for histograms
    has_valid_data = (~np.isnan(dv_insitu_aligned)).any() and (~np.isnan(dv_wsa_aligned)).any()
    
    if not has_valid_data:
        print("WARNING: No valid data for full dataset histograms (all NaN). Skipping histogram creation.")
        print(f"  InSitu aligned data: {(~np.isnan(dv_insitu_aligned)).sum()} valid points out of {len(dv_insitu_aligned)}")
        print(f"  WSA aligned data: {(~np.isnan(dv_wsa_aligned)).sum()} valid points out of {len(dv_wsa_aligned)}")
    else:
        fig_hist, axs_hist = plt.subplots(2, 2, figsize=(16, 10))
        
        # Create bins for time offset histograms (multiples of 8 hours)
        # Include both positive (late) and negative (early) offsets
        max_offset = max(np.max(np.abs(time_offset_insitu_full)), np.max(np.abs(time_offset_wsa_full)))
        time_bins = np.arange(-max_offset - 8, max_offset + 16, 8)  # Bins at ..., -16, -8, 0, 8, 16, ... hours
        
        # Top left: Histogram of time offsets for InSitu-HUXt
        axs_hist[0, 0].hist(time_offset_insitu_full, bins=time_bins, color=color_insitu, alpha=0.7, edgecolor='black')
        axs_hist[0, 0].axvline(avg_offset_insitu_full, color='red', linestyle='--', linewidth=2.5, 
                              label=f'Mean = {avg_offset_insitu_full:.1f}h')
        axs_hist[0, 0].axvline(np.median(time_offset_insitu_full), color='darkred', linestyle=':', linewidth=2.5,
                              label=f'Median = {np.median(time_offset_insitu_full):.1f}h')
        axs_hist[0, 0].set_xlabel('Time Offset [hours]', fontsize=16)
        axs_hist[0, 0].set_ylabel('Frequency', fontsize=16)
        axs_hist[0, 0].set_title('(a) InSitu-HUXt: Time Offsets from DTW Alignment', fontsize=14, fontweight='bold')
        axs_hist[0, 0].legend(fontsize=14)
        axs_hist[0, 0].grid(True, alpha=0.3)
        
        # Top right: Histogram of time offsets for WSA-HUXt
        axs_hist[0, 1].hist(time_offset_wsa_full, bins=time_bins, color=color_wsa, alpha=0.7, edgecolor='black')
        axs_hist[0, 1].axvline(avg_offset_wsa_full, color='red', linestyle='--', linewidth=2.5,
                              label=f'Mean = {avg_offset_wsa_full:.1f}h')
        axs_hist[0, 1].axvline(np.median(time_offset_wsa_full), color='darkred', linestyle=':', linewidth=2.5,
                              label=f'Median = {np.median(time_offset_wsa_full):.1f}h')
        axs_hist[0, 1].set_xlabel('Time Offset [hours]', fontsize=16)
        axs_hist[0, 1].set_ylabel('Frequency', fontsize=16)
        axs_hist[0, 1].set_title('(b) WSA-HUXt: Time Offsets from DTW Alignment', fontsize=14, fontweight='bold')
        axs_hist[0, 1].legend(fontsize=14)
        axs_hist[0, 1].grid(True, alpha=0.3)
        
        # Set same x-limits for top two plots (time offsets)
        time_xlim = [-max_offset - 8, max_offset + 8]
        axs_hist[0, 0].set_xlim(time_xlim)
        axs_hist[0, 1].set_xlim(time_xlim)
        axs_hist[0, 0].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        axs_hist[0, 1].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        # Bottom left: Histogram of velocity differences for InSitu-HUXt (aligned)
        axs_hist[1, 0].hist(dv_insitu_aligned, bins=50, color=color_insitu, alpha=0.7, edgecolor='black')
        axs_hist[1, 0].axvline(0, color='black', linestyle='-', linewidth=2.0)
        axs_hist[1, 0].axvline(np.mean(dv_insitu_aligned), color='red', linestyle='--', linewidth=2.5,
                              label=f'Mean = {np.mean(dv_insitu_aligned):.1f} km/s')
        axs_hist[1, 0].axvline(np.median(dv_insitu_aligned), color='darkred', linestyle=':', linewidth=2.5,
                              label=f'Median = {np.median(dv_insitu_aligned):.1f} km/s')
        axs_hist[1, 0].set_xlabel('Speed Error (Forecast - Obs) [km/s]', fontsize=16)
        axs_hist[1, 0].set_ylabel('Frequency', fontsize=16)
        axs_hist[1, 0].set_title(f'(c) InSitu-HUXt: Speed Errors After DTW Alignment\\n(RMSE={rmse_aligned_insitu_full:.1f} km/s)', 
                                fontsize=14, fontweight='bold')
        axs_hist[1, 0].legend(fontsize=14)
        axs_hist[1, 0].grid(True, alpha=0.3)
        
        # Bottom right: Histogram of velocity differences for WSA-HUXt (aligned)
        axs_hist[1, 1].hist(dv_wsa_aligned, bins=50, color=color_wsa, alpha=0.7, edgecolor='black')
        axs_hist[1, 1].axvline(0, color='black', linestyle='-', linewidth=2.0)
        axs_hist[1, 1].axvline(np.mean(dv_wsa_aligned), color='red', linestyle='--', linewidth=2.5,
                              label=f'Mean = {np.mean(dv_wsa_aligned):.1f} km/s')
        axs_hist[1, 1].axvline(np.median(dv_wsa_aligned), color='darkred', linestyle=':', linewidth=2.5,
                              label=f'Median = {np.median(dv_wsa_aligned):.1f} km/s')
        axs_hist[1, 1].set_xlabel('Speed Error (Forecast - Obs) [km/s]', fontsize=16)
        axs_hist[1, 1].set_ylabel('Frequency', fontsize=16)
        axs_hist[1, 1].set_title(f'(d) WSA-HUXt: Speed Errors After DTW Alignment\\n(RMSE={rmse_aligned_wsa_full:.1f} km/s)', 
                                fontsize=14, fontweight='bold')
        axs_hist[1, 1].legend(fontsize=14)
        axs_hist[1, 1].grid(True, alpha=0.3)
        
        # Set same x-limits for bottom two plots (velocity differences)
        # Use 99th percentile to exclude extreme outliers and make range more readable
        dv_combined = np.concatenate([dv_insitu_aligned, dv_wsa_aligned])
        dv_min = np.percentile(dv_combined, 0.5)
        dv_max = np.percentile(dv_combined, 99.5)
        dv_xlim = [dv_min, dv_max]
        axs_hist[1, 0].set_xlim(dv_xlim)
        axs_hist[1, 1].set_xlim(dv_xlim)
        
        fig_hist.tight_layout()
        fig_hist.savefig(os.path.join(figdir, f'dtw_histograms_{plot_lead_time}day.pdf'), 
                        dpi=150, bbox_inches='tight')
        print(f"Saved DTW histograms as dtw_histograms_{plot_lead_time}day.pdf")
    
    print(f"\\nTime Offset Statistics (2008-2022):")
    print(f"  InSitu-HUXt: mean={avg_offset_insitu_full:.1f}h, median={np.nanmedian(time_offset_insitu_full):.1f}h, std={np.nanstd(time_offset_insitu_full):.1f}h")
    print(f"  WSA-HUXt:    mean={avg_offset_wsa_full:.1f}h, median={np.nanmedian(time_offset_wsa_full):.1f}h, std={np.nanstd(time_offset_wsa_full):.1f}h")
    
    print(f"\\nVelocity Difference Statistics After DTW Alignment (2008-2022):")
    print(f"  InSitu-HUXt: mean={np.nanmean(dv_insitu_aligned):.1f} km/s, median={np.nanmedian(dv_insitu_aligned):.1f} km/s, std={np.nanstd(dv_insitu_aligned):.1f} km/s")
    print(f"  WSA-HUXt:    mean={np.nanmean(dv_wsa_aligned):.1f} km/s, median={np.nanmedian(dv_wsa_aligned):.1f} km/s, std={np.nanstd(dv_wsa_aligned):.1f} km/s")




# Show plots and allow closing with Enter key
if wait_for_enter_before_exit and os.isatty(0):
    print("\nPress Enter to close all plots...")
    input()
plt.close('all')

# %%
