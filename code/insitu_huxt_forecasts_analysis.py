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
import pandas as pd
import glob
from astropy.time import Time
from sunpy.coordinates import sun


#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA

import insitu
import helio_coords as hcoords
from taylor_diagram import combined_taylor_target_diagram, plot_taylor_target_diagram
import hse  # High-Speed Enhancement detection and analysis

start_time = datetime.datetime(2007,5,1)
stop_time = datetime.datetime(2023,1,1)#datetime.datetime(2023,1,1)

#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs

#lead time of forecasts
min_lt = 1
max_lt = 27


icme_list = 'CaneRichardson'#'DONKI'
pre_icme_buffer = 0.2 #days
post_icme_buffer = 1 #days
#fill values for ICME removal if interpolation isn't possible
vsw_fill = 450 *u.km/u.s
bx_fill = 0


plot_lead_time = 6

#directory structure
cwd = os.path.dirname(os.path.realpath(__file__))
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
icmelist_path = os.path.join(os.path.dirname(cwd), 'data', 'Richardson_Cane_Porcessed_ICME_list.csv')
ishuxt_dir = os.path.join(os.getenv('DBOX'), 'Data', 'InSitu_HUXt')
wsahuxt_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC', 'CCMC_WSA_HUXt')




# <codecell> get the OMNI data for the whole itnerval - both forecasts and validation

dl_starttime = start_time - datetime.timedelta(days=28)
dl_endtime = stop_time + datetime.timedelta(days=28)

omni = Hin.get_omni(dl_starttime, dl_endtime)




# <codecell> remove ICMEs
#create a copy of the OMNI data for ICME removal, before any end padding
omni_noicmes = omni.copy()

#load the DONKI ICME list
if icme_list == 'DONKI':
    icmes = Hin.get_DONKI_ICMEs(dl_starttime, dl_endtime)
elif icme_list == 'CaneRichardson':
    icmes = insitu.ICMElist(icmelist_path)


#remove all ICMEs
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



    

    
    
# <codecell> Compute InSItu-HUXt timeseries for different forecast lead times

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

for i, df in enumerate(combined_for_list):
    pass  # Data loaded



# <codecell>  WSA-HUXt forecasts for the same time period
output_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC', 'CCMC_WSA_HUXt')

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
    for ifor in range(0,len(for_list)):
        start_window = ftime + datetime.timedelta(days = ifor + 1)
        stop_window = ftime + datetime.timedelta(days = ifor + 2.0)
        
        mask = ((ts['time'] >= start_window)  & (ts['time'] <= stop_window))
        for_wsa_list[ifor].append(ts.loc[mask])

    
#combine the dataframes at each forecast lead time.
combined_for_wsa_list = [pd.concat(df_list, ignore_index=True) for df_list in for_wsa_list]

fig, axs = plt.subplots(2, 1, figsize=(14, 8))

ax = axs[1]
ax.plot(combined_for_list[plot_lead_time-1]['time'], 
        combined_for_list[plot_lead_time-1]['vsw'], 'r', label = 'InSitu-HUXt')
ax.plot(omni_noicmes['datetime'], omni_noicmes['V'], 'k', label = 'OMNI')
ax.legend()
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlim((start_time, stop_time))
ax.set_ylim((200, 900))

ax = axs[0]
ax.plot(combined_for_wsa_list[plot_lead_time-1]['time'], 
        combined_for_wsa_list[plot_lead_time-1]['vsw'], 'r', label = 'WSA-HUXt')
ax.plot(omni_noicmes['datetime'], omni_noicmes['V'], 'k', label = 'OMNI')
ax.legend()
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlim((start_time, stop_time))
ax.set_ylim((200, 900))

ax.set_title(str(plot_lead_time) + '-day lead-time forecasts', fontsize=14)


# <codecell> compute MAE at different lead times

for_insitu_omni = []
for_wsa_omni = []
#interpolate each forecast series onto the OMNI time step
for n in range(0, len(for_list)):
    for_insitu_omni.append( np.interp(omni_noicmes['mjd'], 
                                      combined_for_list[n]['mjd'], combined_for_list[n]['vsw'],
                                      left = np.nan, right = np.nan) )
    for_wsa_omni.append( np.interp(omni_noicmes['mjd'], 
                                      combined_for_wsa_list[n]['mjd'], combined_for_wsa_list[n]['vsw'],
                                      left = np.nan, right = np.nan) )
    
#compute MAE at each lead time   
mae_wsa_lt =[]
mae_insitu_lt = []
for n in range(0, len(for_list)):
    mae_insitu_lt.append( np.nanmean(abs(for_insitu_omni[n] - omni_noicmes['V'])))
    mae_wsa_lt.append( np.nanmean(abs(for_wsa_omni[n] - omni_noicmes['V'])))
    
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
axs.plot(np.arange(min_lt, max_lt+1, 1), np.array(mae_wsa_lt), 'r', label = 'WSA-HUXt')
axs.plot(np.arange(min_lt, max_lt+1, 1), np.array(mae_insitu_lt), 'b', label = 'InSitu-HUXt')
axs.set_xlabel('Forecast lead time [days]')
axs.set_ylabel('MAE [km/s]')
axs.set_title(str(start_time.year) + ' to ' + str(stop_time.year), fontsize = 14)
axs.legend()


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

# Calculate unbiased RMSD (centered RMSD)
ubrmsd_insitu = np.sqrt(np.mean((pred_insitu_clean - obs_clean - bias_insitu)**2))
ubrmsd_wsa = np.sqrt(np.mean((pred_wsa_clean - obs_clean - bias_wsa)**2))

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

# Combined diagram for overall period
fig_combined, (ax_t, ax_tg) = combined_taylor_target_diagram(
    obs_clean,
    [pred_insitu_clean, pred_wsa_clean],
    labels=['InSitu-HUXt', 'WSA-HUXt'],
    colors=['#2E86AB', '#A23B72'],
    figsize=(16, 7),
    dpi=150,
    units='[km/s]',
    figname=os.path.join(figdir, f'combined_diagram_{plot_lead_time}day')
)

# Combined diagram for normalized version (across all years)
fig_combined_norm, (ax_t_norm, ax_tg_norm) = combined_taylor_target_diagram(
    obs_all_years,
    [pred_insitu_all_years, pred_wsa_all_years],
    labels=['InSitu-HUXt', 'WSA-HUXt'],
    colors=['#2E86AB', '#A23B72'],
    figsize=(16, 7),
    dpi=150,
    normalize=True,
    figname=os.path.join(figdir, f'combined_diagram_{plot_lead_time}day_normalized')
)


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
insitu_df = insitu_df.set_index('datetime')

# Create WSA-HUXt forecast dataframe for the selected lead time
wsa_times = pd.to_datetime(combined_for_wsa_list[plot_lead_time-1]['time'].values)
wsa_df = pd.DataFrame({
    'datetime': wsa_times.round('h'),  # Round to nearest hour for alignment
    'V': combined_for_wsa_list[plot_lead_time-1]['vsw'].values
})
wsa_df = wsa_df.set_index('datetime')

# Apply 8-hour smoothing (as per hse.py methodology)
obs_smoothed = hse.smooth_8h(obs_df.reset_index(), date_col='datetime', window_hours=8)
insitu_smoothed = hse.smooth_8h(insitu_df.reset_index(), date_col='datetime', window_hours=8)
wsa_smoothed = hse.smooth_8h(wsa_df.reset_index(), date_col='datetime', window_hours=8)

# Detect HSE events in each time series
# Parameters: rise_threshold=100 km/s over 48 hours, min_separation=72 hours
events_obs = hse.detect_hss_events(obs_smoothed, speed_col='V', 
                                    rise_threshold=100, rise_hours=48, 
                                    min_separation_hours=72)
events_insitu = hse.detect_hss_events(insitu_smoothed, speed_col='V',
                                       rise_threshold=100, rise_hours=48,
                                       min_separation_hours=72)
events_wsa = hse.detect_hss_events(wsa_smoothed, speed_col='V',
                                    rise_threshold=100, rise_hours=48,
                                    min_separation_hours=72)

print(f"\nHSE Events Detected:")
print(f"  Observations:  {len(events_obs)} events")
print(f"  InSitu-HUXt:   {len(events_insitu)} events")
print(f"  WSA-HUXt:      {len(events_wsa)} events")

# Match events between observations and forecasts
# Using Owens (2005) method with 48-hour maximum timing difference
matches_insitu, unmatched_obs_insitu, unmatched_insitu = hse.match_events(
    events_obs, events_insitu, delta_t_max=48)
matches_wsa, unmatched_obs_wsa, unmatched_wsa = hse.match_events(
    events_obs, events_wsa, delta_t_max=48)

print(f"\nEvent Matching Results (max Δt = 48 hours):")
print(f"  InSitu-HUXt: {len(matches_insitu)} matched, "
      f"{len(unmatched_obs_insitu)} missed, {len(unmatched_insitu)} false alarms")
print(f"  WSA-HUXt:    {len(matches_wsa)} matched, "
      f"{len(unmatched_obs_wsa)} missed, {len(unmatched_wsa)} false alarms")

# Compute timing errors for matched events (using event start/arrival time)
if len(matches_insitu) > 0:
    timing_errors_insitu = hse.compute_timing_errors(matches_insitu, use='start')
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
    timing_errors_wsa = hse.compute_timing_errors(matches_wsa, use='start')
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
print(f"\nHSE Arrival Timing Errors (hours):")
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

# Timing error histograms
timing_bins = np.arange(-60, 65, 10)

ax = axs_hse[0, 0]
if len(timing_errors_insitu) > 0:
    ax.hist(timing_errors_insitu, bins=timing_bins, alpha=0.7, color='#2E86AB', 
            edgecolor='black', linewidth=1.2, label='InSitu-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Arrival Timing Error [hours]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'InSitu-HUXt HSE Timing Errors\n(MAE = {timing_stats_insitu["mean_absolute_error"]:.1f} h)', 
             fontsize=12)
ax.legend()

ax = axs_hse[0, 1]
if len(timing_errors_wsa) > 0:
    ax.hist(timing_errors_wsa, bins=timing_bins, alpha=0.7, color='#A23B72',
            edgecolor='black', linewidth=1.2, label='WSA-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Arrival Timing Error [hours]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'WSA-HUXt HSE Timing Errors\n(MAE = {timing_stats_wsa["mean_absolute_error"]:.1f} h)', 
             fontsize=12)
ax.legend()

# Peak speed error histograms
speed_bins = np.arange(-300, 350, 50)

ax = axs_hse[1, 0]
if len(speed_errors_insitu) > 0:
    ax.hist(speed_errors_insitu, bins=speed_bins, alpha=0.7, color='#2E86AB',
            edgecolor='black', linewidth=1.2, label='InSitu-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Peak Speed Error [km/s]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'InSitu-HUXt HSE Peak Speed Errors\n(MAE = {speed_stats_insitu["mean_absolute_error"]:.1f} km/s)', 
             fontsize=12)
ax.legend()

ax = axs_hse[1, 1]
if len(speed_errors_wsa) > 0:
    ax.hist(speed_errors_wsa, bins=speed_bins, alpha=0.7, color='#A23B72',
            edgecolor='black', linewidth=1.2, label='WSA-HUXt')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Peak Speed Error [km/s]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'WSA-HUXt HSE Peak Speed Errors\n(MAE = {speed_stats_wsa["mean_absolute_error"]:.1f} km/s)', 
             fontsize=12)
ax.legend()

fig_hse.suptitle(f'{plot_lead_time}-Day Lead Time HSE Error Distributions\n'
                 f'({start_time.year} to {stop_time.year})', fontsize=14, fontweight='bold')
fig_hse.tight_layout()
fig_hse.savefig(os.path.join(figdir, f'hse_errors_{plot_lead_time}day.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved HSE error distributions as hse_errors_{plot_lead_time}day.png")


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

# Save summary table to CSV
summary_table.to_csv(os.path.join(figdir, f'hse_summary_{plot_lead_time}day.csv'), index=False)
print(f"\nSaved HSE summary table as hse_summary_{plot_lead_time}day.csv")

plt.show()
# %%
