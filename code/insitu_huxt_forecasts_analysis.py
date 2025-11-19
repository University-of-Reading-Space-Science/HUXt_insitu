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

start_time = datetime.datetime(2007,1,1)
stop_time = datetime.datetime(2008,1,1)#datetime.datetime(2023,1,1)

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


# <codecell> Taylor diagrams

import skill_metrics as sm

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

# Calculate statistics for Taylor diagram
stats_insitu = sm.taylor_statistics(pred_insitu_clean, obs_clean)
stats_wsa = sm.taylor_statistics(pred_wsa_clean, obs_clean)

# Create Taylor diagram for overall period with both models
fig = plt.figure(figsize=(8, 8))

# Combine both statistics into arrays
# Note: [0] is reference statistics (obs), [1] is prediction statistics (model)
# We must include the reference stats as the first element for taylor_diagram
STDs = np.array([float(stats_insitu['sdev'][0]), float(stats_insitu['sdev'][1]), float(stats_wsa['sdev'][1])])
RMSs = np.array([float(stats_insitu['crmsd'][0]), float(stats_insitu['crmsd'][1]), float(stats_wsa['crmsd'][1])])
CORs = np.array([float(stats_insitu['ccoef'][0]), float(stats_insitu['ccoef'][1]), float(stats_wsa['ccoef'][1])])

sm.taylor_diagram(STDs, RMSs, CORs,
                  markerLabel=['Obs', 'InSitu-HUXt', 'WSA-HUXt'], 
                  markerSize=10,
                  tickRMS=[50, 100, 150, 200], 
                  styleOBS='-',
                  colOBS='k', markerobs='o', titleOBS='Obs')

plt.title(f'Taylor Diagram: {plot_lead_time}-day lead time\n{start_time.year} to {stop_time.year}', 
          fontsize=14, pad=20)
plt.tight_layout()


# <codecell> Taylor diagram by year

# Calculate stats for each year
years = range(start_time.year, stop_time.year)
stats_by_year_insitu = []
stats_by_year_wsa = []
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
        obs_year_clean = obs_year[mask_valid]
        pred_insitu_year_clean = pred_insitu_year[mask_valid]
        pred_wsa_year_clean = pred_wsa_year[mask_valid]
        
        stats_by_year_insitu.append(sm.taylor_statistics(pred_insitu_year_clean, obs_year_clean))
        stats_by_year_wsa.append(sm.taylor_statistics(pred_wsa_year_clean, obs_year_clean))
        year_labels.append(str(year))

# Create Taylor diagram by year
fig = plt.figure(figsize=(10, 10))

# Prepare arrays for all years - both InSitu and WSA
# Normalize statistics by the observation standard deviation of each year
# This allows plotting multiple years with different variances on the same diagram
STDs_norm = [1.0]  # Reference STD is always 1.0 after normalization
RMSs_norm = [0.0]  # Reference RMSE is 0.0
CORs_norm = [1.0]  # Reference Correlation is 1.0
labels_all = ['Obs']

for i, year in enumerate(year_labels):
    obs_std = float(stats_by_year_insitu[i]['sdev'][0])
    
    # Add InSitu-HUXt point
    STDs_norm.append(float(stats_by_year_insitu[i]['sdev'][1]) / obs_std)
    RMSs_norm.append(float(stats_by_year_insitu[i]['crmsd'][1]) / obs_std)
    CORs_norm.append(float(stats_by_year_insitu[i]['ccoef'][1]))
    labels_all.append(f'{year} InSitu')
    
    # Add WSA-HUXt point
    STDs_norm.append(float(stats_by_year_wsa[i]['sdev'][1]) / obs_std)
    RMSs_norm.append(float(stats_by_year_wsa[i]['crmsd'][1]) / obs_std)
    CORs_norm.append(float(stats_by_year_wsa[i]['ccoef'][1]))
    labels_all.append(f'{year} WSA')

sm.taylor_diagram(np.array(STDs_norm), np.array(RMSs_norm), np.array(CORs_norm),
                  markerLabel=labels_all,
                  markerSize=8,
                  tickRMS=[0.5, 1.0, 1.5],
                  styleOBS='-',
                  colOBS='k', markerobs='o', titleOBS='Obs')

plt.title(f'Normalized Taylor Diagram by Year: {plot_lead_time}-day lead time', 
          fontsize=14, pad=20)
plt.tight_layout()

plt.show()
# %%
