# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:49:37 2025

@author: mathe
"""
# <codecell> Imports
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



#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset
import joblib
import onnxruntime as ort


#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA


import huxt_insitu as His


ftime = datetime.datetime(2022,12,1)
#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 240*u.solRad #outer boundary for HUXt runs


# <codecell> Get and process OMNI data
#grab the omni data around this interval
dl_starttime = ftime - datetime.timedelta(days=27)
dl_endtime = ftime + datetime.timedelta(days=27*2)
omni = Hin.get_omni(dl_starttime, dl_endtime)

#remove ICMEs
omni_noicmes = His.removeICMEs(omni, 
                icme_list = 'CaneRichardson',
                pre_icme_buffer = 0.2, #days
                post_icme_buffer = 1, #days
                interp_gaps = True )

# <codecell> generate the forecast for this time
model = His.omniHUXt_forecast(ftime, simtime = (27.27+27)*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes, buffertime = 27*u.day)
model.solve([])  
#HA.plot_earth_timeseries(model)
ts = HA.get_observer_timeseries(model, observer='Earth')



#add some CMEs
# model_cmes = His.omniHUXt_forecast(ftime, simtime = (27.27+27)*u.day, 
#                             rmin = rmin, rmax = rmax, 
#                             dt_scale = dt_scale,
#                             omni_input = omni_noicmes, buffertime = 27*u.day)

# v_boundary = np.ones(128) * 400 * (u.km/u.s)
# cr, cr_lon_init = Hin.datetime2huxtinputs(ftime - datetime.timedelta(days = 5))
# Elat = Hin.get_earth_lat(ftime)
# model = H.HUXt(v_boundary=v_boundary, simtime=27*u.day, dt_scale=4,
#                     r_min = 21.5*u.solRad, cr_num = cr, cr_lon_init = cr_lon_init)
#get the DONKI coneCMEs
# cme_list = Hin.get_DONKI_cme_list(model_cmes, dl_starttime, ftime)
# #change each CME to fixed duration
# for cme in cme_list:
#     print(cme.t_launch)
#     cme.cme_fixed_duration = True
#     cme.fixed_duration = 8*60*60*u.s

# model_cmes.solve(cme_list)

#HA.plot_earth_timeseries(model_cmes)
# ts_cmes = HA.get_observer_timeseries(model_cmes, observer='Earth')


# <codecell> Now read in the pre-computed forecast and plot
ishuxt_dir = os.path.join(os.getenv('DBOX'), 'InSitu_HUXt')
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
    
    ts_file = pd.read_csv(filepath, index_col=0, parse_dates=["time"] )



# %%

cwd = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(os.path.dirname(cwd), 'data')
min_lt = 1
max_lt = 27


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

# <codecell> Plot it


fig, axes = plt.subplots(1, 1, figsize=(14, 6))

ax = axes
ax.plot(ts['time'], ts['vsw'],'b', label='OMNI-HUXt')
#ax.plot(ts_cmes['time'], ts_cmes['vsw'],'r--', label='OMNI-HUXt (CMEs)')
ax.plot(omni['datetime'], omni['V'],'k', label='OMNI')

ax.set_ylabel(r'$V_{SW}$ [km s$^{-1}$]' )
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
 
yy = ax.get_ylim()
ax.set_ylim(yy)
ax.plot([ftime, ftime], yy, 'k--')

ax.plot(ts_file['time'], ts_file['vsw'],'r--', label='From file')
xx=ax.get_xlim()
ax.plot(combined_for_list[6]['time'], combined_for_list[0]['vsw'],'g--', label='1-day leadtime')
ax.legend()
ax.set_xlim(xx)
plt.show()
