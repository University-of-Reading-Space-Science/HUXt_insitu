# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:49:37 2025

@author: mathe
"""

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


ftime = datetime.datetime(2024,5,9)
#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs


# <codecell> Get and process OMNI data
#grab the omni data around this interval
dl_starttime = ftime - datetime.timedelta(days=15)
dl_endtime = ftime + datetime.timedelta(days=15)
omni = Hin.get_omni(dl_starttime, dl_endtime)

#remove ICMEs
omni_noicmes = His.removeICMEs(omni, 
                icme_list = 'CaneRichardson',
                pre_icme_buffer = 0.2, #days
                post_icme_buffer = 1, #days
                interp_gaps = True )

# <codecell> generate the forecast for this time
model = His.omniHUXt_forecast(ftime, simtime = (27.27+15)*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes, buffertime = 15*u.day)
model.solve([])  
#HA.plot_earth_timeseries(model)
ts = HA.get_observer_timeseries(model, observer='Earth')



#add some CMEs
model_cmes = His.omniHUXt_forecast(ftime, simtime = (27.27+15)*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes, buffertime = 15*u.day)

# v_boundary = np.ones(128) * 400 * (u.km/u.s)
# cr, cr_lon_init = Hin.datetime2huxtinputs(ftime - datetime.timedelta(days = 5))
# Elat = Hin.get_earth_lat(ftime)
# model = H.HUXt(v_boundary=v_boundary, simtime=27*u.day, dt_scale=4,
#                     r_min = 21.5*u.solRad, cr_num = cr, cr_lon_init = cr_lon_init)
#get the DONKI coneCMEs
cme_list = Hin.get_DONKI_cme_list(model_cmes, dl_starttime, ftime)
#change each CME to fixed duration
for cme in cme_list:
    print(cme.t_launch)
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s

model_cmes.solve(cme_list)

#HA.plot_earth_timeseries(model_cmes)
ts_cmes = HA.get_observer_timeseries(model_cmes, observer='Earth')



# <codecell> Plot it


fig, axes = plt.subplots(1, 1, figsize=(14, 6))

ax = axes
ax.plot(ts['time'], ts['vsw'],'b', label='OMNI-HUXt')
ax.plot(ts_cmes['time'], ts_cmes['vsw'],'r--', label='OMNI-HUXt (CMEs)')
ax.plot(omni['datetime'], omni['V'],'k', label='OMNI')
ax.legend()
ax.set_ylabel(r'$V_{SW}$ [km s$^{-1}$]' )
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.set_xlabel(r'Date (2024)' )
 
yy = ax.get_ylim()
ax.set_ylim(yy)
ax.plot([ftime, ftime], yy, 'k--')