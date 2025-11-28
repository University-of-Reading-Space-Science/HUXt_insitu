# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 08:51:29 2025

Run InSitu-HUXt forecasts for a date range

Analyse and compare forecasts with WSA-HUXt
"""

# <codecell> import libraries

import datetime
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
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

start_time = datetime.datetime(2020,3,1)
stop_time = datetime.datetime(2024,8,1)

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


plot_lead_time = 3

#directory structure
cwd = os.path.dirname(os.path.realpath(__file__))
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
data_dir = os.path.join(os.path.dirname(cwd), 'data')
icmelist_path = os.path.join(data_dir, 'Richardson_Cane_Porcessed_ICME_list.csv')


output_dir = '~/InSitu_HUXt/'




# <codecell> get the OMNI data for the whole itnerval - both forecasts and validation

dl_starttime = start_time - datetime.timedelta(days=28)
dl_endtime = stop_time + datetime.timedelta(days=28)

omni = Hin.get_omni(dl_starttime, dl_endtime)



omni_noicmes = His.removeICMEs(omni, 
                icme_list = 'CaneRichardson',
                pre_icme_buffer = 0.2, #days
                post_icme_buffer = 1, #days
                interp_gaps = True )



    
# <codecell> a single forecast



ftime = datetime.datetime(2022,5,1)
model = His.omniHUXt_forecast(ftime, simtime = 27.27*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes, buffertime = 0*u.day)
model.solve([])  
HA.plot_earth_timeseries(model)
ts = HA.get_observer_timeseries(model, observer='Earth')



# <codecell> set up the forecasts and save the data
    
dates = pd.date_range(start=start_time, end=stop_time, freq="D")    
for ftime in dates:
    print('Running InSitu-HUXt for ' + ftime.strftime("%Y-%m-%d"))
    model = His.omniHUXt_forecast(ftime, simtime = 27.27*u.day, 
                               rmin = rmin, rmax = rmax, 
                               dt_scale = dt_scale,
                               dt_scale = 4,
                               omni_input = omni_noicmes, buffertime = 0*u.day)
    model.solve([])  
    ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning= True)
    
    #save the data
    yr_str = str(ftime.year)
    mn_str = f"{ftime.month:02}"
    dy_str = f"{ftime.day:02}"
    dir_path = os.path.join(output_dir, yr_str)
    #create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    outname = 'InSitu_Earth_' + yr_str +mn_str + dy_str + '.dat'
    ts.to_csv(os.path.join(dir_path,outname))
    
    
    


