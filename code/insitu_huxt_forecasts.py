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
from joblib import Parallel, delayed


#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA


import huxt_insitu as His

start_time = datetime.datetime(2010,3,1)
stop_time = datetime.datetime(2023,1,1)

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
data_dir = os.path.join(os.path.dirname(cwd), 'data')
icmelist_path = os.path.join(data_dir, 'Richardson_Cane_Porcessed_ICME_list.csv')
output_dir = os.getenv('HUXT_INSITU_RAW_DIR',
                       os.path.join(data_dir, 'raw', 'insitu_huxt'))




# <codecell> get the OMNI data for the whole itnerval - both forecasts and validation

dl_starttime = start_time - datetime.timedelta(days=55)
dl_endtime = stop_time + datetime.timedelta(days=55)

omni = Hin.get_omni(dl_starttime, dl_endtime)



omni_noicmes = His.removeICMEs(omni, 
                icme_list = 'CaneRichardson',
                pre_icme_buffer = 0.2, #days
                post_icme_buffer = 1, #days
                interp_gaps = True )



    
# <codecell> a single forecast



ftime = datetime.datetime(2017,4,12)
model = His.omniHUXt_forecast(ftime, simtime = 55*u.day, 
                            rmin = rmin, rmax = rmax, 
                            dt_scale = dt_scale,
                            omni_input = omni_noicmes, buffertime = 28*u.day)
model.solve([])  
HA.plot_earth_timeseries(model)
ts = HA.get_observer_timeseries(model, observer='Earth')



# <codecell> set up the forecasts and save the data

def run_single_forecast(ftime, simtime, rmin, rmax, dt_scale, omni_noicmes, output_dir):
    """
    Run a single InSitu-HUXt forecast and save the output.
    
    Parameters
    ----------
    ftime : datetime.datetime
        Forecast initialization time
    simtime : astropy.units.Quantity
        Simulation time duration
    rmin, rmax : astropy.units.Quantity
        Inner and outer boundary radii
    dt_scale : int
        Time step scaling factor
    omni_noicmes : pandas.DataFrame
        OMNI data with ICMEs removed
    output_dir : str
        Output directory path
    
    Returns
    -------
    str
        Status message for the forecast run
    """
    try:
        print(f'Running InSitu-HUXt for {ftime.strftime("%Y-%m-%d")}')
        model = His.omniHUXt_forecast(ftime, simtime=simtime, 
                                   rmin=rmin, rmax=rmax, 
                                   dt_scale=dt_scale,
                                   omni_input=omni_noicmes, buffertime=0*u.day)
        model.solve([])  
        ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning=True)
        
        # Save the data
        yr_str = str(ftime.year)
        mn_str = f"{ftime.month:02}"
        dy_str = f"{ftime.day:02}"
        dir_path = os.path.join(output_dir, yr_str)
        
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        outname = f'InSitu_Earth_{yr_str}{mn_str}{dy_str}.dat'
        ts.to_csv(os.path.join(dir_path, outname))
        
        return f'✓ Completed: {ftime.strftime("%Y-%m-%d")}'
    except Exception as e:
        return f'✗ Failed: {ftime.strftime("%Y-%m-%d")} - {str(e)}'

    
dates = pd.date_range(start=start_time, end=stop_time, freq="D")

# Run forecasts in parallel
print(f"Running {len(dates)} forecasts in parallel...")
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_single_forecast)(ftime, 28*u.day, rmin, rmax, dt_scale, omni_noicmes, output_dir)
    for ftime in dates
)

# Print summary
print("\n" + "="*70)
print("FORECAST RUN SUMMARY")
print("="*70)
for result in results[-10:]:  # Print last 10 results
    print(result)
if len(results) > 10:
    print(f"... and {len(results) - 10} more forecasts")
print("="*70)
    
    
    



# %%
