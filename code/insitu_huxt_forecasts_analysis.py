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

import insitu
import helio_coords as hcoords

start_time = datetime.datetime(2022,1,1)
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
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
data_dir = os.path.join(os.path.dirname(cwd), 'data')
icmelist_path = os.path.join(data_dir, 'Richardson_Cane_Porcessed_ICME_list.csv')


output_dir = '~/InSitu_HUXt/'

# <codecell> CNN function



def correct_inner_vlon_cnn_onnx(v_inner_array,
                                data_dir=os.path.join(os.getenv('DBOX'), 'python_repos', 
                                                      'HUXt_insitu', 'data')):
    """
    Corrects solar wind speed as a function of longitude using a 1D CNN model
    trained to account for stream interactions during backmapping from 1 AU 
    to 0.1 AU. Uses ONNX, rather than pytorch.

    Parameters:
    - v_inner_array: np.ndarray of shape (128, N) [speed vs. longitude & samples]
    - data_dir: directory containing saved scalers and ONNX model

    Returns:
    - Y_pred: np.ndarray of shape (128, N), CNN-corrected speed
    """

    # Load scalers
    y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))
    x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))

    # Transpose input to shape (N, 128) so each row is a sample
    vcarr_scaled = x_scaler.transform(v_inner_array.T)  # (N, 128)

    # Reshape to ONNX expected input: (batch_size, channels=1, length=128)
    X_input = vcarr_scaled[:, np.newaxis, :].astype(np.float32)  # (N, 1, 128)

    # Load ONNX model
    onnx_path = os.path.join(data_dir, 'CNN_model.onnx')
    ort_session = ort.InferenceSession(onnx_path)

    # Run inference
    input_name = ort_session.get_inputs()[0].name
    output = ort_session.run(None, {input_name: X_input})
    Y_pred_scaled = output[0]  # (N, 1, 128)

    # Postprocess: squeeze to (N, 128)
    Y_pred_scaled = Y_pred_scaled.squeeze(1)

    # Inverse transform
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)  # (N, 128)

    # Transpose back to (128, N) to match input shape
    return Y_pred.T



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
omni_noicmes = omni_noicmes.set_index('datetime')
omni_noicmes[['V', 'BX_GSE']] = omni_noicmes[['V', 'BX_GSE']].interpolate(method='time').ffill().bfill()
omni_noicmes = omni_noicmes.reset_index()



    
# <codecell> a single forecast


def insituHUXt_forecast(ftime, simtime = 27.27*u.day, omni_input = None):
    run_start = ftime 
    #run_stop = ftime + datetime.timedelta(days=simtime.value)
    #simtime = (run_stop-run_start).days * u.day
    
    #if no omni data provided, download it
    if omni_input is None:
        dl_starttime = ftime - datetime.timedelta(days=28)
        dl_endtime = ftime
    
        omni_input = Hin.get_omni(dl_starttime, dl_endtime)
    
    
    #add the carrington longitude to the omni data
    def remainder(cr_frac):
        if np.isscalar(cr_frac):
            return int(np.floor(cr_frac))
        else:
            return np.floor(cr_frac).astype(int)
    cr_frac = sun.carrington_rotation_number(omni_input['datetime'])
    cr = remainder(cr_frac)
    omni_input['lon_carr'] = 2 * np.pi * (1 - (cr_frac - cr)) 
    
    
    #create vCarr  with the omni time series at 1 AU
    #======================================================
    
    #unwrap the carr long
    unwrapped = np.unwrap(omni_input['lon_carr'], discont=np.pi)
    #find the current value
    idx = np.argmin(np.abs(omni_input['datetime'] - ftime))
    curr_lon = unwrapped[idx] 
    #find the data up to 2 pi previously 
    mask = ((unwrapped < curr_lon + 2*np.pi) & (unwrapped >= curr_lon))
    omni_chunk = omni_input.loc[mask].reset_index(drop=True)
    
    #sort by carrington lon
    omni_lon = omni_chunk.sort_values(by='lon_carr').reset_index(drop=True)
    
    #now map back to the inner boundary
    Earth_R_km = hcoords.earth_R(Time(ftime).mjd) *u.km
    vcarr_rmin_back = Hin.map_v_boundary_inwards(omni_lon['V'].to_numpy()*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    
    #interp to typical HUXt resolution
    dphi = 2*np.pi/H.huxt_constants()['nlong']
    longs = np.arange(dphi/2, 2*np.pi, dphi)
    vlon = np.interp(longs, omni_lon['lon_carr'], vcarr_rmin_back)
    
    # apply the CNN to the backmapped data
    vcarr_rmin_back_cnn = correct_inner_vlon_cnn_onnx(vlon.reshape(-1, 1), 
                          data_dir =  data_dir)
    
    
    #set up the model run
    cr, cr_lon_init = Hin.datetime2huxtinputs(run_start)
    model = H.HUXt(v_boundary = vcarr_rmin_back_cnn.flatten() * u.km/u.s, 
                   #v_boundary = vlon, 
                              cr_num = cr, cr_lon_init=cr_lon_init,
                               simtime = simtime, r_min=rmin, r_max=rmax, 
                                dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
                                track_cmes = False, lon_out = 0*u.rad)
    model.solve([])  
    
   
    return model


# ftime = start_time
# model = insituHUXt_forecast(ftime, simtime = 27.27*u.day, omni_input = omni_noicmes)
# HA.plot_earth_timeseries(model)
# ts = HA.get_observer_timeseries(model, observer='Earth')



# <codecell> set up the forecasts and save the data

#the list of forecasts at different lead times
for_list = []
for n in range(min_lt, max_lt):
    for_list.append([])
    
dates = pd.date_range(start=start_time, end=stop_time, freq="D")    
for ftime in dates:
    print('Running InSitu-HUXt for ' + ftime.strftime("%Y-%m-%d"))
    model = insituHUXt_forecast(ftime, simtime = 27.27*u.day, omni_input = omni_noicmes)
    ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning= True)
    
    #save the data
    yr_str = str(ftime.year)
    mn_str = f"{ftime.month:02}"
    dy_str = f"{ftime.day:02}"
    dir_path = os.path.join(output_dir, yr_str)
    #create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    outname = 'InSitu_STA_' + yr_str +mn_str + dy_str + '.dat'
    ts.to_csv(os.path.join(dir_path,outname))
    
    
    
# # <codecell> Analyse InSitu HUXt runs
# for ftime in dates:    
#     print('Analysing InSitu-HUXt for ' + ftime.strftime("%Y-%m-%d"))
    
    
#     #set up the directory path
#     yr_str = str(ftime.year)
#     mn_str = f"{ftime.month:02}"
#     dy_str = f"{ftime.day:02}"
    
#     filename_wildcard = 'InSitu_Earth_' + yr_str + mn_str + dy_str + '.dat'
#     filepath_wildcard = os.path.join(output_dir, yr_str, filename_wildcard)
    
#     # Find matching files
#     matching_files = glob.glob(filepath_wildcard)
    
#     #if the file exists, read in the sub-Earth values
#     if matching_files:
#         print('Loading: ' + ftime.strftime('%Y-%m-%d'))
#         filepath = matching_files[0]
        
#         ts = pd.read_csv(filepath, index_col=0, parse_dates=["time"] )
        
#     #process the HUXt input into forecasts of different lead times
#     for ifor in range(0,len(for_list)):
#         start_window = ftime + datetime.timedelta(days = ifor)
#         stop_window = ftime + datetime.timedelta(days = ifor + 1.0)
        
#         mask = ((ts['time'] >= start_window)  & (ts['time'] <= stop_window))
#         for_list[ifor].append(ts.loc[mask])

    
# #combine the dataframes at each forecast lead time.
# combined_for_list = [pd.concat(df_list, ignore_index=True) for df_list in for_list]

# fig, axs = plt.subplots(2, 1, figsize=(14, 8))

# ax = axs[1]
# ax.plot(combined_for_list[plot_lead_time-1]['time'], 
#         combined_for_list[plot_lead_time-1]['vsw'], 'r', label = 'InSitu-HUXt')
# ax.plot(omni['datetime'], omni['V'], 'k', label = 'OMNI')
# ax.legend()
# ax.set_ylabel('Solar wind speed [km/s]')
# ax.set_xlim((start_time, stop_time))
# ax.set_ylim((200, 900))

# # <codecell>  WSA-HUXt forecasts for the same time period
# output_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC', 'CCMC_WSA_HUXt')

# #the list of forecasts at different lead times
# for_wsa_list = []
# for n in range(min_lt, max_lt):
#     for_wsa_list.append([])
    
# dates = pd.date_range(start=start_time, end=stop_time, freq="D")    
# for ftime in dates:
#     print('Analysing WSA-HUXt for ' + ftime.strftime("%Y-%m-%d"))
    
    
#     #set up the directory path
#     yr_str = str(ftime.year)
#     mn_str = f"{ftime.month:02}"
#     dy_str = f"{ftime.day:02}"
    
#     filename_wildcard = 'HUXt_Earth_' + yr_str + mn_str + dy_str + '.dat'
#     filepath_wildcard = os.path.join(output_dir, yr_str, filename_wildcard)
    
#     # Find matching files
#     matching_files = glob.glob(filepath_wildcard)
    
#     #if the file exists, read in the sub-Earth values
#     if matching_files:
#         print('Loading: ' + ftime.strftime('%Y-%m-%d'))
#         filepath = matching_files[0]
        
#         ts = pd.read_csv(filepath, index_col=0, parse_dates=["time"] )
    
#     #process the HUXt input into forecasts of different lead times
#     for ifor in range(0,len(for_list)):
#         start_window = ftime + datetime.timedelta(days = ifor)
#         stop_window = ftime + datetime.timedelta(days = ifor + 1.0)
        
#         mask = ((ts['time'] >= start_window)  & (ts['time'] <= stop_window))
#         for_wsa_list[ifor].append(ts.loc[mask])

    
# #combine the dataframes at each forecast lead time.
# combined_for_wsa_list = [pd.concat(df_list, ignore_index=True) for df_list in for_wsa_list]



ax = axs[0]
ax.plot(combined_for_wsa_list[plot_lead_time-1]['time'], 
        combined_for_wsa_list[plot_lead_time-1]['vsw'], 'r', label = 'WSA-HUXt')
ax.plot(omni['datetime'], omni['V'], 'k', label = 'OMNI')
ax.legend()
ax.set_ylabel('Solar wind speed [km/s]')
ax.set_xlim((start_time, stop_time))
ax.set_ylim((200, 900))


