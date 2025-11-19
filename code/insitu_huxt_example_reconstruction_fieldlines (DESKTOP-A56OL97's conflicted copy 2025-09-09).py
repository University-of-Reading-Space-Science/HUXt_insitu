# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:15:24 2025

@author: vy902033
"""

# -*- coding: utf-8 -*-
"""
a script to run InSitu-HUXt reconstruction for a given period, and trace field
lines to Earth
@author: mathe
"""



import datetime
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import matplotlib.dates as mdates
from astropy.time import Time
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import sunpy_soar # this is required
from sunpy.net import Fido
import sunpy.net.attrs as a
import sunpy.timeseries
from sunpy.coordinates import HeliocentricEarthEcliptic, get_horizons_coord

#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset
import joblib
import onnxruntime as ort


#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA

#Mattlab libraries
import helio_coords as hcoords
import insitu



icme_time = datetime.datetime(2024,5,11) # centre-date for reconstruction
# icmelist_path = os.path.join( os.getcwd(),
#                                 'Richardson_Cane_Porcessed_ICME_list.csv')

#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs


icme_list = 'CaneRichardson'#'DONKI'
pre_icme_buffer = 0.2 #days
post_icme_buffer = 1 #days
#fill values for ICME removal if interpolation isn't possible
vsw_fill = 450 *u.km/u.s
bx_fill = 0

yy = (250, 1200)
plot_days = [6,20,34]


download_now = False #download UKMO WSA solution as example. Data provided, so no necessary

#directory structure
cwd = os.path.dirname(os.path.realpath(__file__))
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
data_dir = os.path.join(os.path.dirname(cwd), 'data')
icmelist_path = os.path.join(data_dir, 'Richardson_Cane_Porcessed_ICME_list.csv')



# <codecell> Download and process OMNI

#compute the run start and end times so that the ICME is in the centre of the window

run_start = icme_time - datetime.timedelta(days=20)
run_stop =  icme_time + datetime.timedelta(days=20)


simtime = (run_stop-run_start).days * u.day

# download an additional 28 days either side
dl_starttime = run_start - datetime.timedelta(days=28*1.5)
dl_endtime = run_stop + datetime.timedelta(days=28*1.5)


omni = Hin.get_omni(dl_starttime, dl_endtime)

#if the data don't span a large enough time range, repeat the last 27 days
data_end_date = omni['datetime'][len(omni)-1]
if data_end_date < run_stop:
    mask = (omni['datetime'] >= data_end_date - datetime.timedelta(days = 27.27))
    datachunk = omni[mask]
    datachunk.loc[:,'datetime'] = datachunk['datetime'] + datetime.timedelta(days = 27.27)
    datachunk.loc[:,'mjd'] = datachunk['mjd'] + 27.27
    #concatonate teh dataframes
    omni = pd.concat([omni, datachunk], ignore_index=True)



 



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
    




# <codecell> set up inner boundary conditions for HUXt form cortoation of OMNI

#create  vCarr array with the omni time series at 1 AU
#======================================================
time1au_both, v1au_both, b1au_both = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =omni_noicmes,
                                                                corot_type='both')

time1au_dtw_hires, lons, v1au_dtw_hires, b1au_dtw = Hin.generate_vCarr_from_OMNI_DTW(run_start, run_stop, 
                                                                omni_input =omni_noicmes)

#do a run without ICMEs removed, for demonstration
time1au_both_icmes, v1au_both_icmes, b1au_both_icmes = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =omni,
                                                                corot_type='both')


#put dtw data on same time step as the other corotation methods
v1au_dtw = v1au_both.copy()
for nlon in range(0,len(v1au_dtw_hires[:,1])):
    v1au_dtw[nlon,:] = np.interp(time1au_both, time1au_dtw_hires, v1au_dtw_hires[nlon,:])
time1au_dtw = time1au_both
    
#now map each timestep back to the inner boundary
#================================================
vcarr_rmin_both = v1au_both.copy()
vcarr_rmin_dtw = v1au_dtw.copy()
vcarr_rmin_both_icmes = v1au_both_icmes.copy()

for i in range(0, len(time1au_both)):
    #get the Earth heliocentric distance at this time
    Earth_R_km = hcoords.earth_R(time1au_both[i]) *u.km
    #Map from 215 rto 21.5 rS
    
    vcarr_rmin_both[:,i] = Hin.map_v_boundary_inwards(v1au_both[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_dtw[:,i] = Hin.map_v_boundary_inwards(v1au_dtw[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_both_icmes[:,i] = Hin.map_v_boundary_inwards(v1au_both_icmes[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    


def correct_inner_vlon_cnn_onnx(v_inner_array,
                                data_dir=os.path.join(os.getenv('DBOX'), 'Apps', 
                                                      'Overleaf', 'Recurrent-HUXt')):
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




vcarr_rmin_both_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_both, 
                      data_dir = data_dir)
vcarr_rmin_dtw_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_dtw, 
                      data_dir =  data_dir)















# <codecell> run with field-line tracing

#trace a bunch of field lines from a range of evenly spaced Carrington longitudes
lon_grid = H._zerototwopi_((np.arange(-90, 90, 2)*u.deg).to(u.rad)) *u.rad


#change the run duration params
run_start = icme_time - datetime.timedelta(days=10)
run_stop = icme_time + datetime.timedelta(days=2)
simtime = (run_stop-run_start).days * u.day

model = Hin.set_time_dependent_boundary(vcarr_rmin_dtw_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=1, latitude=0*u.deg, frame = 'synodic',
                            track_cmes = False)

#get the DONKI coneCMEs
cme_list = Hin.get_DONKI_cme_list(model, 
                                  run_start - datetime.timedelta(days = 10), run_stop)
#change each CME to fixed duration
for cme in cme_list:
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s

model.solve(cme_list, streak_carr = lon_grid)

HA.plot_earth_timeseries(model)
# <codecell> compute field line lengths

# time = 10*u.day
# fig, ax = HA.plot(model, time)


#loop through each streakline, interpolate longitude to Earth_r and find the closest value
id_t = 2000
n_line = 40
earth_pos = model.get_observer('EARTH')
Earth_r = earth_pos.r[id_t].to(u.km).value


#inter the lon of each streakline to Earth_r
slines = model.streak_particles_r[id_t,:,:,:]
lon = model.lon.value
  
#find the number of times the field line wraps
nwrap = len(slines[0,:,0])

#unwrap the field line
lon_unwrap = [] 
this_sline_r = []
for iwrap in range(0,nwrap):
    lon_unwrap.append(lon + iwrap*2*np.pi)
    this_sline_r.append(slines[n_line, iwrap, :])
lon_unwrap = np.concatenate(lon_unwrap)  
this_sline_r = np.concatenate(this_sline_r)  


sline_E_lon_r = np.interp(2*np.pi, lon_unwrap, this_sline_r)

plt.figure()
plt.plot(lon_unwrap, this_sline_r)
plt.plot(2*np.pi, sline_E_lon_r, 'o')
