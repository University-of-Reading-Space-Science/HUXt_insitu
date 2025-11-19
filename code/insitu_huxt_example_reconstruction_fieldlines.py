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
import pandas as pd
from astropy.time import Time
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



icme_time = datetime.datetime(2024,5,20) # centre-date for reconstruction
# icmelist_path = os.path.join( os.getcwd(),
#                                 'Richardson_Cane_Porcessed_ICME_list.csv')

#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 700*u.solRad #outer boundary for HUXt runs
frame = 'sidereal'
body = 'Solar Orbiter'
body = 'Parker Solar Probe'
#body = 'Mars'
dt_scale = 10



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

# time1au_dtw_hires, lons, v1au_dtw_hires, b1au_dtw = Hin.generate_vCarr_from_OMNI_DTW(run_start, run_stop, 
#                                                                 omni_input =omni_noicmes)

# #do a run without ICMEs removed, for demonstration
# time1au_both_icmes, v1au_both_icmes, b1au_both_icmes = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
#                                                                 omni_input =omni,
#                                                                 corot_type='both')


# #put dtw data on same time step as the other corotation methods
# v1au_dtw = v1au_both.copy()
# for nlon in range(0,len(v1au_dtw_hires[:,1])):
#     v1au_dtw[nlon,:] = np.interp(time1au_both, time1au_dtw_hires, v1au_dtw_hires[nlon,:])
time1au_dtw = time1au_both
    
#now map each timestep back to the inner boundary
#================================================
vcarr_rmin_both = v1au_both.copy()
# vcarr_rmin_dtw = v1au_dtw.copy()
# vcarr_rmin_both_icmes = v1au_both_icmes.copy()

for i in range(0, len(time1au_both)):
    #get the Earth heliocentric distance at this time
    Earth_R_km = hcoords.earth_R(time1au_both[i]) *u.km
    #Map from 215 rto 21.5 rS
    
    vcarr_rmin_both[:,i] = Hin.map_v_boundary_inwards(v1au_both[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    # vcarr_rmin_dtw[:,i] = Hin.map_v_boundary_inwards(v1au_dtw[:,i]*u.km/u.s, 
    #                                 Earth_R_km.to(u.solRad), rmin)
    # vcarr_rmin_both_icmes[:,i] = Hin.map_v_boundary_inwards(v1au_both_icmes[:,i]*u.km/u.s, 
    #                                 Earth_R_km.to(u.solRad), rmin)
    


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
# vcarr_rmin_dtw_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_dtw, 
#                       data_dir =  data_dir)



# <codecell> run with field-line tracing

#trace a bunch of field lines from a range of evenly spaced Carrington longitudes
lon_grid = H._zerototwopi_((np.arange(-180, 180, 2)*u.deg).to(u.rad)) *u.rad


#change the run duration params
run_start = icme_time - datetime.timedelta(days=15)
run_stop = icme_time + datetime.timedelta(days=2)
simtime = (run_stop-run_start).days * u.day

model = Hin.set_time_dependent_boundary(vcarr_rmin_both_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = frame,
                            track_cmes = False)

#get the DONKI coneCMEs
cme_list = Hin.get_DONKI_cme_list(model, 
                                  run_start - datetime.timedelta(days = 15), run_stop)
#change each CME to fixed duration
for cme in cme_list:
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s

model.solve(cme_list, streak_carr = lon_grid)

HA.plot_earth_timeseries(model)
# <codecell> compute field line length between the Sun and a given body

def polar_line_length(r, lon):
    """
    Compute length of line defined by (r, lon) points.
    r    : array-like, radii
    lon  : array-like, angles (radians)
    """
    r = np.asarray(r, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # Convert to Cartesian
    x = r * np.cos(lon)
    y = r * np.sin(lon)

    # Differences between successive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Segment lengths
    seg_lengths = np.sqrt(dx**2 + dy**2)

    return np.sum(seg_lengths)

# time = 10*u.day
# fig, ax = HA.plot(model, time)



def find_closest_streakline_to_target(model, target_r, target_lon, id_t):
    #find the closest of the tracked streaklines to Earth
    #====================================================
    
    #get Erath position
    earth_pos = model.get_observer('EARTH')
    deltalon = 0.0 * u.rad
    if model.frame == 'sidereal':
        earth_pos = model.get_observer('EARTH')
        deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]
    
    #get teh target position in HUXt coords


    target_lon = H._zerototwopi_(target_lon + deltalon)
    
    
    #get all the streaklines at this time
    slines = model.streak_particles_r[id_t,:,:,:]
    nlines = len(slines[:,0,0])
    
    
    #find the number of times the field line wraps
    nwrap = len(slines[0,:,0])
    lon = model.lon.value
    
    #loop through each streakline, interpolate longitude to Earth_r and find the closest value
    sline_E_lon_r = []
    for iline in range(0,nlines):
    
        #unwrap the field line
        lon_unwrap = [] 
        r_unwrap = []
        for iwrap in range(0,nwrap):
            lon_unwrap.append(lon + iwrap*2*np.pi)
            r_unwrap.append(slines[iline, iwrap, :].value)
        lon_unwrap = np.concatenate(lon_unwrap)
        r_unwrap = (np.concatenate(r_unwrap)*u.km).to(u.solRad).value
        
        #find the distance at teh target lon for each wrapped portion of the field line
        r = []
        for iwrap in range(0,nwrap):
            r.append(np.interp(target_lon + iwrap*2*np.pi, lon_unwrap, r_unwrap))
        #print(r)
        
        #add the r value for the field wrap that is closest to the target r
        r = np.array(r, dtype=float)
        if np.any(~np.isnan(r)):
            # mask out NaNs
            mask = ~np.isnan(r)
            # compute absolute differences
            iclosest = np.argmin(np.abs(r[mask] - target_r))
            # translate back to original index
            iclosest = np.arange(len(r))[mask][iclosest]
            
            sline_E_lon_r.append(r[iclosest])
        else:
            sline_E_lon_r.append(np.nan)
        
    #find the streakline closest to Earth
    sline_E_lon_r = np.array(sline_E_lon_r, dtype=float)
    if np.any(~np.isnan(sline_E_lon_r)):

        # mask out NaNs
        mask = ~np.isnan(sline_E_lon_r)
        # compute absolute differences
        i_streak = np.argmin(np.abs(sline_E_lon_r[mask] - target_r))
        # translate back to original index
        i_streak = np.arange(len(sline_E_lon_r))[mask][i_streak]
        sline = slines[i_streak, :, :]

    else:
        i_streak = np.nan
        sline = [np.nan, np.nan]
    
    return i_streak, sline



#get the target location on hte model timestep
times =  model.time_out + model.time_init 
if ((body == 'Solar Orbiter') or (body == 'Parker Solar Probe')):
    
    coord = get_horizons_coord(body, {'start': Time(run_start), 'stop': Time(run_stop), 'step': '1H'})
    smjd = Time(run_start).mjd
    fmjd = Time(run_stop).mjd
    
    # make a dataframe from this
    coords = pd.DataFrame()
    coords['r_AU'] = coord.radius.value
    coords['lon_heeq'] = H._zerototwopi_(coord.lon.value * np.pi/180)
    coords['lat_heeq'] = coord.lat.value * np.pi/180
    coords['mjd'] = np.linspace(smjd, fmjd, len(coords))
    coords['datetime'] = Time(coords['mjd'] + 2400000.5, format='jd').to_datetime()
    
    #interpolate onto model tiemstep
    target_pos_r = (np.interp(times.mjd, 
                             coords['mjd'].to_numpy(), 
                             coords['r_AU'].to_numpy() ) *u.AU).to(u.km)
    target_pos_lon = np.interp(times.mjd, coords['mjd'], coords['lon_heeq'] ) *u.rad
else:
    print('here')
    target_pos = model.get_observer(body)
    target_pos_r = target_pos.r
    target_pos_lon = target_pos.lon
    
    

#loop through each output time, find the closest field line to the target and 
#measure the fieldline length
lengths = []
delta_r = []
for id_t in range(0, len(model.time_out)):
    
    #get the streakline at closest to the target location
    target_r = target_pos_r[id_t].to(u.solRad).value
    target_lon = target_pos_lon[id_t]
    
    i_streak, sline = find_closest_streakline_to_target(model, target_r, target_lon, id_t)
     
    lon = model.lon.value       
    
    
    #unwrap this target-connected fieldline
    nwrap = len(sline[:,0])
    lon_unwrap = [] 
    this_sline_r = []
    for iwrap in range(0,nwrap):
        lon_unwrap.append(lon + iwrap*2*np.pi)
        this_sline_r.append(sline[iwrap, :].value)
    line_lon = np.concatenate(lon_unwrap)
    line_r = (np.concatenate(this_sline_r)* u.km).to(u.solRad).value.tolist()
    
    #Measure the length of the selected field line between inner boundary and target
    #==============================================================================
    
    # get the non-nan values 
    mask = np.isfinite(line_r)
    plotlon = np.array(line_lon)[mask]
    plotr = np.array(line_r)[mask]
    
    #connect it to the inner boundary
    #==========================
    r_min = model.r[0].to(u.solRad).value
    dr = plotr[-1] - r_min
    # compute the lon of the footpoint assuming a constant solar wind speed
    dt = (dr * u.solRad / (350 *u.km /u.s)).to(u.s)
    dlon_streak = (2*np.pi)*(dt/model.rotation_period).value 
    inner_lon = plotlon[-1] + dlon_streak
    #add this inner point
    plotr = np.append(plotr, r_min)
    plotlon = np.append(plotlon, inner_lon)
    
    #connect it to the target r
    #==========================
    #get Earth position
    earth_pos = model.get_observer('EARTH')
    
    #get the target position in HUXt coords
    deltalon = 0.0 * u.rad
    if model.frame == 'sidereal':
        earth_pos = model.get_observer('EARTH')
        deltalon = earth_pos.lon_hae[id_t] - earth_pos.lon_hae[0]
    target_lon = H._zerototwopi_(target_lon + deltalon)
    
    #find the distance at the target lon for each wrapped rotation
    r = []
    for iwrap in range(0,nwrap):
        r.append(np.interp(target_lon + iwrap*2*np.pi, plotlon, plotr))
    
    #add the r value that is closest to the target r
    r = np.array(r, dtype=float)
    if np.any(~np.isnan(r)):
        # mask out NaNs
        mask = ~np.isnan(r)
        # compute absolute differences
        iwrap_closest = np.argmin(np.abs(r[mask] - target_r))
        # translate back to original index
        iwrap_closest = np.arange(len(r))[mask][iwrap_closest]
    target_lon_unwrap = target_lon + 2*np.pi*iwrap_closest
    # r_streak_end = np.interp(target_lon_unwrap, plotlon, plotr)
    
    #find where the streakline crosses the target r
    crossings =  [i for i in range(len(plotr)-1)
           if (plotr[i] - target_r) * (plotr[i+1] - target_r) <= 0]
    
    

    
    
    #if there's more than one crossing, find the closest in longitude
    if len(crossings)>1:
        #print('Warning, mulitple intersections of streakline with target R, id_t = ' +str(id_t))
        
        crossing_lons = []
        for icrossing in range(0, len(crossings)):
            crossing_lons.append(plotlon[crossings[icrossing]])
        
        #find the closest crossing in longitude
        icrossing_closest = np.argmin(np.abs(crossing_lons - target_lon_unwrap))

        idx = crossings[icrossing_closest]

    elif len(crossings) ==1:
        idx = crossings[0]
    else:
        idx = np.nan
            
    if np.isnan(idx):
        lengths.append(np.nan)
    else:   
        #interpolate the final r between the crossong longs
        end_lon = np.interp(target_r, plotr[idx:idx+1], plotlon[idx:idx+1])
        
        # #prune the streakline to the crossing
        connectr = plotr[idx+1:]
        connectlon = plotlon[idx+1:]
        
        #add the last point in
        connectr = np.insert(connectr, 0, target_r)
        connectlon = np.insert(connectlon, 0, end_lon)
        
        
        
        # #find teh target lon id for that wrapped field line
        
            
        #     #ditch all values beyond this one
        #     r[iclosest]
        
        # fig, ax = HA.plot(model, time = model.time_out[id_t])
        # ax.plot(plotlon, plotr, 'r')
        # ax.plot(connectlon, connectr,'w--')
        
        #measure the length of this fieldline to the target
        length = polar_line_length(connectr, connectlon)
        
        lengths.append(length)
    


  
plt.figure()
plt.plot(times.datetime, np.array(lengths)/215)
plt.plot(coords['datetime'], coords['r_AU'], 'r')
plt.xlabel("Date")
plt.ylabel("Length [AU]")
plt.show()

# <codecell> Animation

#HA.animate(model, tag = 'Mars_FieldLineLength_May2024.mp4', duration=20, fps=20)