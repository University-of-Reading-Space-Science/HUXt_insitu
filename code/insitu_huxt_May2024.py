# -*- coding: utf-8 -*-
"""
a script to run insitu-HUXt and preoduce figures for Owens et al 2025.
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

# <codecell> display the accerelation profile
v0p1au = [200, 300, 400, 500, 600, 700]; Nv = len(v0p1au)
rs = np.arange(1, 230, 1)


constants = H.huxt_constants()
alpha = constants['alpha']  # Scale parameter for residual SW acceleration
rH = constants['r_accel'].to(u.kilometer).value  # Spatial scale parameter for residual SW acceleration
Tsyn = constants['synodic_period'].to(u.s).value
r_0 = (30 * u.solRad).to(u.km).value
r_orig = rmin.to(u.km).value
rkm = (rs*u.solRad).to(u.km).value

v_func_r = np.ones((len(rs), len(v0p1au)))
for i, v in enumerate(v0p1au):

    # Compute the 30 rS speed
    v0 = v / (1 + alpha * (1 - np.exp(-(r_orig - r_0) / rH)))

    # comppute new speed at each r
    v_r = np.ones((len(rs)))
    for n, r in enumerate(rkm):
        
        v_r[n] = v0 * (1 + alpha * (1 - np.exp(-(r - r_0) / rH)))
    
    v_func_r[:,i] = v_r
    


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for i in range(0, Nv):
    ax.plot(rs,  v_func_r[:,i], 'r')
    ax.text(130, v_func_r[150, i] - 28, r"V(21.5$r_S$) = " + str(v0p1au[i]) + ' km/s',
            fontsize=14)
    
    
yy = ax.get_ylim()
ax.set_ylim(yy)
ax.set_ylabel(r'$V_R$ [km/s]')
ax.set_xlabel(r'Radial distance [$r_S$]')
ax.plot([21.5, 21.5], yy, 'k--')
ax.plot([215, 215], yy, 'k--')
ax.set_xlim((0,230))
ax.text(21.5 -8, yy[1] +20, '0.1 AU\n=21.5' +r'$r_S$', fontsize=14)
ax.text(215 -8, yy[1] +20, '1 AU\n=215' +r'$r_S$', fontsize=14)   


fig.savefig(os.path.join(figdir, 'acc_profile.pdf'))

# <codecell> Some functions from Mattlab and additional data

#add the Solar Orbiter ICMEs by hand
icmes_solo = pd.DataFrame({
    "Shock_time": [ "2024-04-20 01:00", "2024-05-09 15:59", "2024-05-21 08:09"],
    "ICME_start": [ "2024-04-20 07:50", "2024-05-09 18:21", "2024-05-21 18:45"],
    "ICME_end": [   "2024-04-21 01:30", "2024-05-09 22:20", "2024-05-22 09:53"]})

# Convert all columns to datetime
icmes_solo = icmes_solo.apply(pd.to_datetime)


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
    


# <codecell> plot the 1 AU data summary
xx = (omni_noicmes.loc[0,'datetime'], omni_noicmes.loc[len(omni_noicmes) - 1,'datetime'])

#create  vCarr array with the omni time series at 1 AU
#======================================================
time1au_both, v1au_both, b1au_both = Hin.generate_vCarr_from_OMNI(xx[0], xx[1], 
                                                                omni_input =omni_noicmes,
                                                                corot_type='both')

time1au_back, v1au_back, b1au_back = Hin.generate_vCarr_from_OMNI(xx[0], xx[1], 
                                                                omni_input =omni_noicmes,
                                                                corot_type='back')

time1au_forward, v1au_forward, b1au_forward = Hin.generate_vCarr_from_OMNI(xx[0], xx[1], 
                                                                omni_input =omni_noicmes,
                                                                corot_type='forward')

time1au_dtw, lon_grid_dtw, v1au_dtw, b1au_dtw = Hin.generate_vCarr_from_OMNI_DTW(xx[0], xx[1], 
                                                                omni_input =omni_noicmes)



# <codecell> plot the summary of V at 1 AU


xx = (omni_noicmes.loc[0,'datetime'], omni_noicmes.loc[len(omni_noicmes) - 1,'datetime'])
yy = (250, 1100)
vv = (300,550)


fig, axs = plt.subplots(6, 1, figsize=(10, 12))


#time series with and without CMEs
#=================================
ax = axs[0]
ax.set_ylim(yy)
ax.set_xlim(xx)

#add in ICMEs
count =0
for i in range(0, len(icmes)):
    icme_start = icmes['Shock_time'][i] 
    icme_stop = icmes['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        #print('plotting ICME #' +str(i))
    
        if count == 0:
            ax.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            ax.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    

ax.plot(omni['datetime'], omni['V'],'k', label='OMNI')   
ax.plot(omni_noicmes['datetime'], omni_noicmes['V'],'r', label='OMNI, no ICMEs')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

ax.legend(bbox_to_anchor=(0.17, 1.03), ncol=3)
ax.set_ylabel(r'V [km/s]')  
ax.text(0.01, 0.98, "(a)",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 

ax.set_xticklabels([])



#plot of long and time
#=====================
#compute the Carrington longitude of Earth
temp = hcoords.carringtonlatlong_earth(omni_noicmes['mjd'])
omni_noicmes['Carr_lon'] = temp[:,1]

ax = axs[1]
ax.set_ylim(0,360)
ax.set_xlim(xx)

sc = ax.scatter( 
    omni_noicmes['datetime'], omni_noicmes['Carr_lon'] * 180/np.pi,
    c=omni_noicmes['V'],  cmap='viridis', s=100, edgecolor='none',
    marker='_',  norm=plt.Normalize(vv[0], vv[1])
)

cax = fig.add_axes([0.87, 0.12, 0.02, 0.57])  # Adjust as needed
cbar = fig.colorbar(sc, cax=cax)
fig.text(0.87, 0.72, 'V [km/s]', va='center', rotation='horizontal',
         fontsize =16)
ax.set_xticklabels([])
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.set_yticks([0, 90, 180, 270])
ax.text(0.01, 0.98, "(b)",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 



#convert to datetime and compute the longitudes
dphi = 2*np.pi/128 
lon_edges = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
t = Time(time1au_both, format='mjd')
datetime_array = t.to_datetime()


#plot of gridded  long and time
#===========================
ax = axs[2]
ax.pcolormesh(datetime_array, lon_edges*180/np.pi, v1au_back, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.set_xticklabels([])
ax.text(0.01, 0.98, "(c) Corotation - back",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 

ax = axs[3]
ax.pcolormesh(datetime_array, lon_edges*180/np.pi, v1au_forward, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.set_xticklabels([])
ax.text(0.01, 0.98, "(d) Corotation - forward",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 

ax = axs[4]
ax.pcolormesh(datetime_array, lon_edges*180/np.pi, v1au_both, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.set_xticklabels([])
ax.text(0.01, 0.98, "(e) Corotation - both",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 


ax = axs[5]
t = Time(time1au_dtw, format='mjd')
datetime_array = t.to_datetime()
ax.pcolormesh(datetime_array, lon_edges*180/np.pi, v1au_dtw, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
ax.text(0.01, 0.98, "(f) DTW",  transform=ax.transAxes,  ha='left', va='top', fontsize = 14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')) 


ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.set_xlabel(r'Date (2024)')

fig.text(0.035, 0.43, 'Carrington longitude [deg]', va='center', rotation='vertical',
         fontsize =16)
fig.subplots_adjust(right=0.85, hspace=0.1)

fig.savefig(os.path.join(figdir, 'May2024_1au_summary.pdf'))

# <codecell> set up inner boundary conditions for HUXt form cortoation of OMNI

#create  vCarr array with the omni time series at 1 AU
#======================================================
time1au_both, v1au_both, b1au_both = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =omni_noicmes,
                                                                corot_type='both')
time1au_back, v1au_back, b1au_back = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =omni_noicmes,
                                                                corot_type='back')
time1au_forward, v1au_forward, b1au_forward = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =omni_noicmes,
                                                                corot_type='forward')

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
vcarr_rmin_back = v1au_back.copy()
vcarr_rmin_forward = v1au_forward.copy()
vcarr_rmin_dtw = v1au_dtw.copy()
vcarr_rmin_both_icmes = v1au_both_icmes.copy()

for i in range(0, len(time1au_both)):
    #get the Earth heliocentric distance at this time
    Earth_R_km = hcoords.earth_R(time1au_both[i]) *u.km
    #Map from 215 rto 21.5 rS
    
    vcarr_rmin_both[:,i] = Hin.map_v_boundary_inwards(v1au_both[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_back[:,i] = Hin.map_v_boundary_inwards(v1au_back[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_forward[:,i] = Hin.map_v_boundary_inwards(v1au_forward[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_dtw[:,i] = Hin.map_v_boundary_inwards(v1au_dtw[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    vcarr_rmin_both_icmes[:,i] = Hin.map_v_boundary_inwards(v1au_both_icmes[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
    


# # <codecell> CNN it (tensor flow). 

# data_dir = os.path.join(os.getenv('DBOX'),'Apps','Overleaf','Recurrent-HUXt' )
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib

# cnn = load_model(os.path.join(data_dir, 'CNN.h5'))
# x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler.save'))
# y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler.save'))


# vcarr_scaled = x_scaler.transform(vcarr_rmin_both.T)
# # Reshape for Conv1D: (samples, length, channels)
# vcarr_scaled = vcarr_scaled[..., np.newaxis]

# # Predict and Inverse Transform from test data sets
# vcarr_pred_scaled = cnn.predict(vcarr_scaled)
# vcarr_pred = y_scaler.inverse_transform(vcarr_pred_scaled.squeeze())
# vcarr_rmin_both_cnn = vcarr_pred.T




# <codecell> CNN it (torch). 


# def correct_inner_vlon_cnn(v_inner_array, 
#                            data_dir =  os.path.join(os.getenv('DBOX'),'Apps','Overleaf','Recurrent-HUXt' )):
    
#     """
#     a function to correct speed as a function of longitude that is generated 
#     by backmapping 1 AU observations to 0.1 AU in order to 
#     account for stream interactions. 
    
    
#     """
    
    

        
#     # Load scalers
#     y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))
#     x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))
    
#     vcarr_scaled = x_scaler.transform(v_inner_array.T)
#     # Reshape for Conv1D: (samples, length, channels)
#     vcarr_scaled = vcarr_scaled[..., np.newaxis]
    
#     # Convert to Torch tensor
#     vcarr_tensor = torch.from_numpy(vcarr_scaled).float()  

#     # Load test data (X_test) from earlier session or rerun the preprocessing
#     # Assuming X_test is still available in memory
#     X_test_tensor = vcarr_tensor.detach().clone().float().permute(0, 2, 1)
#     # Initialize and load model
#     #model = Conv1DCNN(X_test_tensor.shape[2])
#     # Define the model architecture again
#     class Conv1DCNN(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = nn.Sequential(
#                 nn.Conv1d(1, 32, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.Conv1d(32, 64, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.Conv1d(64, 1, kernel_size=3, padding=1)
#             )

#         def forward(self, x):
#             return self.model(x)

#     # Create model instance and load weights
#     cnn = Conv1DCNN()
#     cnn.load_state_dict(torch.load(os.path.join(data_dir, 'CNN_torch_weights.pth')))
#     cnn.eval()


#     # Predict
#     with torch.no_grad():
#         Y_pred_tensor = cnn(X_test_tensor)  # (batch, channels, length)

#     # Convert predictions and targets back to numpy
#     Y_pred_scaled = Y_pred_tensor.permute(0, 2, 1).squeeze().numpy()

#     # Inverse transform
#     Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    
    
#     return Y_pred.T




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
vcarr_rmin_back_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_back, 
                      data_dir =  data_dir)
vcarr_rmin_forward_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_forward, 
                      data_dir =  data_dir)
vcarr_rmin_dtw_cnn = correct_inner_vlon_cnn_onnx(vcarr_rmin_dtw, 
                      data_dir =  data_dir)

# data_dir = os.path.join(os.getenv('DBOX'),'Apps','Overleaf','Recurrent-HUXt' )
# import torch
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler


# cnn = torch.load(os.path.join(data_dir, 'CNN_torch.pth'))
# cnn.eval()  # Set model to evaluation mode
# x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))
# y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))


# vcarr_scaled = x_scaler.transform(vcarr_rmin_both.T)
# # Reshape for Conv1D: (samples, length, channels)
# vcarr_scaled = vcarr_scaled[..., np.newaxis]

# # Convert to Torch tensor
# vcarr_tensor = torch.from_numpy(vcarr_scaled).float()  # shape: (batch, length, channels)

# # Move to device if needed (CPU assumed)
# with torch.no_grad():
#     vcarr_pred_scaled = cnn(vcarr_tensor).numpy()      # shape: (n_samples, n_lons, 1)

# # Inverse transform and reshape
# vcarr_pred = y_scaler.inverse_transform(vcarr_pred_scaled.squeeze())  # (n_samples, n_lons)
# vcarr_rmin_both_cnn = vcarr_pred.T  # Transpose back to match original layout


# <codecell> plot the backmapped and CNN inner boundary conditions for the May events

xx = (run_start, run_stop)
vv = (250,500)

#convert to datetime and compute the longitudes
dphi = 2*np.pi/128 
lon_edges = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
t = Time(time1au_both, format='mjd')
datetime_array = t.to_datetime()


fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# ax = axs[0]
# sc = ax.pcolormesh(datetime_array, lon_edges*180/np.pi, vcarr_rmin_both_icmes, 
#               shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
# ax.set_ylim(0,360)
# ax.set_yticks([0, 90, 180, 270, 360])
# ax.set_xlim(xx)
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# ax.set_xticklabels([])
# ax.set_title('(a) OMNI (inc ICMEs), Corotation(both), backmap to 0.1 AU', fontsize = 16)


ax = axs[0]
sc = ax.pcolormesh(datetime_array, lon_edges*180/np.pi, vcarr_rmin_both, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270, 360])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.set_xticklabels([])
ax.set_title('(b) OMNI (no ICMEs), Corotation(both), backmap to 0.1 AU', fontsize = 16)

cax = fig.add_axes([0.90, 0.15, 0.02, 0.6])  # Adjust as needed
cbar = fig.colorbar(sc, cax=cax)
fig.text(0.89, 0.8, 'V [km/s]', va='center', rotation='horizontal',
         fontsize =16)

ax = axs[1]
ax.pcolormesh(datetime_array, lon_edges*180/np.pi, vcarr_rmin_both_cnn, 
              shading='auto', edgecolor = 'face',norm=plt.Normalize(vv[0], vv[1]))
ax.set_ylim(0,360)
ax.set_yticks([0, 90, 180, 270, 360])
ax.set_xlim(xx)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.set_xlabel(r'Date (2024)')
ax.set_title('(c) As above, but with CNN applied', fontsize = 16)

fig.text(0.04, 0.5, 'Carrington longitude [deg]', va='center', rotation='vertical',
         fontsize =16)
fig.subplots_adjust(right=0.85, hspace=0.3)

fig.savefig(os.path.join(figdir, 'May2024_0p1au_summary.pdf'))


# <codecell> run HUXt

#set up the model at 21.5 rS with backmapped OMNI
#========================================================


model_both_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_both_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
                            track_cmes = False)
model_both_nocmes.solve([])  

model_back_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_back_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
                            track_cmes = False)
model_back_nocmes.solve([])  

model_forward_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_forward_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
                            track_cmes = False)
model_forward_nocmes.solve([])  

model_dtw_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_dtw_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
                            track_cmes = False)
model_dtw_nocmes.solve([]) 




#Add some CMEs to the mix
#========================
model_both_cmes = Hin.set_time_dependent_boundary(vcarr_rmin_both_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic',
                            track_cmes = False)

model_back_cmes = Hin.set_time_dependent_boundary(vcarr_rmin_back_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic',
                            track_cmes = False)

model_forward_cmes = Hin.set_time_dependent_boundary(vcarr_rmin_forward_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic',
                            track_cmes = False)

model_dtw_cmes = Hin.set_time_dependent_boundary(vcarr_rmin_dtw_cnn, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic',
                            track_cmes = False)

#get the DONKI coneCMEs
cme_list = Hin.get_DONKI_cme_list(model_both_cmes , 
                                  run_start - datetime.timedelta(days = 5), run_stop)
#change each CME to fixed duration
for cme in cme_list:
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s


model_both_cmes.solve(cme_list)  
model_back_cmes.solve(cme_list)  
model_forward_cmes.solve(cme_list)  
model_dtw_cmes.solve(cme_list)  


#generate movie
#HA.animate(model_both_cmes, 'OMNI-DONKI-HUXt', duration =30)

# <codecell>Get orbiter data and coords

time_range = a.Time(run_start, run_stop) 

# Get solar wind speed data
res = Fido.search(time_range, a.soar.Product("swa-pas-mom"))
v_files = Fido.fetch(res)
ts = sunpy.timeseries.TimeSeries(v_files, concatenate=True)
v_df = ts.to_dataframe()

#average to 1 h
v_df_hourly = v_df.resample('1H', label='right', closed='right').mean()
v_df_hourly.index = v_df_hourly.index - pd.Timedelta(minutes=30)

# #Get magnetic field data
# res = Fido.search(time_range, a.soar.Product("mag-rtn-normal"))
# mag_files = Fido.fetch(res)
# # Get B time series and smooth
# ts = sunpy.timeseries.TimeSeries(mag_files, concatenate=True)
# b_df = ts.to_dataframe()


# fig, ax = plt.subplots(4, 1, figsize=(20, 10))
# ax[0].plot(v_df.index, v_df['velocity_0'], 'k-')
# ax[0].plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'r')
# ax[1].plot(b_df.index, b_df['B_RTN_0'], 'k-')
# ax[2].plot(b_df.index, b_df['B_RTN_1'], 'k-')
# ax[3].plot(b_df.index, b_df['B_RTN_2'], 'k-')


# for aa in ax[:-1]:
#     aa.set_xticklabels([])

# for aa in ax:
#     aa.set_xlim(v_df.index.min(), v_df.index.max())

# ax[0].set_ylabel('Vr (km/s)')
# ax[1].set_ylabel('Br (nT)')
# ax[2].set_ylabel('Bt (nT)')
# ax[3].set_ylabel('Bn (nT)')

# ax[3].set_xlabel('Date')

# fig.subplots_adjust(hspace=0.05)



#get Orbiter location
body = 'Solar Orbiter'

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

# convert to Carrington longitude for plotting purposes
Carrington_coord = coord.transform_to(sunpy.coordinates.HeliographicCarrington(observer="self"))
coords['lon_carr'] = Carrington_coord.lon.value * np.pi/180

cr, cr_lon_init = Hin.datetime2huxtinputs(run_start)
# Use the HUXt ephemeris data to get Earth lat over the CR
# ========================================================
dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=simtime, dt_scale=dt_scale, cr_num=cr,
                    cr_lon_init=cr_lon_init, lon_out=0.0*u.deg, r_min=rmin, r_max=rmax)
# retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
# get average Earth lat
E_lat = np.nanmean(earth.lat_c)
solo_lat = np.nanmean(coords['lat_heeq'])*u.rad

# <codecell> Custom plot routine


def insert_line_break_at_hyphen(s, max_length):
    if len(s) <= max_length:
        return s
    break_point = s.rfind('-', 0, max_length)
    if break_point == -1:
        return s  # No hyphen found; return unchanged
    return s[:break_point] + '\n' + s[break_point:]


def custom_plot(modela_earth, modelb_earth, modela_solo, modelb_solo, modela_str, modelb_str,
                xx, yy, plot_days):
    ts_earth_a = HA.get_observer_timeseries(modela_earth, observer='Earth')
    ts_earth_b = HA.get_observer_timeseries(modelb_earth, observer='Earth')
    ts_solo_a = HA.get_HUXt_at_position_HEEQ(modela_solo, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
    ts_solo_b = HA.get_HUXt_at_position_HEEQ(modelb_solo, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
    
    # get the nearest timesteps
    t1 = np.argmin(abs(coords['mjd'] - (modela_earth.time_init + datetime.timedelta(days = plot_days[0])).mjd))
    t2 = np.argmin(abs(coords['mjd'] - (modela_earth.time_init + datetime.timedelta(days = plot_days[1])).mjd))
    t3 = np.argmin(abs(coords['mjd'] - (modela_earth.time_init + datetime.timedelta(days = plot_days[2])).mjd))
    
    
    fig = plt.figure( figsize=(10, 12))
    gs = gridspec.GridSpec(4, 3, hspace=0.1, wspace=0.1)
    
    ax1a = fig.add_subplot(gs[0, 0], projection='polar')         
    ax2a = fig.add_subplot(gs[0, 1], projection='polar')         
    ax3a = fig.add_subplot(gs[0, 2], projection='polar')  
    
    ax1b = fig.add_subplot(gs[1, 0], projection='polar')         
    ax2b = fig.add_subplot(gs[1, 1], projection='polar')         
    ax3b = fig.add_subplot(gs[1, 2], projection='polar') 
    
    ax4 = fig.add_subplot(gs[2, :])  
    ax5 = fig.add_subplot(gs[3, :])  
     
    
    #plot the ecliptic view, with Earth and Orbiter added
    HA.plot(modela_earth, plot_days[0]*u.day, fighandle = fig, axhandle= ax1a, minimalplot=True)
    ax1a.plot(0, 215, 'wo', markeredgecolor = 'k')
    ax1a.plot(coords.loc[t1,'lon_heeq'], coords.loc[t1,'r_AU']*215, 'yo', markeredgecolor = 'k')
    ax1a.set_title((run_start + datetime.timedelta(days=plot_days[0])).strftime("%Y-%m-%d"), fontsize = 14)
    ax1a.text(0.01, 0.98, "(a)",  transform=ax1a.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    HA.plot(modela_earth, plot_days[1]*u.day, fighandle = fig, axhandle= ax2a, minimalplot=True)
    ax2a.plot(0, 215, 'wo', markeredgecolor = 'k', label = r'Earth/OMNI $<\theta>$ = ' + format(E_lat.to(u.deg).value, ".1f") + r'$^\circ$')
    ax2a.plot(coords.loc[t2,'lon_heeq'], coords.loc[t2,'r_AU']*215, 'yo', markeredgecolor = 'k', label = r'Solar Orbiter $<\theta>$ = ' + format(solo_lat.to(u.deg).value, ".1f") + r'$^\circ$')
    ax2a.set_title( (run_start + datetime.timedelta(days=plot_days[1])).strftime("%Y-%m-%d"), fontsize = 14)
    ax2a.text(0.01, 0.98, "(b)",  transform=ax2a.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    HA.plot(modela_earth, plot_days[2]*u.day, fighandle = fig, axhandle= ax3a, minimalplot=True)
    ax3a.plot(0, 215, 'wo', markeredgecolor = 'k')
    ax3a.plot(coords.loc[t3,'lon_heeq'], coords.loc[t3,'r_AU']*215, 'yo', markeredgecolor = 'k')
    ax3a.set_title((run_start + datetime.timedelta(days=plot_days[2])).strftime("%Y-%m-%d"), fontsize = 14)
    ax3a.text(0.01, 0.98, "(c)",  transform=ax3a.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    ax2a.legend(bbox_to_anchor=(2.45, 1.45), ncol=2, fontsize=14, frameon=False)

    #add the model label
    thisstr = insert_line_break_at_hyphen(modela_str, 19)
    ax1a.text(-0.1, 0.5, thisstr, transform=ax1a.transAxes, multialignment='center',
            rotation=90, va='center', ha='right', fontsize=14)
    
    
    HA.plot(modelb_earth, plot_days[0]*u.day, fighandle = fig, axhandle= ax1b, minimalplot=True)
    ax1b.plot(0, 215, 'wo', markeredgecolor = 'k')
    ax1b.plot(coords.loc[t1,'lon_heeq'], coords.loc[t1,'r_AU']*215, 'yo', markeredgecolor = 'k')
    ax1b.text(0.01, 0.98, "(d)",  transform=ax1b.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    HA.plot(modelb_earth, plot_days[1]*u.day, fighandle = fig, axhandle= ax2b, minimalplot=True)
    ax2b.plot(0, 215, 'wo', markeredgecolor = 'k')
    ax2b.plot(coords.loc[t2,'lon_heeq'], coords.loc[t2,'r_AU']*215, 'yo', markeredgecolor = 'k')
    ax2b.text(0.01, 0.98, "(e)",  transform=ax2b.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    HA.plot(modelb_earth, plot_days[2]*u.day, fighandle = fig, axhandle= ax3b, minimalplot=True)
    ax3b.plot(0, 215, 'wo', markeredgecolor = 'k')
    ax3b.plot(coords.loc[t3,'lon_heeq'], coords.loc[t3,'r_AU']*215, 'yo', markeredgecolor = 'k')
    ax3b.text(0.01, 0.98, "(f)",  transform=ax3b.transAxes,  ha='left', va='top', fontsize = 14,) 
    
    thisstr = insert_line_break_at_hyphen(modelb_str, 18)
    ax1b.text(-0.1, 0.5, thisstr, transform=ax1b.transAxes, multialignment='center',
            rotation=90, va='center', ha='right', fontsize=14)
           
    #timeseries
    #===========
    
    #compute the MAE
    smjd = Time(run_start).mjd;  fmjd = Time(run_stop).mjd; 
    mask = ((omni['mjd'] >= smjd) & (omni['mjd'] <= fmjd))
    huxt_a = np.interp(omni.loc[mask,'mjd'], ts_earth_a['mjd'], ts_earth_a['vsw'])
    mae_a = np.nanmean(abs(huxt_a - omni.loc[mask,'V']))
    huxt_b = np.interp(omni.loc[mask,'mjd'], ts_earth_b['mjd'], ts_earth_b['vsw'])
    mae_b = np.nanmean(abs(huxt_b - omni.loc[mask,'V']))
    
    
    #plot Earth TS
    axs = ax4
    xx=(run_start, run_stop)
    axs.set_xlim(xx)
    axs.set_ylim(yy)
    count = 0
    #add in ICMEs
    axs.plot(omni['datetime'], omni['V'], 'k', label = 'Observed')
    for i in range(0, len(icmes)):
        icme_start = icmes['Shock_time'][i] 
        icme_stop = icmes['ICME_end'][i] 
        if icme_start < xx[1] and icme_stop > xx[0]:
            if count == 0:
                axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
                count = count + 1
            else:
                axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
    axs.plot(omni['datetime'], omni['V'], 'k')
    axs.plot(ts_earth_a['time'], ts_earth_a['vsw'], 'r', label = modela_str)
    axs.plot(ts_earth_b['time'], ts_earth_b['vsw'], 'b--', label = modelb_str)
    axs.set_ylabel('V [km/s]')
    #axs.legend(ncol=4, fontsize=14)
    # Set x-axis major ticks every 5 days
    axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
    axs.set_xticklabels([])
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.text(0.02, 0.96, "(g) Earth",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
    axs.text(0.06, 0.84, "MAE = " + format(mae_a, ".1f"), color = 'r' ,
             transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
    axs.text(0.06, 0.73, "MAE = " + format(mae_b, ".1f"), color = 'b' ,
             transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
    
    axs.legend(loc = 'upper right', bbox_to_anchor=(1.011, 1.046), ncol=1, fontsize=14)
    
    
    #add a colourbar
    cax = fig.add_axes([0.90, 0.53, 0.02, 0.32])  # Adjust as needed
    # Define normalization and colormap
    norm = mcolors.Normalize(vmin=200, vmax=810)
    cmap = cm.viridis  # or any other colormap
    # Create a ScalarMappable and add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Needed to avoid warnings
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=14) 
    fig.text(0.90, 0.87, 'V [km/s]', va='center', rotation='horizontal',
             fontsize =14)
    
    
    #Plot Orbiter TS
    v_df_hourly['mjd'] = Time(v_df_hourly.index).mjd
    mask = ((v_df_hourly['mjd'] >= smjd) & (v_df_hourly['mjd'] <= fmjd))
    huxt_a = np.interp(v_df_hourly.loc[mask,'mjd'], ts_solo_a['mjd'], ts_solo_a['vsw'])
    mae_a = np.nanmean(abs(huxt_a - v_df_hourly.loc[mask,'velocity_0']))
    huxt_b = np.interp(v_df_hourly.loc[mask,'mjd'], ts_solo_b['mjd'], ts_solo_b['vsw'])
    mae_b = np.nanmean(abs(huxt_b - v_df_hourly.loc[mask,'velocity_0']))
    
    axs = ax5
    xx=(run_start, run_stop)
    axs.set_xlim(xx)
    axs.set_ylim(yy)
    count = 0
    axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label = 'SolO')
    #add in ICMEs
    for i in range(0, len(icmes_solo)):
        icme_start = icmes_solo['Shock_time'][i] 
        icme_stop = icmes_solo['ICME_end'][i] 
        if icme_start < xx[1] and icme_stop > xx[0]:
            if count == 0:
                axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
                count = count + 1
            else:
                axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
    axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k')
    axs.plot(ts_solo_a['time'], ts_solo_a['vsw'], 'r', label = modela_str)
    axs.plot(ts_solo_b['time'], ts_solo_b['vsw'], 'b--', label = modelb_str)
    axs.set_ylabel('V [km/s]')
    #axs.legend(ncol =4, fontsize=14)
    # Set x-axis major ticks every 5 days
    axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    #fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
    axs.set_xlabel(r'Date (2024)')
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.text(0.02, 0.96, "(h) Solar Orbiter",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
    axs.text(0.06, 0.84, "MAE = " + format(mae_a, ".1f"), color = 'r' ,
             transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
    axs.text(0.06, 0.73, "MAE = " + format(mae_b, ".1f"), color = 'b' ,
             transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 
   
    
    return fig, [ax1a, ax2a, ax3a, ax1b, ax2b, ax3b, ax4, ax5]
# <codecell> plot the case for removing ICMEs
##########################################################
#Turn off acceleration speed cap for these runs and figure
##########################################################
#do runs without CNN and with ICMEs, for demo purposes
# model_both_nocmes_noCNN = Hin.set_time_dependent_boundary(vcarr_rmin_both, time1au_both, 
#                             run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
#                             dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
#                             track_cmes = False)
# model_both_nocmes_noCNN.solve([])  
# model_both_nocmes_noCNN_icmes = Hin.set_time_dependent_boundary(vcarr_rmin_both_icmes, time1au_both, 
#                             run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
#                             dt_scale=dt_scale, latitude=0*u.deg, frame = 'synodic', 
#                             track_cmes = False)
# model_both_nocmes_noCNN_icmes.solve([])  


# #====================================
# xx = (omni_noicmes.loc[0,'datetime'], omni_noicmes.loc[len(omni_noicmes) - 1,'datetime'])


# modela_earth = model_both_nocmes_noCNN_icmes; modela_str = 'OMNI(inc ICMEs)-HUXt'
# modelb_earth = model_both_nocmes_noCNN; modelb_str = 'OMNI(no ICMEs)-HUXt'
# modela_solo = model_both_nocmes_noCNN_icmes; 
# modelb_solo = model_both_nocmes_noCNN; 
# #====================================

# fig, axs = custom_plot(modela_earth, modelb_earth, modela_solo, modelb_solo, modela_str, modelb_str,
#                        xx, yy, plot_days)
# fig.savefig(os.path.join(figdir, 'May2024_ICMEs_noICMEs-HUXt.pdf'))





# <codecell> plot the OMNI-HUXt run
#====================================
xx = (omni_noicmes.loc[0,'datetime'], omni_noicmes.loc[len(omni_noicmes) - 1,'datetime'])


modela_earth = model_both_nocmes; modela_str = 'OMNI-HUXt'
modelb_earth = model_both_cmes; modelb_str = 'OMNI-Cone-HUXt'
modela_solo = model_both_nocmes; 
modelb_solo = model_both_cmes; 
#====================================


fig, axs = custom_plot(modela_earth, modelb_earth, modela_solo, modelb_solo, modela_str, modelb_str,
                       xx, yy, plot_days)
fig.savefig(os.path.join(figdir, 'May2024_OMNI-HUXt.pdf'))


# <codecell> plot simple summary time series



#extract all the data from the various model runs

ts_both_earth_cmes = HA.get_observer_timeseries(model_both_cmes, observer='Earth')
ts_both_earth_nocmes = HA.get_observer_timeseries(model_both_nocmes, observer='Earth')
ts_back_earth_cmes = HA.get_observer_timeseries(model_back_cmes, observer='Earth')
ts_back_earth_nocmes = HA.get_observer_timeseries(model_back_nocmes, observer='Earth')
ts_forward_earth_cmes = HA.get_observer_timeseries(model_forward_cmes, observer='Earth')
ts_forward_earth_nocmes = HA.get_observer_timeseries(model_forward_nocmes, observer='Earth')
ts_dtw_earth_cmes = HA.get_observer_timeseries(model_dtw_cmes, observer='Earth')
ts_dtw_earth_nocmes = HA.get_observer_timeseries(model_dtw_nocmes, observer='Earth')

ts_both_solo_nocmes = HA.get_HUXt_at_position_HEEQ(model_both_nocmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_both_solo_cmes = HA.get_HUXt_at_position_HEEQ(model_both_cmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_back_solo_nocmes = HA.get_HUXt_at_position_HEEQ(model_back_nocmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_back_solo_cmes = HA.get_HUXt_at_position_HEEQ(model_back_cmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_forward_solo_nocmes = HA.get_HUXt_at_position_HEEQ(model_forward_nocmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_forward_solo_cmes = HA.get_HUXt_at_position_HEEQ(model_forward_cmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_dtw_solo_nocmes = HA.get_HUXt_at_position_HEEQ(model_dtw_nocmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])
ts_dtw_solo_cmes = HA.get_HUXt_at_position_HEEQ(model_dtw_cmes, coords['mjd'], coords['r_AU']*215, coords['lon_heeq'])


fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axs = axes[0]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes)):
    icme_start = icmes['Shock_time'][i] 
    icme_stop = icmes['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
axs.plot(omni['datetime'], omni['V'], 'k', label = 'OMNI')
axs.plot(ts_back_earth_nocmes['time'], ts_back_earth_nocmes['vsw'], 'r--', label = 'OMNI-HuXt (back)')
axs.plot(ts_forward_earth_nocmes['time'], ts_forward_earth_nocmes['vsw'], 'b--', label = 'OMNI-HUXt (forward)')
axs.plot(ts_both_earth_nocmes['time'], ts_both_earth_nocmes['vsw'], 'b', label = 'OMNI-HUXt (both)')
axs.plot(ts_dtw_earth_nocmes['time'], ts_dtw_earth_nocmes['vsw'], 'r', label = 'OMNI-HUXt (dtw)')

axs.set_ylabel(r'$V_{SW}$ (km/s)')
axs.legend(ncol=3)
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
axs.set_xticklabels([])


# axs = axes[1]
# xx=(run_start, run_stop)
# axs.set_xlim(xx)
# axs.set_ylim(yy)
# count = 0
# #add in ICMEs
# for i in range(0, len(icmes)):
#     icme_start = icmes['Shock_time'][i] 
#     icme_stop = icmes['ICME_end'][i] 
#     if icme_start < xx[1] and icme_stop > xx[0]:
#         if count == 0:
#             axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
#             count = count + 1
#         else:
#             axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
# axs.plot(omni['datetime'], omni['V'], 'k', label = 'OMNI')
# axs.plot(ts_back_earth_cmes['time'], ts_back_earth_cmes['vsw'], 'r--', label = source_str+'-Cone-HUXt (back)')
# axs.plot(ts_forward_earth_cmes['time'], ts_forward_earth_cmes['vsw'], 'b--', label = source_str+'-Cone-HUXt (forward)')
# axs.plot(ts_both_earth_cmes['time'], ts_both_earth_cmes['vsw'], 'b', label = source_str+'-Cone-HUXt (both)')
# axs.plot(ts_dtw_earth_cmes['time'], ts_dtw_earth_cmes['vsw'], 'r', label = source_str+'-Cone-HUXt (dtw)')

# axs.set_ylabel(r'$V_{SW}$ (km/s)')
# axs.legend(ncol=3)
# # Set x-axis major ticks every 5 days
# axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
# axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# #fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
# axs.set_xticklabels([])


axs = axes[1]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes_solo)):
    icme_start = icmes_solo['Shock_time'][i] 
    icme_stop = icmes_solo['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label = 'SolO')
axs.plot(ts_back_solo_nocmes['time'], ts_back_solo_nocmes['vsw'], 'r--', label = 'OMNI-HUXt (back)')
axs.plot(ts_forward_solo_nocmes['time'], ts_forward_solo_nocmes['vsw'], 'b--', label = 'OMNI-HUXt (forward)')
axs.plot(ts_both_solo_nocmes['time'], ts_both_solo_nocmes['vsw'], 'b', label = 'OMNI-HUXt (both)')
axs.plot(ts_dtw_solo_nocmes['time'], ts_dtw_solo_nocmes['vsw'], 'r', label = 'OMNI-HUXt (dtw)')

axs.set_ylabel(r'$V_{SW}$ (km/s)')
axs.legend(ncol=3)
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
#axs.set_xticklabels([])

axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
axs.set_xlabel(r'Date (2024)')

# axs = axes[3]
# xx=(run_start, run_stop)
# axs.set_xlim(xx)
# axs.set_ylim(yy)
# count = 0
# #add in ICMEs
# for i in range(0, len(icmes_solo)):
#     icme_start = icmes_solo['Shock_time'][i] 
#     icme_stop = icmes_solo['ICME_end'][i] 
#     if icme_start < xx[1] and icme_stop > xx[0]:
#         if count == 0:
#             axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1, label = 'ICME')   
#             count = count + 1
#         else:
#             axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='lightgrey', alpha=1)    
# axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label = 'SolO')
# axs.plot(ts_back_solo_cmes['time'], ts_back_solo_cmes['vsw'], 'r--', label = source_str+'-Cone-HUXt (back)')
# axs.plot(ts_forward_solo_cmes['time'], ts_forward_solo_cmes['vsw'], 'b--', label = source_str+'-Cone-HUXt (forward)')
# axs.plot(ts_both_solo_cmes['time'], ts_both_solo_cmes['vsw'], 'b', label = source_str+'-Cone-HUXt (both)')
# axs.plot(ts_dtw_solo_cmes['time'], ts_dtw_solo_cmes['vsw'], 'r', label = source_str+'-Cone-HUXt (dtw)')

# axs.set_ylabel(r'$V_{SW}$ (km/s)')
# axs.legend(ncol=3)
# # Set x-axis major ticks every 5 days
# axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
# axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# #fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
# axs.set_xticklabels([])


# <codecell> Expanded "ensemble" plots


fig, axes = plt.subplots(4, 1, figsize=(10, 12))


#axis 1
#================================================================================
axs = axes[0]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes)):
    icme_start = icmes['Shock_time'][i] 
    icme_stop = icmes['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1)    
axs.plot(omni['datetime'], omni['V'], 'k', label = 'Obs')
axs.plot(ts_back_earth_nocmes['time'], ts_back_earth_nocmes['vsw'], 'r', label = 'Corot (back)')
axs.plot(ts_forward_earth_nocmes['time'], ts_forward_earth_nocmes['vsw'], 'b', label = 'Corot (forward)')
axs.plot(ts_both_earth_nocmes['time'], ts_both_earth_nocmes['vsw'], 'w', label = 'Corot (both)')
axs.plot(ts_dtw_earth_nocmes['time'], ts_dtw_earth_nocmes['vsw'], 'g', label = 'DTW')
axs.plot(ts_back_earth_cmes['time'], ts_back_earth_cmes['vsw'],'r--', )
axs.plot(ts_forward_earth_cmes['time'], ts_forward_earth_cmes['vsw'],'b--')
axs.plot(ts_both_earth_cmes['time'], ts_both_earth_cmes['vsw'],'w--')
axs.plot(ts_dtw_earth_cmes['time'], ts_dtw_earth_cmes['vsw'],'g--')

axs.set_ylabel(r'$V_{SW}$ (km/s)')
axs.legend(bbox_to_anchor=(0.37, 1.50), ncol=3, fontsize=14, frameon=True, 
           facecolor = (0.83, 0.83, 0.83), loc = 'upper center')
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
axs.set_xticklabels([])
axs.set_facecolor((0.83, 0.83, 0.83)) 
axs.set_yticks([ 400, 600, 800, 1000])
axs.text(0.02, 0.96, "(a) Earth",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 

#axis 3
#================================================================================

#create a mini-ensemble
L =  len(ts_both_earth_cmes)
cme_median = np.ones(L)
nocme_median = np.ones(L)
min_all =  np.ones(L)
max_all =  np.ones(L)
for t in range(0,L):

    nocme_data = [     ts_back_earth_nocmes.loc[t,'vsw'],
                    ts_forward_earth_nocmes.loc[t,'vsw'],
                    ts_both_earth_nocmes.loc[t,'vsw'],
                    ts_dtw_earth_nocmes.loc[t,'vsw'] ]
    cme_data = [ ts_back_earth_cmes.loc[t,'vsw'],
                 ts_forward_earth_cmes.loc[t,'vsw'],
                 ts_both_earth_cmes.loc[t,'vsw'],
                 ts_dtw_earth_cmes.loc[t,'vsw']   ]
    
    all_data = [nocme_data, cme_data]
    
    cme_median[t] = np.median(cme_data)
    nocme_median[t] = np.median(nocme_data)
    min_all[t] = np.min(all_data)
    max_all[t] = np.max(all_data)


axs = axes[1]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes)):
    icme_start = icmes['Shock_time'][i] 
    icme_stop = icmes['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1)   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1)    
axs.fill_between(ts_back_earth_nocmes['time'], min_all, max_all, color='red', alpha=0.3, label = 'All (range)')   
axs.plot(omni['datetime'], omni['V'], 'k')
axs.plot(ts_back_earth_nocmes['time'], nocme_median, 'r', label = 'Ambient (median)')
axs.plot(ts_back_earth_nocmes['time'], cme_median, 'r--', label = 'CME (median)')

axs.set_ylabel(r'$V_{SW}$ (km/s)')
axs.legend(loc = 'upper right', bbox_to_anchor=(1.011, 1.046), ncol=1, fontsize=14)
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
axs.set_xticklabels([])
axs.set_facecolor((0.83, 0.83, 0.83)) 
axs.set_yticks([ 400, 600, 800, 1000])
axs.text(0.02, 0.96, "(b) Earth",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 


#axis 3
#================================================================================


axs = axes[2]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes_solo)):
    icme_start = icmes_solo['Shock_time'][i] 
    icme_stop = icmes_solo['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1)    
axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label = 'SolO')
axs.plot(ts_back_solo_nocmes['time'], ts_back_solo_nocmes['vsw'], 'r', label = 'Corot (back)')
axs.plot(ts_forward_solo_nocmes['time'], ts_forward_solo_nocmes['vsw'], 'b', label = 'Corot (forward)')
axs.plot(ts_both_solo_nocmes['time'], ts_both_solo_nocmes['vsw'], 'w', label = 'Corot (both)')
axs.plot(ts_dtw_solo_nocmes['time'], ts_dtw_solo_nocmes['vsw'], 'g', label = 'DTW')
axs.plot(ts_back_solo_cmes['time'], ts_back_solo_cmes['vsw'],'r--', )
axs.plot(ts_forward_solo_cmes['time'], ts_forward_solo_cmes['vsw'],'b--')
axs.plot(ts_both_solo_cmes['time'], ts_both_solo_cmes['vsw'],'w--')
axs.plot(ts_dtw_solo_cmes['time'], ts_dtw_solo_cmes['vsw'],'g--')

axs.set_ylabel(r'$V_{SW}$ (km/s)')
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
axs.set_xticklabels([])
axs.set_facecolor((0.83, 0.83, 0.83)) 
axs.set_yticks([400, 600, 800, 1000])
axs.text(0.02, 0.96, "(c) Solar Orbiter",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 

#axis 4
#================================================================================

#create a mini-ensemble
L =  len(ts_both_solo_cmes)
cme_median = np.ones(L)
nocme_median = np.ones(L)
min_all =  np.ones(L)
max_all =  np.ones(L)
for t in range(0,L):
    cme_data = [ ts_back_solo_cmes.loc[t,'vsw'],
                 ts_forward_solo_cmes.loc[t,'vsw'],
                 ts_both_solo_cmes.loc[t,'vsw'],
                 ts_dtw_solo_cmes.loc[t,'vsw']   ]
    nocme_data = [ts_back_solo_nocmes.loc[t,'vsw'],
                ts_forward_solo_nocmes.loc[t,'vsw'],
                ts_both_solo_nocmes.loc[t,'vsw'],
                ts_dtw_solo_nocmes.loc[t,'vsw'] ]
    
    all_data = [cme_data, nocme_data ]
    
    
    cme_median[t] = np.median(cme_data)
    min_all[t] = np.min(all_data)
    max_all[t] = np.max(all_data)
    nocme_median[t] = np.median(nocme_data)


axs = axes[3]
xx=(run_start, run_stop)
axs.set_xlim(xx)
axs.set_ylim(yy)
count = 0
#add in ICMEs
for i in range(0, len(icmes_solo)):
    icme_start = icmes_solo['Shock_time'][i] 
    icme_stop = icmes_solo['ICME_end'][i] 
    if icme_start < xx[1] and icme_stop > xx[0]:
        if count == 0:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1, label = 'ICME')   
            count = count + 1
        else:
            axs.fill_between((icme_start, icme_stop), yy[0], yy[1], color='grey', alpha=1)    
axs.plot(v_df_hourly.index, v_df_hourly['velocity_0'], 'k', label = 'SolO')
axs.fill_between(ts_back_solo_nocmes['time'], min_all, max_all, color='red', alpha=0.3)  
axs.plot(ts_back_solo_nocmes['time'], nocme_median, 'r', label = 'Ambient (median)') 
axs.plot(ts_back_solo_nocmes['time'], cme_median, 'r--', label = 'CME (median)')
axs.set_ylabel(r'$V_{SW}$ (km/s)')
# Set x-axis major ticks every 5 days
axs.xaxis.set_major_locator(mdates.DayLocator(interval=5))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.97, hspace=0.05)
#axs.set_xticklabels([])
axs.set_yticks([400, 600, 800, 1000])
axs.set_facecolor((0.83, 0.83, 0.83)) 
axs.set_xlabel(r'Date [2024]')
axs.text(0.02, 0.96, "(d) Solar Orbiter",  transform=axs.transAxes,  ha='left', va='top', fontsize = 14) 



fig.savefig(os.path.join(figdir, 'May2024_ensemble.pdf'))

# <codecell> Do a wsa-HUXt run, for comparison

# ==============================================================================
# get Met Office data for this date - assumes API is an env vairable.
# ==============================================================================
sdate = icme_time - datetime.timedelta(days=2)
fdate = icme_time

if download_now: 
    success, wsafilepath, conefilepath, model_time = Hin.getMetOfficeWSAandCone(sdate, fdate)
else:
    wsafilepath = 'models%2Fenlil%2F2024%2F5%2F11%2F0%2Fwsa.gong.fits'
    conefilepath = 'models%2Fenlil%2F2024%2F5%2F11%2F0%2Fcone2bc.in'
      


# compute the HUXt run start date, to allow for CMEs before the forecast date
#starttime = forecasttime - datetime.timedelta(days=run_buffer_time.to(u.day).value) 
    


    

if os.path.exists(wsafilepath):
    vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
        = Hin.get_WSA_maps(wsafilepath)
            
    # get the WSA values at Earth lat.
    v_in = Hin.get_WSA_long_profile(wsafilepath, lat=E_lat)
    v_in_solo = Hin.get_WSA_long_profile(wsafilepath, lat=solo_lat)

    # deaccelerate them
    v_in, lon_temp = Hin.map_v_inwards(v_in, 215*u.solRad, vr_longs, rmin)
    v_in_solo, lon_temp = Hin.map_v_inwards(v_in_solo, 215*u.solRad, vr_longs, rmin)
    
model_wsa_nocmes = H.HUXt(v_boundary=v_in, simtime=simtime,
               latitude=np.mean(E_lat), cr_lon_init=cr_lon_init,
               dt_scale=dt_scale, cr_num=cr,
               r_min=rmin, r_max=rmax, frame = 'synodic', 
               track_cmes= False)

model_wsa_nocmes.solve([])

model_wsa_nocmes_solo = H.HUXt(v_boundary=v_in_solo, simtime=simtime,
               latitude=np.mean(E_lat), cr_lon_init=cr_lon_init,
               dt_scale=dt_scale, cr_num=cr,
               r_min=rmin, r_max=rmax, frame = 'synodic', 
               track_cmes= False)

model_wsa_nocmes_solo.solve([])


#run with the DONKI coneCMEs

#get the DONKI coneCMEs
cme_list = Hin.get_DONKI_cme_list(model_wsa_nocmes , 
                                  run_start - datetime.timedelta(days = 5), run_stop)
#change each CME to fixed duration
for cme in cme_list:
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s



model_wsa_cmes = H.HUXt(v_boundary=v_in, simtime=simtime,
               latitude=np.mean(E_lat), cr_lon_init=cr_lon_init,
               dt_scale=dt_scale, cr_num=cr,
               r_min=rmin, r_max=rmax, frame = 'synodic', 
               track_cmes= False)

model_wsa_cmes.solve(cme_list)

model_wsa_cmes_solo = H.HUXt(v_boundary=v_in_solo, simtime=simtime,
               latitude=np.mean(E_lat), cr_lon_init=cr_lon_init,
               dt_scale=dt_scale, cr_num=cr,
               r_min=rmin, r_max=rmax, frame = 'synodic', 
               track_cmes= False)

model_wsa_cmes_solo.solve(cme_list)

# <codecell> plot the WSA results


#====================================
modela_earth = model_wsa_nocmes; modela_str = 'WSA-HUXt'
modelb_earth = model_wsa_cmes; modelb_str = 'WSA-Cone-HUXt'
modela_solo = model_wsa_nocmes_solo; 
modelb_solo = model_wsa_cmes_solo; 
#====================================



fig, axs = custom_plot(modela_earth, modelb_earth, modela_solo, modelb_solo, modela_str, modelb_str,
                       xx, yy, plot_days)
fig.savefig(os.path.join(figdir, 'May2024_WSA-HUXt.pdf'))