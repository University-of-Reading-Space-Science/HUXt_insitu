# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:49:10 2025

functions reuired for insitu-huxt

@author: mathe
"""


import datetime
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import sun
import pandas as pd
import re  #for dealing with non-numeric characters in a string of unknown length
import urllib.request

import joblib
import onnxruntime as ort


#HUXt libraries
import huxt as H
import huxt_inputs as Hin

import helio_coords as hcoords

# <codecell> remove ICMEs
def _setup_dirs_():
    
    cwd = os.path.abspath(os.path.dirname(__file__))
    root = os.path.dirname(cwd)
    datapath = os.path.join(root,'Data')
    
    dirs = {"datapath": datapath}
    dirs['root'] = root
    
    return dirs

def ICMElist(filepath = None):
    # -*- coding: utf-8 -*-
    """
    A script to read and process Ian Richardson's ICME list.

    Some pre-processing is required:
        Download the following webpage as a html file: 
            http://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm
        Open in Excel, remove the year rows, delete last column (S) which is empty
        Cut out the data table only (delete header and footer)
        Save as a CSV.

    """
    
    if filepath is None:
        datapath = _setup_dirs_()['datapath']
        filepath = os.path.join(datapath,
                                'Richardson_Cane_Porcessed_ICME_list.csv')
    
    
    icmes=pd.read_csv(filepath,header=None)
    #delete the first row
    icmes.drop(icmes.index[0], inplace=True)
    icmes.index = range(len(icmes))
    
    for rownum in range(0,len(icmes)):
        for colnum in range(0,3):
            #convert the three date stamps
            datestr=icmes[colnum][rownum]
            year=int(datestr[:4])
            month=int(datestr[5:7])
            day=int(datestr[8:10])
            hour=int(datestr[11:13])
            minute=int(datestr[13:15])
            #icmes.set_value(rownum,colnum,datetime(year,month, day,hour,minute,0))
            icmes.at[rownum,colnum] = datetime.datetime(year,month, day,hour,minute,0)
            
            #print(datestr)
            
        #tidy up the plasma properties
        for paramno in range(10,17):
            dv=str(icmes[paramno][rownum])
            
            #print(str(paramno)+ ' ' + dv)
            
            if dv == '...' or dv == 'dg' or dv == 'nan' or dv == '... P' or dv == '... Q':
                #icmes.set_value(rownum,paramno,np.nan)
                icmes.at[rownum,paramno] = np.nan
            else:
                #remove any remaining non-numeric characters
                dv=re.sub('[^0-9]','', dv)
                #icmes.set_value(rownum,paramno,float(dv))
                icmes.at[rownum,paramno] = float(dv)
        
    
    #chage teh column headings
    icmes=icmes.rename(columns = {0:'Shock_time',
                                  1:'ICME_start',
                                  2:'ICME_end',
                                  10:'dV',
                                  11: 'V_mean',
                                  12:'V_max',
                                  13:'Bmag',
                                  14:'MCflag',
                                  15:'Dst',
                                  16:'V_transit'})
    return icmes
    
#icmes=ICMElist()             

def STEREO_ICME_list(filepath = None, download_now = False):
    """
    a script to read in Lan Jian's STEREO  ICME list, from:
    https://stereo-ssc.nascom.nasa.gov/pub/ins_data/impact/level3/LanJian_STEREO_ICME_List.txt
    """
    
    if filepath is None:
        datapath = _setup_dirs_()['datapath']
        filepath = os.path.join(datapath,
                                'LanJian_STEREO_ICME_List.txt')
    
    if download_now:
        urllib.request.urlretrieve('https://stereo-ssc.nascom.nasa.gov/pub/ins_data/impact/level3/LanJian_STEREO_ICME_List.txt', 
                                   filepath)
    
    
    # Read the file, ignoring lines that start with #
    data = pd.read_csv(filepath, sep='\t', comment='#', header=None)
    
    # If you want to assign column names (example names given)
    data.columns = ["ICME_start", "ICME_end", "magnetic_obstacle_start_time", 
                    "STEREO", "hybrid_event", "ambiguous_event", "Ptmax_including_sheath", 
                    "Bmax_including_sheath", "Vmax_including_sheath",
                    "Ptmax_excluding_sheath", "Bmax_excluding_sheath",
                    "Vmax_excluding_sheath",
                    "speed_change", "Group", "magnetic_cloud_index",
                    "Comment"]
    
    #convert to datetime
    data['ICME_start'] = pd.to_datetime(data['ICME_start'], errors='coerce')
    data['ICME_end'] = pd.to_datetime(data['ICME_end'], errors='coerce')
    data['magnetic_obstacle_start_time'] = pd.to_datetime(data['magnetic_obstacle_start_time'], errors='coerce')
    
    # # Display rows where datetime conversion failed
    # invalid_dates = data[data[['ICME_start_time', 'ICME_end_time', 'magnetic_obstacle_start_time']].isnull().any(axis=1)]
    # if not invalid_dates.empty:
    #     print("Rows with invalid datetime entries:")
    #     print(invalid_dates)
    
    
    # Display the first few rows of the dataframe
    #print(data.head())
    
    return data

def removeICMEs(omni, 
                icme_list = 'CaneRichardson',
                pre_icme_buffer = 0.2, #days
                post_icme_buffer = 1, #days
                interp_gaps = True ):
    
    #create a copy of the OMNI data for ICME removal
    omni_noicmes = omni.copy()
    
    dl_starttime = omni.loc[0]['datetime'] - datetime.timedelta(days=27)
    dl_endtime = omni.loc[len(omni)-1]['datetime'] + datetime.timedelta(days=27)
    
    #load the DONKI ICME list
    if icme_list == 'DONKI':
        icmes = Hin.get_DONKI_ICMEs(dl_starttime, dl_endtime)
    elif icme_list == 'CaneRichardson':
        icmes = ICMElist()
    
    
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
    
    if interp_gaps:
        #now interp through all datagaps
        omni_noicmes = omni_noicmes.set_index('datetime')
        omni_noicmes[['V', 'BX_GSE']] = omni_noicmes[['V', 'BX_GSE']].interpolate(method='time').ffill().bfill()
        omni_noicmes = omni_noicmes.reset_index()

    return omni_noicmes


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


def omniHUXt_forecast(ftime, simtime = 27.27*u.day, 
                        rmin = 21.5*u.solRad, rmax = 230*u.solRad, 
                        dt_scale = 4,
                        omni_input = None, buffertime = 5*u.day):
    run_start = ftime 
    #run_stop = ftime + datetime.timedelta(days=simtime.value)
    #simtime = (run_stop-run_start).days * u.day
    
    #if no omni data provided, download it and remove ICMEs
    if omni_input is None:
        dl_starttime = ftime - datetime.timedelta(days=28)
        dl_endtime = ftime + datetime.timedelta(days=28)
    
        omni = Hin.get_omni(dl_starttime, dl_endtime)
        
        omni_input = removeICMEs(omni)
    
    #cut out the precise bit of the OMNI data that is required
    mask = (omni_input['datetime'] <= ftime) 
    omni_input = omni_input.loc[mask]
    
    
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
    vcarr_rmin_back_cnn = correct_inner_vlon_cnn_onnx(vlon.reshape(-1, 1))
    
    
    #set up the model run to start 5 days before the forecast time, to allow for CMEs
    cr, cr_lon_init = Hin.datetime2huxtinputs(ftime - datetime.timedelta(days = buffertime.value))
    Elat = Hin.get_earth_lat(ftime)
    model = H.HUXt(v_boundary = vcarr_rmin_back_cnn.flatten() * u.km/u.s, 
                              cr_num = cr, cr_lon_init=cr_lon_init,
                               simtime = simtime, r_min=rmin, r_max=rmax, 
                                dt_scale=dt_scale, latitude=Elat, frame = 'synodic', 
                                track_cmes = False, lon_out = 0*u.rad)
    return model

