# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:38:44 2025

a script to create WSA/HUXt forecasts at Earth, STA and STB over a long time span
"""




import datetime
import numpy as np
import os
import glob
import astropy.units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA

wsa_ccmc_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC')
output_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC', 'CCMC_WSA_HUXt')
#wsa_ccmc_dir = "/data/CCMC_WSA/"
#output_dir = '~/CCMC_WSA_HUXt/'


before_days = 27.27
after_days = 27.3
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 240*u.solRad
plot_now = False
solve_sta = False
solve_stb = False


start_dt = datetime.datetime(2008,1,1)
stop_dt = datetime.datetime(2023,1,1)


def order_min_angular_distance(start, stop):
    # Normalize angles to [0, 2π)
    start = start % (2 * np.pi)
    stop = stop % (2 * np.pi)
    
    # Compute counter-clockwise angular distance
    d_ccw = (stop - start) % (2 * np.pi)
    d_cw = (start - stop) % (2 * np.pi)

    # Return the order that gives the smaller angular distance
    if d_ccw <= d_cw:
        return start, stop  # CCW is shorter
    else:
        return stop, start  # CW is shorter (flip)


def process_date(dt):
    """
    Process a single forecast date
    """
    print('Looking for: ' + dt.strftime('%Y-%m-%d'))
    
    
    #get the forecast cr and cr_lon_init
    cr_forecast, cr_lon_init_forecast = Hin.datetime2huxtinputs(dt)
    
    #set up the directory path
    yr_str = str(dt.year)
    mn_str = f"{dt.month:02}"
    dy_str = f"{dt.day:02}"
    
    filename_wildcard = 'vel_' + yr_str + mn_str + dy_str + '*R000_gong*z.fits'
    filepath_wildcard = os.path.join(wsa_ccmc_dir, yr_str, filename_wildcard)
    
    # Find matching files
    matching_files = glob.glob(filepath_wildcard)
    
    #if the file exists, read in the sub-Earth values
    if matching_files:
        print('Loading: ' + dt.strftime('%Y-%m-%d'))
        filepath = matching_files[0]
        
        #find the start time and simtime
        runstart = dt - datetime.timedelta(days = before_days)
        simtime_days = after_days + before_days
        
        #get the start CR num and long
        cr_runstart, cr_lon_init_runstart = Hin.datetime2huxtinputs(runstart)
        
        #get Earth lat
        #Elat = Hin.get_earth_lat(dt)
        dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day, 
                             cr_num=cr_forecast, cr_lon_init=cr_lon_init_forecast)
        # retrieve a bodies position at each model timestep:
        earth = dummymodel.get_observer('EARTH')
        # get average Earth lat
        Elat = np.nanmean(earth.lat_c)
        
        #get the v and B data
        v_in = Hin.get_WSA_long_profile(filepath, lat = Elat)
        b_in = Hin.get_WSA_br_long_profile(filepath, lat = Elat)

        #deaccelerate the WSA speeds
        nlons = len(v_in)
        dlon = 2*np.pi/nlons
        vr_longs = np.arange(dlon/2, 2*np.pi, dlon)*u.rad
        v_in, lon_temp = Hin.map_v_inwards(v_in, 215*u.solRad, vr_longs, rmin)
        
        #set up the model
        model = H.HUXt(v_boundary=v_in, b_boundary=b_in, 
                       simtime=simtime_days*u.day,
                       latitude=Elat, cr_lon_init=cr_lon_init_runstart,
                       dt_scale=dt_scale, cr_num=cr_runstart,
                       r_min=rmin, r_max=rmax, frame = 'synodic',
                       lon_out = 0.0*u.rad) 
        model.solve([])
    
        
        #extract the timeseries at Earth
        ts = HA.get_observer_timeseries(model, observer='Earth')
        #add a forecast lead time column
        ts['leadtime'] = (ts['time'] - dt)/np.timedelta64(1, 'D')
        if plot_now: 
            HA.plot_earth_timeseries(model)    
        
        #save the data
        dir_path = os.path.join(output_dir, yr_str)
        #create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        outname = 'HUXt_Earth_' + yr_str +mn_str + dy_str + '.dat'
        ts.to_csv(os.path.join(dir_path,outname))
        
    
    
        #repeat for STA
        #===============
        
        if solve_sta and dt.year >=2007:
            # Use the HUXt ephemeris data to get STA lat over the CR
            # ========================================================
            dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day, 
                                 cr_num=cr_forecast, cr_lon_init=cr_lon_init_forecast, lon_out=0.0*u.rad)
            # retrieve a bodies position at each model timestep:
            sta = dummymodel.get_observer('STA')
            # get average Earth lat
            STA_lat = np.nanmean(sta.lat_c)
            
            #get the required lon range too
            dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=simtime_days*u.day, 
                                 cr_num=cr_runstart, cr_lon_init=cr_lon_init_runstart)
            dlon = dummymodel.dlon.value
            sta = dummymodel.get_observer('STA')
            lon_start, lon_stop = order_min_angular_distance((sta.lon).max().value, 
                                                             (sta.lon).min().value)
            STA_lon_start = H._zerototwopi_(lon_start - 2*dlon) * u.rad
            STA_lon_stop = H._zerototwopi_(lon_stop + 2*dlon) *u.rad
            
            #get the v and B data
            v_in = Hin.get_WSA_long_profile(filepath, lat = STA_lat)
            b_in = Hin.get_WSA_br_long_profile(filepath, lat = STA_lat)

            #deaccelerate the WSA speeds
            nlons = len(v_in)
            dlon = 2*np.pi/nlons
            vr_longs = np.arange(dlon/2, 2*np.pi, dlon)*u.rad
            v_in, lon_temp = Hin.map_v_inwards(v_in, 215*u.solRad, vr_longs, rmin)
            
            #set up the model
            model = H.HUXt(v_boundary=v_in, b_boundary=b_in, 
                           simtime=simtime_days*u.day,
                           latitude=STA_lat, cr_lon_init=cr_lon_init_runstart,
                           dt_scale=dt_scale, cr_num=cr_runstart,
                           lon_start= STA_lon_start, lon_stop = STA_lon_stop,
                           r_min=rmin, r_max=rmax, frame = 'synodic') 
            model.solve([])
    
            
            #extract the timeseries at Earth
            ts = HA.get_observer_timeseries(model, observer='STA')
            #add a forecast lead time column
            ts['leadtime'] = (ts['time'] - dt)/np.timedelta64(1, 'D')
            if plot_now:
                plt.figure()
                plt.plot(ts['time'], ts['vsw'])
     
            
            #save the data
            dir_path = os.path.join(output_dir, yr_str)
            #create directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            outname = 'HUXt_STA_' + yr_str +mn_str + dy_str + '.dat'
            ts.to_csv(os.path.join(dir_path,outname))
            
            
        # and STB
        #========================================
        if solve_stb and ((dt.year >=2007) & (dt<=datetime.datetime(2014,10,14))):
            # Use the HUXt ephemeris data to get STA lat over the CR
            # ========================================================
            dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day, 
                                 cr_num=cr_forecast,cr_lon_init=cr_lon_init_forecast, lon_out = 0*u.rad)
            # retrieve a bodies position at each model timestep:
            stb = dummymodel.get_observer('STB')
            # get average Earth lat
            STB_lat = np.nanmean(stb.lat_c)
            
            #get the required lon range too
            dummymodel = H.HUXt(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=simtime_days*u.day, 
                                 cr_num=cr_runstart, cr_lon_init=cr_lon_init_runstart)
            dlon = dummymodel.dlon.value
            stb = dummymodel.get_observer('STB')
            lon_start, lon_stop = order_min_angular_distance((stb.lon).max().value, 
                                                            (stb.lon).min().value)
            STB_lon_start = H._zerototwopi_(lon_start - 2*dlon) * u.rad
            STB_lon_stop = H._zerototwopi_(lon_stop + 2*dlon) *u.rad
            
            
            #get the v and B data
            v_in = Hin.get_WSA_long_profile(filepath, lat = STB_lat)
            b_in = Hin.get_WSA_br_long_profile(filepath, lat = STB_lat)
            
            #deaccelerate the WSA speeds
            nlons = len(v_in)
            dlon = 2*np.pi/nlons
            vr_longs = np.arange(dlon/2, 2*np.pi, dlon)*u.rad
            v_in, lon_temp = Hin.map_v_inwards(v_in, 215*u.solRad, vr_longs, rmin)

            #set up the model
            model = H.HUXt(v_boundary=v_in, b_boundary=b_in, 
                           simtime=simtime_days*u.day,
                           latitude=STB_lat, cr_lon_init=cr_lon_init_forecast,
                           dt_scale=dt_scale, cr_num=cr_forecast,
                           lon_start= STB_lon_start, lon_stop = STB_lon_stop,
                           r_min=rmin, r_max=rmax, frame = 'synodic') 
            model.solve([])
    
            
            #extract the timeseries at Earth
            ts = HA.get_observer_timeseries(model, observer='STB')
            
            #add a forecast lead time column
            ts['leadtime'] = (ts['time'] - dt)/np.timedelta64(1, 'D')
            if plot_now:
                plt.figure()
                plt.plot(ts['time'], ts['vsw'])    
            
            #save the data
            dir_path = os.path.join(output_dir, yr_str)
            #create directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)
            outname = 'HUXt_STB_' + yr_str +mn_str + dy_str + '.dat'
            ts.to_csv(os.path.join(dir_path,outname))
    
    return dt


if __name__ == '__main__':
    # Generate list of dates to process
    dates_to_process = []
    dt = start_dt
    while dt <= stop_dt:
        dates_to_process.append(dt)
        dt += datetime.timedelta(days=1)
    
    # Determine number of processes to use
    n_processes = cpu_count()
    print(f'Processing {len(dates_to_process)} dates using {n_processes} parallel processes')
    
    # Process dates in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_date, dates_to_process)
    
    print('All processing complete!')
