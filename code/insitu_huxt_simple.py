# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:35:35 2025

@author: mathe
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:40:52 2025

@author: mathe
"""



import datetime
import os
import astropy.units as u
import glob
import re
import numpy as np
import time
import requests
import matplotlib.pyplot as plt
import pandas as pd

#HUXt libraries
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA








run_start = datetime.datetime(2020,10,1)
run_stop =  datetime.datetime(2021,1,1)


#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 240*u.solRad #outer boundary for HUXt runs

recon_noicmes = True
icme_list = 'CaneRichardson'#'DONKI'
icme_buffer = 1.5*u.day
sw_buffer = 0.1*u.day
#fill values for ICME removal if interpolation isn't possible
vsw_fill = 450 *u.km/u.s
bx_fill = 0

cme_source = 'donki' #'ukmo'#


#directory structure
cwd = os.path.dirname(os.path.realpath(__file__))
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
data_dir = os.path.join(os.path.dirname(cwd), 'data')
icmelist_path = os.path.join(data_dir, 'Richardson_Cane_Porcessed_ICME_list.csv')


# <codecell> functions


def rename_cone_file(file_in):
    """
    Function to rename Cone CME files for a more sensible archive. UKMO API file naming is WILD!
    Args:
        file_in: String name of the file to download
    Returns:
        file_out: String name of the full path to download the file to.
    """

    separator = '%2F'
    parts = file_in.split(separator)

    #  takes the parts of the filename and assigns them to what part of the date they are
    year = int(parts[2])
    month = int(parts[3])
    day = int(parts[4])
    hour = int(parts[5])
    #  combines the parts into a datetime object
    date = datetime.datetime(year, month, day, hour)
    #  converts the date into a string, adding zeroes if the month or day is only one number
    date_str = date.strftime("%Y%m%d%H")

    #  creates the new file name from the date string
    file_out = "cone_cme_{}.in".format(date_str)
    
    return file_out

def batch_download_cone_files(startdate, stopdate, download_dir):
    #grab all cone files between the given dates
    
    version = 'v1'
    api_key = os.getenv("UKMO_API")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    
    startdatestr = startdate.strftime("%Y-%m-%dT%H:%M:%S")
    enddatestr = stopdate.strftime("%Y-%m-%dT%H:%M:%S")
    
    request_url = url_base + "/" + version + "/data/swc-enlil-wsa?from=" + startdatestr + "&to=" + enddatestr
    response = requests.get(request_url, headers={"accept": "application/json", "apikey": api_key})
    
    conefilepath = ''
    if response.status_code == 200:
    
        # Convert to json
        js = response.json()
        nfiles = len(js['data'])
        
        #grab all of these files
        conefile_list = []
        for i in range(0,nfiles): 
            time.sleep(2) #artifical delay to prevent breaching the max request per min
    
            cone_file_name = js['data'][i]['cone_file']
    
            cone_file_url = url_base + "/" + version + "/" + cone_file_name
    
            response_cone = requests.get(cone_file_url, headers={"apikey": api_key})
            
            #rename the conefile
            cone_renamed = rename_cone_file(cone_file_name)
            
            if response_cone.status_code == 200:
                conefilepath = os.path.join(download_dir, cone_renamed)
                
                open(conefilepath, "wb").write(response_cone.content)
                conefile_list.append(conefilepath)



def earth_R(mjd):
    #returns the heliocentric distance of Earth (in km). Based on Franz+Harper2002
    AU = 149597870.691
    
    #first up, switch to JD.
    JD=mjd+2400000.5
    d0=JD-2451545
    T0=d0/36525


    L2=100.4664568 + 35999.3728565*T0
    g2=L2-(102.9373481+0.3225654*T0)
    g2=g2*np.pi/180 #mean anomaly

    rAU=1.00014 - 0.01671*np.cos(g2)-0.00014*np.cos(2*g2);
    R=rAU*AU
    
    return R

def ICMElist(filepath):
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


def find_cone_files_from_archive(start_date, end_date, directory, midnightonly = False):

    # Define the regular expression for the filename format cone_cme_YYYYMMDDHH
    date_pattern = re.compile(r"cone_cme_(\d{4})(\d{2})(\d{2})(\d{2}).in")
    
    def extract_date_from_filename(filename):
        match = date_pattern.search(filename)
        if match:
            year, month, day, hour = match.groups()
            date_str = f"{year}-{month}-{day} {hour}:00"
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            return date_obj
        return None
    
    if midnightonly: #only grab cone files at midnight
        # Use glob to find all files that match the pattern
        matching_files = glob.glob(f"{directory}/cone_cme_*00.in")
    else: #get all cone files 
        #Use glob to find all files that match the pattern
        matching_files = glob.glob(f"{directory}/cone_cme_*")
    
    # List to store matching file names within the date range
    filtered_files = []
    
    # Iterate over the matching files
    for file_path in matching_files:
        # Extract the filename from the path
        filename = file_path.split('/')[-1]
        file_date = extract_date_from_filename(filename)
        if file_date and start_date <= file_date <= end_date:
            filtered_files.append(filename)
            
    # Create a list of tuples (filename, date)
    filename_date_pairs = [(filename, extract_date_from_filename(filename)) for filename in filtered_files]
    
    # Sort the list of tuples by the date
    sorted_filename_date_pairs = sorted(filename_date_pairs, key=lambda x: x[1])
    
    # Extract the sorted filenames
    sorted_filenames = [filename for filename, date in sorted_filename_date_pairs]

    return sorted_filenames

# <codecell> Download and process OMNI
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

obs = omni.copy()
if recon_noicmes:
    
    #create a copy of the OMNI data for ICME removal, before any end padding
    omni_noicmes = omni.copy()
    
    #load the DONKI ICME list
    if icme_list == 'DONKI':
        icmes = Hin.get_DONKI_ICMEs(dl_starttime, dl_endtime)
    elif icme_list == 'CaneRichardson':
        icmes = ICMElist(icmelist_path)
    
    
    #remove all ICMEs
    omni_noicmes =  Hin.remove_ICMEs(omni, icmes, interpolate = True, 
                     icme_buffer = icme_buffer, interp_buffer = sw_buffer,
                     params = ['V', 'BX_GSE'], fill_vals = None)
      
    
    #if the no icme data don't span a large enough time range, repeat the last 27 days
    data_end_date = omni_noicmes['datetime'][len(omni_noicmes)-1]
    if data_end_date < run_stop:
        mask = (omni_noicmes['datetime'] >= data_end_date - datetime.timedelta(days = 27.27))
        datachunk = omni_noicmes[mask]
        datachunk.loc[:,'datetime'] = datachunk['datetime'] + datetime.timedelta(days = 27.27)
        datachunk.loc[:,'mjd'] = datachunk['mjd'] + 27.27
        #concatonate teh dataframes
        omni_noicmes = pd.concat([omni_noicmes, datachunk], ignore_index=True)
    
    obs = omni_noicmes.copy()
      
plt.figure()
plt.plot(omni['datetime'], omni['V'],'k', label='OMNI')   
plt.plot(omni_noicmes['datetime'], omni_noicmes['V'],'r', label='OMNI, no ICMEs')
plt.legend()
    




# <codecell> set up inner boundary conditions for HUXt form cortoation of OMNI
method = 'both'
    
#create  vCarr array with the omni time series at 1 AU
#======================================================


time1au_both, v1au_both, b1au_both = Hin.generate_vCarr_from_OMNI(run_start, run_stop, 
                                                                omni_input =obs,
                                                                corot_type='both')

# time1au_dtw_hires, lons, v1au_dtw_hires, b1au_dtw = Hin.generate_vCarr_from_OMNI_DTW(run_start, run_stop, 
#                                                                 omni_input =obs)

# #put dtw data on same time step as the other corotation methods
# v1au_dtw = v1au_both.copy()
# for nlon in range(0,len(v1au_dtw_hires[:,1])):
#     v1au_dtw[nlon,:] = np.interp(time1au_both, time1au_dtw_hires, v1au_dtw_hires[nlon,:])
# time1au_dtw = time1au_both


#now map each timestep back to the inner boundary
#================================================
vcarr_rmin_both = v1au_both.copy()
vcarr_rmin_dtw = v1au_both.copy()

for i in range(0, len(time1au_both)):
    #get the Earth heliocentric distance at this time
    Earth_R_km = earth_R(time1au_both[i]) *u.km
    #Map from 215 rto 21.5 rS
    
    vcarr_rmin_both[:,i] = Hin.map_v_boundary_inwards(v1au_both[:,i]*u.km/u.s, 
                                    Earth_R_km.to(u.solRad), rmin)
   
    # vcarr_rmin_dtw[:,i] = Hin.map_v_boundary_inwards(v1au_dtw[:,i]*u.km/u.s, 
    #                                 Earth_R_km.to(u.solRad), rmin)
    
   
    


# <codecell> run HUXt

#set up the model at 21.5 rS with backmapped OMNI
#========================================================



model_both_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_both, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'sidereal', 
                            track_cmes = False)
model_both_nocmes.solve([])  

# model_dtw_nocmes = Hin.set_time_dependent_boundary(vcarr_rmin_dtw, time1au_dtw, 
#                             run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
#                             dt_scale=dt_scale, latitude=0*u.deg, frame = 'sidereal', 
#                             track_cmes = False)
# model_dtw_nocmes.solve([])  


# #set up the model at 21.5 rS with backmapped OMNI
# #========================================================
# model_cnn = Hin.set_time_dependent_boundary(vcarr_rmin_cnn, time1au, 
#                             run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
#                             dt_scale=dt_scale, latitude=0*u.deg, frame = 'sidereal', 
#                             track_cmes = False)
# model_cnn.solve([])    

# <codecell> Extract conditions at Saturn

# #get conditions at Mercury

# Saturn_ts_both_nocmes = HA.get_observer_timeseries(model_both_nocmes, observer='Saturn', suppress_warning = True)
# #Saturn_ts_dtw_nocmes = HA.get_observer_timeseries(model_dtw_nocmes, observer='Saturn', suppress_warning = True)


# plt.figure()
# plt.plot(Saturn_ts_both_nocmes['time'], Saturn_ts_both_nocmes['vsw'], 'k', label = 'OMNI-HUXt')
# #plt.plot(Saturn_ts_dtw_nocmes['time'], Saturn_ts_dtw_nocmes['vsw'], 'b', label = 'OMNI-HUXt')
# plt.ylabel(r'$V_{SW}$ [km/s]')
# plt.xlim((run_start,run_stop)  ) 
# plt.legend() 




# <codecell> Add some CMEs to the mix

model_both_cmes = Hin.set_time_dependent_boundary(vcarr_rmin_both, time1au_both, 
                            run_start, simtime = simtime, r_min=rmin, r_max=rmax, 
                            dt_scale=dt_scale, latitude=0*u.deg, frame = 'sidereal', 
                            track_cmes = False)


cme_start = run_start - datetime.timedelta(days = 5)
cme_stop = run_stop
if cme_source == 'donki':
    #get the DONKI coneCMEs
    cme_list = Hin.get_DONKI_cme_list(model_both_cmes , 
                                      cme_start, cme_stop)
elif cme_source == 'ukmo':
     datadir = H._setup_dirs_()['HUXt_data']
     #batch_download_cone_files(cme_start, cme_stop, datadir)
     conefile_list = find_cone_files_from_archive(cme_start, cme_stop, datadir,
                                             midnightonly = True)


     #load all the CME lists in, create a list of lists of CME objects
     cmelist_list = []
     for conefile in conefile_list:
         conepath = os.path.join(datadir, conefile)
         cmelist = Hin.ConeFile_to_ConeCME_list_time(conepath, run_start)
         cmelist_list.append(cmelist)


     #consolidate the list of CME lists into a single list
     cme_list = Hin.consolidate_cme_lists(cmelist_list)
        
#change each CME to fixed duration
for cme in cme_list:
    cme.cme_fixed_duration = True
    cme.fixed_duration = 8*60*60*u.s



model_both_cmes.solve(cme_list)  



Saturn_ts_both_cmes = HA.get_observer_timeseries(model_both_cmes, observer='Saturn', suppress_warning = True)


# <codecell> Plot with CMEs too

# plt.figure()
# #plt.plot(mars_ts_back['time'],mars_ts_back['vsw'], label = 'OMNI/HUXt (back)')
# #plt.plot(mars_ts_forward['time'],mars_ts_forward['vsw'], label = 'OMNI/HUXt (forward)')
# plt.plot(Saturn_ts_both_nocmes['time'], Saturn_ts_both_nocmes['vsw'], 'k', label = 'OMNI/HUXt (Ambient)')
# plt.plot(Saturn_ts_both_cmes['time'], Saturn_ts_both_cmes['vsw'], 'r', label = 'OMNI/HUXt (CMEs)')
# #plt.plot(maven['time'],maven['vsw'], 'bo', label = 'MAVEN')
# plt.ylabel(r'$V_{SW}$ [km/s]')
# plt.xlim((run_start,run_stop)  ) 
# plt.ylim((300, 700))
# plt.legend() 

# <codecell> animate it.

#HA.animate(model_both_cmes, duration=30, tag='Saturn_Nov2024')