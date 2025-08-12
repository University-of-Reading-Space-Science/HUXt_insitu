# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:24:38 2024

in-situ data anaylsis routines

@author: mathewjowens
"""



import pandas as pd
import os as os
import numpy as np
import re  #for dealing with non-numeric characters in a string of unknown length
from datetime import datetime
import urllib.request

#Mattlab files


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
        import system as system
        datapath =  system._setup_dirs_()['datapath']
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
            icmes.at[rownum,colnum] = datetime(year,month, day,hour,minute,0)
            
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
        import system as system
        datapath =  system._setup_dirs_()['datapath']
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
