# -*- coding: utf-8 -*-
"""
Created on Mon Jan 6 2026

Read WSA speeds at Earth latitude from 2008 through 2023
Following the pattern from insitu_huxt_forecasts_analysis.py
"""

import datetime
import numpy as np
import os
import astropy.units as u
import pandas as pd
import glob
from astropy.time import Time
import matplotlib.pyplot as plt

# HUXt libraries
import huxt_inputs as Hin

# Date range
start_year = 2008
end_year = 2023

# Directory structure
cwd = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(os.path.dirname(cwd), 'data')
wsa_fits_dir = os.path.join(os.getenv('DBOX'), 'Data', 'WSA_CCMC')

# Storage for WSA speed profiles
wsa_data = []

print("Reading WSA speeds at Earth latitude from 2008-2023...")
print(f"WSA directory: {wsa_fits_dir}")
print("Using HUXt ephemeris to compute Earth latitude for each date")
print("="*60)

# Loop through each year
for year in range(start_year, end_year + 1):
    yr_str = str(year)
    year_dir = os.path.join(wsa_fits_dir, yr_str)
    
    # Check if year directory exists
    if not os.path.exists(year_dir):
        print(f"Warning: Directory not found for year {year}: {year_dir}")
        continue
    
    # Get all WSA fits files for this year (format: vel_YYYYMMDDHHmmR000_gong_z.fits)
    search_pattern = os.path.join(year_dir, 'vel_*.fits')
    wsa_files = glob.glob(search_pattern)
    wsa_files.sort()
    
    print(f"\nYear {year}: Found {len(wsa_files)} WSA files")
    
    # Process each file
    for wsa_filepath in wsa_files:
        try:
            # Extract date from filename (vel_YYYYMMDDHHmmR000_gong_z.fits)
            filename = os.path.basename(wsa_filepath)
            date_str = filename.split('_')[1][:12]  # YYYYMMDDHHmm
            file_datetime = datetime.datetime.strptime(date_str, '%Y%m%d%H%M')
            
            # Get Earth's latitude for this date using HUXt ephemeris
            earth_lat = Hin.get_earth_lat(file_datetime)
            
            # Get WSA longitude profile at Earth latitude
            vr_profile = Hin.get_WSA_long_profile(wsa_filepath, lat=earth_lat)
            
            # Store the data
            wsa_data.append({
                'datetime': file_datetime,
                'year': year,
                'earth_lat_rad': earth_lat.value,
                'earth_lat_deg': earth_lat.to(u.deg).value,
                'filepath': wsa_filepath,
                'vr_profile': vr_profile,
                'mean_speed': np.mean(vr_profile.value),
                'min_speed': np.min(vr_profile.value),
                'max_speed': np.max(vr_profile.value),
                'n_lon_bins': len(vr_profile)
            })
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue
    
    if len(wsa_files) > 0:
        year_speeds = [d['mean_speed'] for d in wsa_data if d['year'] == year]
        print(f"  Mean speed for {year}: {np.mean(year_speeds):.1f} km/s")

print("\n" + "="*60)
print(f"Total WSA files processed: {len(wsa_data)}")

# Convert to DataFrame for easier analysis
df_wsa = pd.DataFrame([{
    'datetime': d['datetime'],
    'year': d['year'],
    'earth_lat_deg': d['earth_lat_deg'],
    'mean_speed': d['mean_speed'],
    'min_speed': d['min_speed'],
    'max_speed': d['max_speed'],
    'n_lon_bins': d['n_lon_bins']
} for d in wsa_data])

# Save to CSV
output_file = os.path.join(datadir, 'wsa_speeds_earth_lat_2008_2023.csv')
df_wsa.to_csv(output_file, index=False)
print(f"\nSaved WSA speed data to: {output_file}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Date range: {df_wsa['datetime'].min()} to {df_wsa['datetime'].max()}")
print(f"Total files: {len(df_wsa)}")
print(f"\nEarth latitude range (deg):")
print(f"  Min: {df_wsa['earth_lat_deg'].min():.2f}")
print(f"  Max: {df_wsa['earth_lat_deg'].max():.2f}")
print(f"  Mean: {df_wsa['earth_lat_deg'].mean():.2f}")
print(f"\nSpeed statistics (km/s):")
print(f"  Mean speed (all files): {df_wsa['mean_speed'].mean():.1f}")
print(f"  Std dev:                {df_wsa['mean_speed'].std():.1f}")
print(f"  Min speed:              {df_wsa['min_speed'].min():.1f}")
print(f"  Max speed:              {df_wsa['max_speed'].max():.1f}")
print(f"\nLongitude bins per file: {df_wsa['n_lon_bins'].iloc[0]}")

# Print yearly summary
print("\n" + "="*60)
print("YEARLY SUMMARY")
print("="*60)
yearly_summary = df_wsa.groupby('year').agg({
    'datetime': 'count',
    'mean_speed': ['mean', 'std'],
    'min_speed': 'min',
    'max_speed': 'max'
})
yearly_summary.columns = ['N_files', 'Mean_Speed', 'Std_Speed', 'Min_Speed', 'Max_Speed']
print(yearly_summary.to_string())


# <codecell> Create histogram of all WSA speeds (all longitudes and times combined)

print("\n" + "="*60)
print("CREATING HISTOGRAM OF ALL WSA SPEEDS")
print("="*60)

# Collect all speeds from all longitude profiles at all times
all_speeds = []
for d in wsa_data:
    all_speeds.extend(d['vr_profile'].value)

all_speeds = np.array(all_speeds)

print(f"Total speed values: {len(all_speeds):,}")
print(f"Mean speed: {np.mean(all_speeds):.1f} km/s")
print(f"Median speed: {np.median(all_speeds):.1f} km/s")
print(f"Std dev: {np.std(all_speeds):.1f} km/s")
print(f"Min speed: {np.min(all_speeds):.1f} km/s")
print(f"Max speed: {np.max(all_speeds):.1f} km/s")


# <codecell> Load pre-computed WSA-HUXt forecasts at 1 AU

print("\n" + "="*60)
print("LOADING WSA-HUXT FORECASTS AT 1 AU")
print("="*60)

# Load all WSA-HUXt forecast files for all lead times
wsa_huxt_speeds = []

for lt in range(1, 28):  # Lead times 1-27 days
    wsa_filename = f'wsa_huxt_forecasts_{lt}day_leadtime.csv'
    wsa_filepath = os.path.join(datadir, wsa_filename)
    
    if os.path.exists(wsa_filepath):
        df_wsa_huxt = pd.read_csv(wsa_filepath, parse_dates=['time'])
        wsa_huxt_speeds.extend(df_wsa_huxt['vsw'].values)
        print(f"Loaded {wsa_filename}: {len(df_wsa_huxt)} forecasts")
    else:
        print(f"Warning: File not found: {wsa_filename}")

wsa_huxt_speeds = np.array(wsa_huxt_speeds)

# Remove NaN values
wsa_huxt_speeds = wsa_huxt_speeds[~np.isnan(wsa_huxt_speeds)]

print(f"\nTotal WSA-HUXt speed values at 1 AU: {len(wsa_huxt_speeds):,}")
print(f"Mean speed: {np.mean(wsa_huxt_speeds):.1f} km/s")
print(f"Median speed: {np.median(wsa_huxt_speeds):.1f} km/s")
print(f"Std dev: {np.std(wsa_huxt_speeds):.1f} km/s")
print(f"Min speed: {np.min(wsa_huxt_speeds):.1f} km/s")
print(f"Max speed: {np.max(wsa_huxt_speeds):.1f} km/s")


# <codecell> Create combined histogram figure with both distributions

print("\n" + "="*60)
print("CREATING COMBINED HISTOGRAM FIGURE")
print("="*60)

# Create figure with 2 subplots (upper and lower)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Use bins from 200 to 900 km/s with 10 km/s spacing
bins = np.arange(200, 910, 10)

# Upper subplot: WSA speeds at Earth latitude
counts1, bin_edges1, patches1 = ax1.hist(all_speeds, bins=bins, color='#2E86AB', 
                                          alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axvline(np.mean(all_speeds), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(all_speeds):.1f} km/s')
ax1.axvline(np.median(all_speeds), color='orange', linestyle='--', linewidth=2,
            label=f'Median = {np.median(all_speeds):.1f} km/s')
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(f'(a) WSA Speed Distribution at Earth Latitude ({start_year}-{end_year})\n'
              f'All Longitudes and Times Combined (N = {len(all_speeds):,})', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=11)

# Lower subplot: WSA-HUXt speeds at 1 AU
counts2, bin_edges2, patches2 = ax2.hist(wsa_huxt_speeds, bins=bins, color='#A23B72', 
                                          alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(np.mean(wsa_huxt_speeds), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(wsa_huxt_speeds):.1f} km/s')
ax2.axvline(np.median(wsa_huxt_speeds), color='orange', linestyle='--', linewidth=2,
            label=f'Median = {np.median(wsa_huxt_speeds):.1f} km/s')
ax2.set_xlabel('Solar Wind Speed [km/s]', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(f'(b) WSA-HUXt Speed Distribution at 1 AU ({start_year}-{end_year})\n'
              f'All Lead Times Combined (N = {len(wsa_huxt_speeds):,})', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)

fig.tight_layout()

# Save figure
figdir = os.path.join(os.getenv('DBOX'), 'Apps', 'Overleaf', 'InSitu-HUXt_forecast')
if not os.path.exists(figdir):
    figdir = os.path.join(os.path.dirname(cwd), 'figures')
    os.makedirs(figdir, exist_ok=True)

fig_path = os.path.join(figdir, f'wsa_speed_histograms_{start_year}_{end_year}.pdf')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nSaved combined histogram figure to: {fig_path}")

plt.show()
