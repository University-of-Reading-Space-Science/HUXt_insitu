"""
Script to determine how much time Parker Solar Probe (PSP) and Solar Orbiter
have spent within 1-, 2-, 3-, 4-, and 5-degrees HEEQ latitude of Earth,
between their respective launch dates and end of 2026.

Uses:
 - HUXt ephemeris HDF5 file for Earth HEEQ latitude
 - sunpy get_horizons_coord for PSP and Solar Orbiter HEEQ latitude
"""

import os
import datetime
import numpy as np
import h5py
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.coordinates import get_horizons_coord
import pandas as pd

# Setup paths (mirror huxt.py _setup_dirs_)
cwd = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(cwd)
ephem_path = os.path.join(root, 'data', 'ephemeris', 'ephemeris.hdf5')

# ---- Define date ranges ----
# Start slightly after actual launch to ensure Horizons ephemeris coverage
psp_launch = datetime.datetime(2018, 8, 13)
solo_launch = datetime.datetime(2020, 2, 11)
end_date = datetime.datetime(2026, 12, 31)

# ---- Helper: get Earth HEEQ latitude from ephemeris HDF5 at hourly cadence ----
def get_earth_lat_from_ephem(start_dt, stop_dt):
    """Get Earth HEEQ latitude from HUXt ephemeris HDF5, interpolated hourly."""
    ephem = h5py.File(ephem_path, 'r')
    all_time = Time(ephem['EARTH']['HEEQ']['time'], format='jd')
    all_lat_deg = ephem['EARTH']['HEEQ']['latitude'][:]  # in degrees
    ephem.close()

    # Build hourly time array
    n_hours = int((stop_dt - start_dt).total_seconds() / 3600)
    hourly_dt = [start_dt + datetime.timedelta(hours=i) for i in range(n_hours + 1)]
    hourly_time = Time(hourly_dt)

    # Interpolate
    lat_rad = np.deg2rad(all_lat_deg)
    earth_lat_rad = np.interp(hourly_time.jd, all_time.jd, lat_rad)
    earth_lat_deg = np.rad2deg(earth_lat_rad)

    return hourly_time, hourly_dt, earth_lat_deg


# ---- Helper: get spacecraft HEEQ latitude from JPL Horizons ----
def get_sc_lat_from_horizons(obj_name, start_dt, stop_dt):
    """
    Get spacecraft HEEQ latitude from JPL Horizons via sunpy.
    Queries in 6-month chunks to avoid timeout issues.
    Returns JD times and latitude in degrees.
    """
    all_lats = []
    all_times_jd = []

    # Query in 180-day chunks
    chunk_start = start_dt
    while chunk_start < stop_dt:
        chunk_stop = min(chunk_start + datetime.timedelta(days=180), stop_dt)
        print(f"  Querying Horizons for {obj_name}: {chunk_start.date()} to {chunk_stop.date()}")

        t_start = Time(chunk_start)
        t_stop = Time(chunk_stop)

        coord = get_horizons_coord(obj_name, {
            'start': t_start,
            'stop': t_stop,
            'step': '4H'
        })

        all_lats.append(coord.lat.deg)  # HEEQ latitude in degrees
        # Use obstime for accurate JD values
        all_times_jd.append(coord.obstime.jd)

        chunk_start = chunk_stop

    lats = np.concatenate(all_lats)
    times_jd = np.concatenate(all_times_jd)

    # Remove any duplicate times from chunk boundaries
    _, unique_idx = np.unique(times_jd, return_index=True)
    times_jd = times_jd[unique_idx]
    lats = lats[unique_idx]

    return times_jd, lats


# ---- Main analysis ----
def analyse_lat_proximity(sc_name, sc_horizons_id, launch_dt, end_dt):
    """
    For a given spacecraft, compute time spent within various latitude
    thresholds of Earth.
    """
    print(f"\n{'='*60}")
    print(f"Analysing {sc_name}")
    print(f"Period: {launch_dt.date()} to {end_dt.date()}")
    print(f"{'='*60}")

    # Get Earth latitude for this period
    print("Getting Earth HEEQ latitude from ephemeris...")
    earth_time, earth_dt, earth_lat = get_earth_lat_from_ephem(launch_dt, end_dt)

    # Get spacecraft latitude from Horizons
    print(f"Getting {sc_name} HEEQ latitude from JPL Horizons...")
    sc_times_jd, sc_lat = get_sc_lat_from_horizons(sc_horizons_id, launch_dt, end_dt)

    # Interpolate spacecraft lat onto the same hourly Earth time grid
    sc_lat_interp = np.interp(earth_time.jd, sc_times_jd, sc_lat)

    # Compute absolute latitude difference
    lat_diff = np.abs(sc_lat_interp - earth_lat)

    # Total duration in days
    total_hours = len(earth_lat)
    total_days = total_hours / 24.0

    # Compute time within each threshold
    thresholds = [1, 2, 3, 4, 5]
    print(f"\nTotal period: {total_days:.1f} days ({total_days/365.25:.2f} years)")
    print(f"\n{'Threshold':>12s} | {'Hours':>8s} | {'Days':>8s} | {'% of time':>10s}")
    print("-" * 50)

    results = {}
    for thresh in thresholds:
        mask = lat_diff <= thresh
        hours_within = np.sum(mask)
        days_within = hours_within / 24.0
        pct = 100.0 * hours_within / total_hours
        print(f"  <= {thresh} deg    | {hours_within:8d} | {days_within:8.1f} | {pct:9.1f}%")
        results[thresh] = {'hours': int(hours_within), 'days': days_within, 'percent': pct}

    return results, lat_diff, earth_lat, sc_lat_interp


if __name__ == '__main__':

    # PSP: JPL Horizons ID is 'Parker Solar Probe' or -96
    print("=" * 60)
    print("Querying latitude proximity of PSP and Solar Orbiter to Earth")
    print("=" * 60)

    psp_results, psp_diff, earth_lat_psp, psp_lat = analyse_lat_proximity(
        'Parker Solar Probe', 'Parker Solar Probe', psp_launch, end_date
    )

    solo_results, solo_diff, earth_lat_solo, solo_lat = analyse_lat_proximity(
        'Solar Orbiter', 'Solar Orbiter', solo_launch, end_date
    )

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'':>15s} | {'PSP':>28s} | {'Solar Orbiter':>28s}")
    print(f"{'':>15s} | {'(launch 2018-08-12)':>28s} | {'(launch 2020-02-10)':>28s}")
    print("-" * 75)

    psp_total_days = sum(1 for _ in range(int((end_date - psp_launch).total_seconds() / 3600) + 1)) / 24.0
    solo_total_days = sum(1 for _ in range(int((end_date - solo_launch).total_seconds() / 3600) + 1)) / 24.0

    for thresh in [1, 2, 3, 4, 5]:
        psp_d = psp_results[thresh]['days']
        psp_p = psp_results[thresh]['percent']
        solo_d = solo_results[thresh]['days']
        solo_p = solo_results[thresh]['percent']
        print(f"  <= {thresh} deg lat  | {psp_d:7.1f} days ({psp_p:5.1f}%)      | {solo_d:7.1f} days ({solo_p:5.1f}%)")

    print("\nDone.")
