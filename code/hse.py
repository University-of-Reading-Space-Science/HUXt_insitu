__author__ = 'Harriet Turner'
__email__ = 'harriet.turner2@metoffice.gov.uk'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def smooth_8h(df, date_col, window_hours=8):
    """
    Function to smooth the contents of a dataframe.
    Parameters
    ----------
    df
    date_col
    window_hours

    Returns
    -------

    """

    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Handle duplicate timestamps by aggregating (mean)
    df = df.groupby(date_col).mean().sort_index()

    # Create full hourly index
    full_index = pd.date_range(start=df.index.min(),
                               end=df.index.max(),
                               freq="h")

    # Reindex (missing hours become NaN)
    df = df.reindex(full_index)

    # Apply rolling mean
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].rolling(window=window_hours,
                                                min_periods=1, center=True).mean()

    return df


def detect_hss_events(df, speed_col='V', rise_threshold=150, rise_hours=48, min_separation_hours=24):
    """
    Identify high-speed-stream (HSS) events in a solar-wind time series.

    Parameters
    ----------
    df : DataFrame with DateTimeIndex and a speed column
    speed_col : column name containing solar wind speed
    rise_threshold : speed increase needed to define event (km/s)
    rise_hours : time over which the rise must occur
    min_separation_hours : minimum allowed separation between events

    Returns
    -------
    events : list of dicts with 'start', 'peak', 'peak_speed', 'max_gradient_time'
    """

    df = df.copy()

    events = []
    times = df.index
    V = df[speed_col].values

    # convert rise_hours to number of samples (assuming uniform cadence)
    dt = (times[1] - times[0]).total_seconds() / 3600  # dt in hours
    step = int(rise_hours / dt)

    i = 0
    while i < len(df) - step:
        dv = V[i + step] - V[i]
        if dv >= rise_threshold:
            # HSE detected at position i+step
            # The HSE spans from i to i+step at minimum
            
            # Find the actual start: the local minimum in the window [i, i+step]
            start_idx = i + np.argmin(V[i:i + step + 1])
            
            # Find the end of HSE: continue until speed stops increasing
            end_idx = i + step
            for j in range(i + step, len(V) - 1):
                if V[j + 1] < V[j]:
                    # Speed started decreasing
                    break
                end_idx = j + 1
            
            # Peak is the maximum speed in the HSE period
            peak_idx = start_idx + np.argmax(V[start_idx:end_idx + 1])
            
            # Calculate max gradient time in the rise phase [start_idx, peak_idx]
            if peak_idx > start_idx + 1:
                # Calculate gradient at each point in rise phase
                rise_speeds = V[start_idx:peak_idx + 1]
                # gradients[j] = (V[start_idx+j+1] - V[start_idx+j]) / dt
                gradients = np.diff(rise_speeds) / dt
                # The time of max gradient is at the point where gradient is maximum
                # np.diff reduces length by 1, so gradient[j] corresponds to time[start_idx+j]
                max_grad_offset = np.argmax(gradients)
                max_grad_idx = start_idx + max_grad_offset
            elif peak_idx == start_idx + 1:
                # Only 2 points, max gradient is at start
                max_grad_idx = start_idx
            else:
                # Single point, use start
                max_grad_idx = start_idx

            events.append({
                'start': times[start_idx],
                'start_idx': start_idx,
                'end': times[end_idx],
                'end_idx': end_idx,
                'peak': times[peak_idx],
                'peak_speed': V[peak_idx],
                'max_gradient_time': times[max_grad_idx]
            })

            # skip ahead to avoid detecting same event twice
            i = end_idx + 1
        else:
            i += 1

    if not events:
        return []

    # Combine events where start of event A is within min_separation_hours of END of event B
    # Recalculate max gradient time across the entire combined interval
    events = sorted(events, key=lambda e: e['start'])
    merged = [events[0]]
    
    for ev in events[1:]:
        last_event = merged[-1]
        # Check separation: time from END of last event to START of current event
        dt_sep = (ev['start'] - last_event['end']).total_seconds() / 3600.0
        
        if dt_sep >= min_separation_hours:
            # Events are sufficiently separated, keep as distinct event
            merged.append(ev)
        else:
            # Events are too close, merge them
            # Combine the intervals and recalculate max gradient time
            
            # Find indices for the combined interval (start of last event to end of current event)
            combined_start_idx = last_event['start_idx']
            combined_end_idx = ev['end_idx']
            
            # Recalculate max gradient over entire combined interval
            combined_V = V[combined_start_idx:combined_end_idx + 1]
            if len(combined_V) > 1:
                combined_gradients = np.diff(combined_V) / dt
                max_grad_offset = np.argmax(combined_gradients)
                combined_max_grad_idx = combined_start_idx + max_grad_offset
            else:
                combined_max_grad_idx = combined_start_idx
            
            # Find peak in combined interval
            combined_peak_idx = combined_start_idx + np.argmax(combined_V)
            
            # Update the last merged event with combined properties
            merged[-1] = {
                'start': last_event['start'],           # Keep start of first event
                'start_idx': combined_start_idx,
                'end': ev['end'],                       # Use end of later event
                'end_idx': combined_end_idx,
                'peak': times[combined_peak_idx],       # Peak in combined interval
                'peak_speed': V[combined_peak_idx],
                'max_gradient_time': times[combined_max_grad_idx]  # Recalculated max gradient
            }

    return merged



def match_events(eventsA, eventsB, delta_t_max=72):
    """
    Match event lists using Owens (2005) method.

    Parameters
    ----------
    eventsA, eventsB : lists of event dicts ({start, peak, peak_speed, max_gradient_time})
    dt_max_hours : maximum allowed timing difference
    v_scale : normalisation for speed differences

    Returns
    -------
    matches : list of (eventA, eventB, metric)
    unmatched_A : events in A without matches
    unmatched_B : events in B without matches
    """

    matches = []

    for Ai in eventsA:
        candidates = []

        for j, Bj in enumerate(eventsB):

            # -----------------------
            # STRICT OWENS FILTER:
            # Reject if max_gradient_time >delta_t_max hours apart
            # -----------------------
            dt = abs((Bj['max_gradient_time'] - Ai['max_gradient_time']).total_seconds()) / 3600.0
            if dt > delta_t_max:
                continue  # skip – not a valid candidate

            # metric (same as your earlier one)
            D = (dt / delta_t_max)

            candidates.append((D, j, Bj))

        # choose best candidate
        if candidates:
            best = min(candidates, key=lambda x: x[0])
            matches.append((Ai, best[2], best[0], best[1]))

    # one-to-one resolution
    matches.sort(key=lambda x: x[2])
    final = []
    used_B = set()

    for Ai, Bj, D, j in matches:
        if j not in used_B:
            used_B.add(j)
            final.append((Ai, Bj, D))

    unmatched_A = [A for A in eventsA if A not in [m[0] for m in final]]
    unmatched_B = [B for B in eventsB if B not in [m[1] for m in final]]

    return final, unmatched_A, unmatched_B


def compute_timing_errors(matches, use='peak'):
    """
    Compute timing errors for matched HSS events.

    Parameters
    ----------
    matches : list of (A_event, B_event)
        Output from match_events_owens
    use : 'peak' or 'start' or 'max_gradient'
        - 'peak'  → timing error = B.peak - A.peak  (Owens definition)
        - 'start' → timing error = B.start - A.start (arrival time)
        - 'max_gradient' → timing error = B.max_gradient_time - A.max_gradient_time

    Returns
    -------
    errors_hours : list of timing errors in hours (B - A)
    """

    errors = []
    for A, B, metric_value in matches:
        if use == 'peak':
            dt = (B['peak'] - A['peak']).total_seconds() / 3600.0
        elif use == 'start':
            dt = (B['start'] - A['start']).total_seconds() / 3600.0
        elif use == 'max_gradient':
            dt = (B['max_gradient_time'] - A['max_gradient_time']).total_seconds() / 3600.0
        else:
            raise ValueError("use must be 'peak', 'start', or 'max_gradient'")
        errors.append(dt)

    return errors


def summary_error_stats(errors):
    errors = np.array(errors)
    return {
        "mean_error": np.mean(errors),
        "mean_absolute_error": np.mean(np.abs(errors)),
        "median_error": np.median(errors),
        "std_error": np.std(errors),
        "n_events": len(errors),
        "early_fraction": np.sum(errors < 0) / len(errors),
        "late_fraction": np.sum(errors > 0) / len(errors)
    }


def peak_speed_errors(matches):
    """
    Compute the peak speed errors of matched HSS events.

    Parameters
    ----------
    matches : list of tuples
        Each tuple is (observed_event, model_event) or
        (observed_event, model_event, metric) if using metric-based matching.
        Each event should have a 'peak_speed' key.

    Returns
    -------
    speed_errors : list of floats
        Difference: model_peak - observed_peak (km/s)
        Positive error means model overestimates the peak speed.
    summary : dict
        Summary statistics: mean error, mean absolute error, std, number of events
    """

    speed_errors = []

    for match in matches:
        if len(match) == 3:
            obs, mod, _ = match
        else:
            obs, mod = match

        # Error = model - obs (positive means model overestimates)
        speed_errors.append(mod['peak_speed'] - obs['peak_speed'])

    summary = {
        "mean_error": np.nanmean(speed_errors),
        "mean_absolute_error": np.nanmean(np.abs(speed_errors)),
        "std_error": np.nanstd(speed_errors),
        "n_events": len(speed_errors)
    }

    return speed_errors, summary


if __name__ == "__main__":

    forecast_file = '/Users/yx919401/Research/Analysis/HUXt_Mars/Forecast/Lead_time_forecasts/5_day_lead_time_forecast.csv'
    forecast_data = pd.read_csv(forecast_file)
    forecast_data['date'] = pd.to_datetime(forecast_data['date'], format='%Y-%m-%d %H:%M:%S')

    """data_req = forecast_data.loc[(forecast_data['date'] >= datetime(2019, 1, 1)) &
                                 (forecast_data['date'] < datetime(2020, 1, 1))]"""

    smoothed_data = smooth_8h(forecast_data, date_col='date', window_hours=8)

    # Detect events
    eventsA = detect_hss_events(smoothed_data, speed_col='Obs')
    eventsB = detect_hss_events(smoothed_data, speed_col='BRaVDA_vsw')

    # Match events
    matches, unmatched_A, unmatched_B = match_events(eventsA, eventsB)

    print("\n=== MATCHED EVENTS ===")
    for Ai, Bj, D in matches:
        print(f"A peak: {Ai['peak']}  |  B peak: {Bj['peak']}  |  metric: {D:.3f}")

    print("\nMatched events:", len(matches))
    print("Unmatched A events:", len(unmatched_A))
    print("Unmatched B events:", len(unmatched_B))

    arrival_errors = compute_timing_errors(matches, use='start')
    print("Arrival timing errors (hours):", arrival_errors)
    print(summary_error_stats(arrival_errors))

    speed_errors, summary = peak_speed_errors(matches)
    print(speed_errors)
    print("Mean absolute error:", summary['mean_absolute_error'])
    print("Mean error:", summary['mean_error'])

    speed_bins = np.arange(-250, 400, 50)
    timing_bins = np.arange(-50, 50, 10)

    plt.hist(speed_errors, edgecolor='black', linewidth=1.2, bins=speed_bins)
    plt.xlabel('Speed error, km/s')
    plt.ylabel('Count')
    plt.ylim([0, 15])
    plt.show()
