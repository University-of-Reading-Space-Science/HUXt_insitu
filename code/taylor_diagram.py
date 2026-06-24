"""
Combined Taylor and Target diagrams for model evaluation.

This module provides functions to create publication-quality combined Taylor and 
Target diagrams using matplotlib subfigures.

Key functions:
- plot_taylor_target_diagram: Plot pre-computed statistics
- combined_taylor_target_diagram: Wrapper that computes statistics from obs and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

def compute_taylor_target_stats(obs, pred):
    """
    Compute Taylor and Target diagram statistics for a single prediction.
    
    Parameters:
    -----------
    obs : array-like
        Observations (1D array)
    pred : array-like
        Predictions (1D array, same shape as obs)
    
    Returns:
    --------
    stats : dict
        Dictionary containing:
        - 'std': Standard deviation of predictions
        - 'corr': Correlation coefficient between obs and pred
        - 'crmse': Centered root-mean-square error
        - 'bias': Mean bias (pred - obs)
        - 'ubrmse': Unbiased root-mean-square error
    
    Examples:
    ---------
    >>> obs = np.random.normal(400, 100, 1000)
    >>> pred = obs + np.random.normal(10, 30, 1000)
    >>> stats = compute_taylor_target_stats(obs, pred)
    >>> print(f"Bias: {stats['bias']:.1f}, ubRMSE: {stats['ubrmse']:.1f}")
    """
    # Ensure inputs are numpy arrays
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    
    # Calculate basic statistics
    obs_std = np.std(obs, ddof=1)
    obs_mean = np.mean(obs)
    pred_std = np.std(pred, ddof=1)
    pred_mean = np.mean(pred)
    
    # Taylor diagram statistics
    corr = np.corrcoef(obs - obs_mean, pred - pred_mean)[0, 1]
    crmse = np.sqrt(obs_std**2 + pred_std**2 - 2*obs_std*pred_std*corr)
    
    # Target diagram statistics
    bias = np.mean(pred - obs)
    ubrmse = np.sqrt(np.mean((pred - obs - bias)**2))
    
    return {
        'std': pred_std,
        'corr': corr,
        'crmse': crmse,
        'bias': bias,
        'ubrmse': ubrmse,
    }

def combined_taylor_target_diagram(obs, predictions, labels, colors=None, markers=None,
                                   figsize=(16, 7), dpi=150, figname=None, normalize=False, units='',
                                   std_ranges=None, obs_std_range=None):
    """
    Wrapper function that computes statistics and creates a combined Taylor and Target diagram.
    
    This function computes RMSE, correlation, bias, and other statistics from observations
    and predictions, then calls plot_taylor_target_diagram() to visualize them.
    
    Parameters:
    -----------
    obs : array-like
        Observations (1D array)
    predictions : list of array-like
        List of prediction arrays to compare. Each should be same shape as obs.
    labels : list of str
        Labels for each prediction (e.g., ['Model 1', 'Model 2'])
    colors : list of str, optional
        RGB hex colors for markers. Default: uses tab10 colormap
    markers : list of str, optional
        Marker styles (e.g., ['o', 's']). Default: circles for all
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (16, 7)
    dpi : int, optional
        Dots per inch for saving. Default: 150
    figname : str, optional
        Filename to save the figure (without extension). If None, doesn't save.
    normalize : bool, optional
        If True, normalize by observation std dev. Default: False
    units : str, optional
        Unit string for axis labels (e.g., '[km/s]'). Default: ''
    std_ranges : list of tuple, optional
        List of (min_std, max_std) tuples for each model, to show as shaded regions on Taylor diagram.
        If provided, should have same length as predictions. Default: None
    obs_std_range : tuple, optional
        (min_obs_std, max_obs_std) tuple for observations, to show as shaded region on Taylor diagram.
        Default: None
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    (ax_taylor, ax_target) : tuple of Axes
        The Taylor diagram and Target diagram axes
        
    Examples:
    ---------
    >>> obs = np.random.normal(400, 100, 1000)
    >>> pred1 = obs + np.random.normal(10, 30, 1000)
    >>> pred2 = obs + np.random.normal(50, 60, 1000)
    >>> fig, (ax_t, ax_tg) = combined_taylor_target_diagram(
    ...     obs, [pred1, pred2], 
    ...     labels=['InSitu-HUXt', 'WSA-HUXt'],
    ...     colors=['#2E86AB', '#A23B72'],
    ...     units='[km/s]',
    ...     figname='forecast_comparison'
    ... )
    """
    # Ensure inputs are numpy arrays
    obs = np.asarray(obs)
    predictions = [np.asarray(p) for p in predictions]
    
    # Calculate statistics from observations and predictions using the standalone function
    stats = [compute_taylor_target_stats(obs, pred) for pred in predictions]
    
    # Get observation std for reference
    obs_std = np.std(obs, ddof=1)
    
    # Normalize if requested
    if normalize:
        obs_std_for_plot = 1.0  # Normalized reference
        for s in stats:
            s['std'] /= obs_std
            s['crmse'] /= obs_std
            s['bias'] /= obs_std
            s['ubrmse'] /= obs_std
    else:
        obs_std_for_plot = obs_std  # Use actual observation std
    
    # Call the plotting function
    return plot_taylor_target_diagram(stats, labels, obs_std=obs_std_for_plot,
                                     colors=colors, markers=markers,
                                     figsize=figsize, dpi=dpi, figname=figname,
                                     normalize=normalize, units=units, std_ranges=std_ranges,
                                     obs_std_range=obs_std_range)


def plot_taylor_target_diagram(stats_list, labels, obs_std=None, colors=None, markers=None,
                               figsize=(16, 7), dpi=150, figname=None, normalize=False, units='',
                               std_ranges=None, obs_std_range=None):
    """
    Plot a combined Taylor and Target diagram from pre-computed statistics.
    
    This is the core plotting function that creates two panels: a Taylor diagram (left) 
    showing correlation and standard deviation statistics, and a Target diagram (right) 
    showing bias and unbiased RMSE statistics.
    
    Parameters:
    -----------
    stats_list : list of dict
        List of dictionaries, one per model/prediction. Each dict must contain:
        - 'std': Standard deviation (float)
        - 'corr': Correlation coefficient (float, 0-1)
        - 'crmse': Centered RMSE (float)
        - 'bias': Bias (float)
        - 'ubrmse': Unbiased RMSE (float)
    labels : list of str
        Labels for each statistic set (e.g., ['Model 1', 'Model 2'])
    obs_std : float, optional
        Standard deviation of observations (for reference point on Taylor diagram).
        If None, defaults to 1.0 if normalized, otherwise uses max of prediction stds.
    colors : list of str, optional
        RGB hex colors for markers. Default: uses tab10 colormap
    markers : list of str, optional
        Marker styles (e.g., ['o', 's']). Default: circles for all
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (16, 7)
    dpi : int, optional
        Dots per inch for saving. Default: 150
    figname : str, optional
        Filename to save the figure (without extension). If None, doesn't save.
    normalize : bool, optional
        If True, values are assumed to be normalized. Default: False
    units : str, optional
        Unit string for axis labels (e.g., '[km/s]'). Default: ''
    std_ranges : list of tuple, optional
        List of (min_std, max_std) tuples for each model, to show as shaded regions on Taylor diagram.
        If provided, should have same length as stats_list. Default: None
    obs_std_range : tuple, optional
        (min_obs_std, max_obs_std) tuple for observations, to show as shaded region on Taylor diagram.
        Default: None
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    (ax_taylor, ax_target) : tuple of Axes
        The Taylor diagram and Target diagram axes
        
    Examples:
    ---------
    >>> stats = [
    ...     {'std': 80, 'corr': 0.95, 'crmse': 20, 'bias': 5, 'ubrmse': 15},
    ...     {'std': 70, 'corr': 0.85, 'crmse': 40, 'bias': 20, 'ubrmse': 35}
    ... ]
    >>> fig, (ax_t, ax_tg) = plot_taylor_target_diagram(
    ...     stats, ['Model 1', 'Model 2'],
    ...     obs_std=100,
    ...     colors=['#2E86AB', '#A23B72'],
    ...     units='[km/s]'
    ... )
    """
    from matplotlib.projections import PolarAxes
    import mpl_toolkits.axisartist.grid_finder as gf
    import mpl_toolkits.axisartist.floating_axes as fa
    
    stats = stats_list  # Use pre-computed stats directly
    
    # Set reference obs_std if not provided
    if obs_std is None:
        if normalize:
            obs_std = 1.0
        else:
            # Use the mean of the prediction stds as a reasonable reference
            obs_std = np.mean([s['std'] for s in stats])
    
    # ===== TAYLOR DIAGRAM =====
    def _create_taylor(fig, subfig, stats, labels, colors, markers, obs_std, normalize, units):
        """Create Taylor diagram in subfigure"""
        
        max_std = max([s['std'] for s in stats]) * 1.5
        
        # Calculate standard deviation contour levels
        if normalize:
            std_levels = np.array([0.5, 1.0, 1.5])
            std_levels = std_levels[std_levels < max_std]
        else:
            max_std_rounded = np.ceil(max_std / 10) * 10
            if max_std_rounded <= 50:
                contour_spacing = 10
            elif max_std_rounded <= 100:
                contour_spacing = 20
            elif max_std_rounded <= 200:
                contour_spacing = 40
            elif max_std_rounded <= 500:
                contour_spacing = 100
            else:
                contour_spacing = 200
            
            std_levels = np.arange(contour_spacing, max_std, contour_spacing)
        
        # Create polar axes
        tr = PolarAxes.PolarTransform()
        
        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = gf.FixedLocator(tlocs)
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        # Standard deviation labels
        max_std_plot = max_std
        gl2 = gf.FixedLocator(std_levels)
        std_labels_dict = dict(zip(std_levels, [f'{int(s)}' if s == int(s) else f'{s:.1f}' for s in std_levels]))
        std_labels_dict[0] = ''
        tf2 = gf.DictFormatter(std_labels_dict)
        
        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, 0, max_std_plot),
            grid_locator1=gl1, tick_formatter1=tf1,
            grid_locator2=gl2, tick_formatter2=tf2
        )
        
        ax = fa.FloatingSubplot(subfig, 111, grid_helper=ghelper)
        subfig.add_subplot(ax)
        
        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].major_ticklabels.set_fontsize(14)
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_fontsize(16)
        
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard Deviation" + (" (normalized)" if normalize else f" {units}"))
        ax.axis["left"].label.set_fontsize(16)
        ax.axis["left"].major_ticklabels.set_fontsize(14)
        
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["right"].major_ticklabels.set_fontsize(14)
        
        ax.axis["bottom"].set_visible(False)
        
        ax = ax.get_aux_axes(tr)
        ax.grid(False)
        
        # Add "0" label at origin
        ax.text(0, 0, '0', fontsize=14, ha='right', va='top', color='black', zorder=20)
        
        # Add correlation grid lines
        corr_grid_values = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        for corr_val in corr_grid_values:
            theta_grid = np.arccos(corr_val)
            ax.plot([theta_grid, theta_grid], [0, max_std_plot], 'b-', 
                   linewidth=0.5, alpha=0.3, zorder=1)
        
        # Add standard deviation contours
        for std_val in std_levels:
            theta_std = np.linspace(0, np.pi/2, 100)
            r_std = std_val * np.ones_like(theta_std)
            ax.plot(theta_std, r_std, color='grey', linewidth=1.0, alpha=0.5, zorder=1, linestyle='--')
        
        # Add RMSE contours
        if normalize:
            rmse_levels = np.array([0.5, 1.0, 1.5])
        else:
            max_rmse = max_std * 1.3
            n_contours = 3
            rmse_spacing = max_rmse / (n_contours + 1)
            rmse_levels = np.arange(rmse_spacing, max_rmse, rmse_spacing)
            if rmse_spacing > 20:
                rmse_spacing_rounded = np.round(rmse_spacing / 20) * 20
                rmse_levels = np.arange(rmse_spacing_rounded, max_rmse, rmse_spacing_rounded)
        
        for i, rmse in enumerate(rmse_levels):
            angles = np.linspace(0, np.pi, 200)
            x_cart = obs_std + rmse * np.cos(angles)
            y_cart = rmse * np.sin(angles)
            r_polar = np.sqrt(x_cart**2 + y_cart**2)
            theta_polar = np.arctan2(y_cart, x_cart)
            mask = (theta_polar >= 0) & (theta_polar <= np.pi/2) & (r_polar <= max_std_plot)
            
            if np.any(mask):
                ax.plot(theta_polar[mask], r_polar[mask], 'k-', 
                       linewidth=1.0, alpha=0.5, zorder=2)
                
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    mid_idx = valid_indices[len(valid_indices) // 2]
                    theta_label = theta_polar[mid_idx]
                    r_label = r_polar[mid_idx]
                    
                    if r_label <= max_std_plot * 0.98:
                        if i == 0:
                            label_text = f'uRMSE = {rmse:.0f}' if not normalize else f'uRMSE = {rmse:.2f}'
                        else:
                            label_text = f'{rmse:.0f}' if not normalize else f'{rmse:.2f}'
                        ax.text(theta_label, r_label, label_text, 
                               fontsize=12, color='black', ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                        edgecolor='none', alpha=0.9), zorder=4)
        
        # Add shaded region for observation std range if provided
        if obs_std_range is not None:
            from matplotlib.patches import Polygon
            theta_fill = np.linspace(0, np.pi/2, 100)
            
            min_obs_std, max_obs_std = obs_std_range
            
            # Create polygon for the annulus (ring) between min and max obs std
            # Outer arc
            r_outer = max_obs_std * np.ones_like(theta_fill)
            x_outer = r_outer * np.sin(theta_fill)
            y_outer = r_outer * np.cos(theta_fill)
            
            # Inner arc
            r_inner = min_obs_std * np.ones_like(theta_fill)
            x_inner = r_inner * np.sin(theta_fill)
            y_inner = r_inner * np.cos(theta_fill)
            
            # Combine to create closed polygon
            verts = list(zip(theta_fill, r_outer)) + list(zip(theta_fill[::-1], r_inner[::-1]))
            poly = Polygon(verts, closed=True, facecolor='black', alpha=0.08, 
                         edgecolor='none', zorder=1)
            ax.add_patch(poly)
        
        # Add reference std arc
        t = np.linspace(0, np.pi/2, 100)
        r = obs_std * np.ones_like(t)
        ax.plot(t, r, 'k-', linewidth=1.5, alpha=0.7)
        
        # Add y-axis label
        ylabel_text = "Standard Deviation (normalized)" if normalize else "Standard Deviation [km/s]"
        ax.text(-0.08, 0.5, ylabel_text, transform=ax.transAxes, 
               fontsize=15, rotation=90, va='center', ha='center')
        
        # Plot reference point
        ax.plot(0, obs_std, 'k*', markersize=15, label='Observations', zorder=10)
        
        # Set colors and markers
        if colors is None:
            from matplotlib import colormaps
            colors = colormaps['tab10'](np.linspace(0, 1, len(stats)))
        if markers is None:
            markers = ['o'] * len(stats)
        
        # Plot model points
        for i, (stat, label) in enumerate(zip(stats, labels)):
            theta = np.arccos(stat['corr'])
            r = stat['std']
            ax.plot(theta, r, marker=markers[i], color=colors[i], 
                   markersize=12, label=label, zorder=10, 
                   markeredgecolor='white', markeredgewidth=1.5)
        
        # Add (a) label at top left
        ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', 
                va='top', ha='left')
        
        return ax
    
    # ===== TARGET DIAGRAM =====
    def _create_target(subfig, stats, labels, colors, normalize, units):
        """Create Target diagram in subfigure"""
        
        ax = subfig.subplots()
        
        # Plot axes
        ax.axhline(0, color='k', linewidth=1, alpha=0.3)
        ax.axvline(0, color='k', linewidth=1, alpha=0.3)
        
        # Plot models
        if colors is None:
            from matplotlib import colormaps
            colors = colormaps['tab10'](np.linspace(0, 1, len(stats)))
        
        for i, s in enumerate(stats):
            ax.plot(s['bias'], s['ubrmse'], 'o', color=colors[i], 
                    markersize=12, label=labels[i], zorder=10,
                    markeredgecolor='white', markeredgewidth=1.5)
        
        # Add star at origin
        ax.plot(0, 0, 'k*', markersize=15, zorder=10)
        
        # Calculate axis limits intelligently
        max_bias = max(np.abs([s['bias'] for s in stats]))
        max_ubrmse = max([s['ubrmse'] for s in stats])
        min_bias = min([s['bias'] for s in stats])
        
        # Start with a conservative margin around data
        x_margin = max(np.abs(max_bias), np.abs(min_bias)) * 0.15
        y_margin = max_ubrmse * 0.15
        
        # Initial axis limits with margins
        x_max = max(np.abs(max_bias), np.abs(min_bias)) + x_margin
        y_max = max_ubrmse + y_margin
        
        # For square aspect ratio, the height must equal width
        # Width = 2 * x_max, Height = y_max - y_min
        # So we need: y_max - y_min = 2 * x_max
        # Set y_min to maintain space for origin visibility
        y_min = -x_max  # This gives height = 2*x_max, matching width of 2*x_max
        
        # Ensure all points are comfortably within limits with additional safety margin
        safety_margin = 1.15
        while (max_bias > x_max / safety_margin or min_bias < -x_max / safety_margin or 
               max_ubrmse > y_max / safety_margin):
            x_max *= 1.2
            y_max = x_max  # Maintain square aspect
            y_min = -x_max
        
        # Set plot limits
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        
        # Add RMSE circles based on actual plot boundary
        # Circles should extend to the farthest boundary: x-axis, y-axis, or diagonal corner
        max_rmse_plot = max(x_max, y_max, np.sqrt(x_max**2 + y_max**2))
        
        # We want approximately 4-5 circles based on the plot boundary
        target_circles = 4.5
        raw_spacing = max_rmse_plot / target_circles
        
        # Round to nice number using base-10 scaling
        # Get order of magnitude
        if raw_spacing > 0:
            magnitude = 10 ** np.floor(np.log10(raw_spacing))
            normalized = raw_spacing / magnitude
            
            # Choose best multiplier: 1, 2, 5
            if normalized < 1.5:
                circle_spacing = 1 * magnitude
            elif normalized < 3.5:
                circle_spacing = 2 * magnitude
            else:
                circle_spacing = 5 * magnitude
        else:
            circle_spacing = 1
        
        # Generate circles from spacing up to plot boundary
        rmse_circles = []
        rmse = circle_spacing
        while rmse <= max_rmse_plot:
            rmse_circles.append(rmse)
            rmse += circle_spacing
        
        # Calculate the angle to the plot corner (top-right) for label placement
        corner_angle = np.arctan2(y_max, x_max)
        label_cos = np.cos(corner_angle)
        label_sin = np.sin(corner_angle)
        
        for rmse_circle in rmse_circles:
            circle = Circle((0, 0), rmse_circle, fill=False, edgecolor='grey', 
                           linestyle='--', linewidth=1, alpha=0.4)
            ax.add_patch(circle)
            
            # Place label along the angle to the plot corner
            # Label is positioned at (rmse_circle * cos(angle), rmse_circle * sin(angle))
            label_x = rmse_circle * label_cos
            label_y = rmse_circle * label_sin
            
            # Only label if it fits completely within the plot area
            # Check if label position is within plot boundaries
            if label_x <= x_max and label_y <= y_max:
                # Format label nicely
                if circle_spacing >= 1:
                    label_str = f'{rmse_circle:.0f}'
                else:
                    label_str = f'{rmse_circle:.1f}'
                # Add "RMSE = " prefix to the first contour
                if rmse_circle == rmse_circles[0]:
                    label_str = f'RMSE = {label_str}'
                ax.text(label_x, label_y, label_str, 
                        fontsize=12, color='black', alpha=0.8, ha='center', va='center')
        
        ax.set_xlabel('Bias' + (' (normalized)' if normalize else f' {units}'), fontsize=16)
        ax.set_ylabel('Unbiased RMSE' + (' (normalized)' if normalize else f' {units}'), fontsize=16)
        ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold', 
                va='top', ha='left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        
        return ax
    
    # Create the combined figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    sf_left = fig.add_subfigure(gs[0, 0])
    sf_right = fig.add_subfigure(gs[0, 1])
    
    # Create both diagrams
    ax_taylor = _create_taylor(fig, sf_left, stats, labels, colors, markers, obs_std, normalize, units)
    ax_target = _create_target(sf_right, stats, labels, colors, normalize, units)
    
    # Add shared legend
    if colors is None:
        from matplotlib import colormaps
        colors = colormaps['tab10'](np.linspace(0, 1, len(stats)))
    
    handles = [Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=18, linestyle='None')]
    for i, (label, color) in enumerate(zip(labels, colors)):
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=10, markeredgecolor='white', markeredgewidth=1.5))
    
    legend_labels = ['Observations'] + labels
    fig.legend(handles, legend_labels, loc='upper center', ncol=len(handles), 
              bbox_to_anchor=(0.5, 0.98), frameon=True, framealpha=0.9, fontsize=14)
    
    # Save if requested
    if figname:
        fig.savefig(f'{figname}.pdf', dpi=dpi, bbox_inches='tight')
        print(f"Saved figure as {figname}.pdf")
    
    return fig, (ax_taylor, ax_target)


