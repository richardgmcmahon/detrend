#!/usr/bin/env python
"""
Simulate a quadratic function with Gaussian noise and outliers.

Optionally compute and overplot moving median and moving mean.

Examples
--------
Basic usage with default parameters:
    python quadratic_sim.py --seed 42 -o plot.png

With moving statistics (median and mean):
    python quadratic_sim.py --seed 42 --moving -o plot.png

With half-stepped moving statistics (sparser points):
    python quadratic_sim.py --seed 42 --moving --half-step -o plot.png

Custom quadratic coefficients (y = 0.5x^2 + 2x - 3):
    python quadratic_sim.py -a 0.5 -b 2 -c -3 --moving -o plot.png

More data points with larger window:
    python quadratic_sim.py -n 1000 --window 200 --moving --half-step -o plot.png

Custom outlier linear function (y = 2x + 5):
    python quadratic_sim.py --seed 42 --moving --outlier-m 2 --outlier-c 5 -o plot.png

Changes
-------
v0.2.0 (2025-01):
    - Outliers now generated from a linear function (y = m*x + c) instead of 
      displaced quadratic points, simulating contamination from a different 
      population
    - Added --outlier-m and --outlier-c CLI arguments
    - Fixed half-step indexing to produce correct number of points 
      (e.g., 9 points for n=1000, window=200)
    - Aligned function defaults with CLI defaults

v0.1.0:
    - Initial version with quadratic simulation, Gaussian noise, and outliers
    - Moving median and mean with optional half-stepping
    - Preprocessing sort function
    - Elapsed time reporting
"""

__version__ = "0.2.0"
__author__ = "Claude"

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


def simulate_quadratic(x, a=1.0, b=0.0, c=0.0, noise_sigma=1.0, 
                       outlier_fraction=0.1, outlier_scale=5.0, seed=None,
                       outlier_linear_m=0.0, outlier_linear_c=10.0):
    """
    Generate y = a*x^2 + b*x + c with Gaussian noise and outliers.
    
    Outliers are drawn from a linear function y = m*x + c with Gaussian noise,
    representing contamination from a different population.
    
    Parameters
    ----------
    x : array-like
        Input x values
    a, b, c : float
        Quadratic coefficients (y = ax^2 + bx + c)
    noise_sigma : float
        Standard deviation of Gaussian noise
    outlier_fraction : float
        Fraction of points to make outliers (0 to 1)
    outlier_scale : float
        Noise multiplier for outliers (relative to noise_sigma)
    seed : int or None
        Random seed for reproducibility
    outlier_linear_m : float
        Slope of the linear function for outliers
    outlier_linear_c : float
        Intercept of the linear function for outliers
        
    Returns
    -------
    y_true : ndarray
        True quadratic values (no noise)
    y_noisy : ndarray
        Noisy values with outliers
    outlier_mask : ndarray
        Boolean mask indicating which points are outliers
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.asarray(x)
    y_true = a * x**2 + b * x + c
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_sigma, size=x.shape)
    y_noisy = y_true + noise
    
    # Select outlier indices
    n_outliers = int(len(x) * outlier_fraction)
    outlier_indices = np.random.choice(len(x), size=n_outliers, replace=False)
    outlier_mask = np.zeros(len(x), dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Generate outliers from a linear function with noise
    x_outliers = x[outlier_mask]
    y_outlier_linear = outlier_linear_m * x_outliers + outlier_linear_c
    outlier_noise = np.random.normal(0, noise_sigma * outlier_scale, size=n_outliers)
    y_noisy[outlier_mask] = y_outlier_linear + outlier_noise
    
    return y_true, y_noisy, outlier_mask


def preprocess_sort(x, y_true=None, y_noisy=None, outlier_mask=None, sort=True):
    """
    Sort all arrays by x values.
    
    Parameters
    ----------
    x : array-like
        X values (independent variable)
    y_true : array-like or None
        True function values
    y_noisy : array-like or None
        Noisy data values
    outlier_mask : array-like or None
        Boolean outlier mask
    sort : bool
        If True, sort arrays by x. If False, return copies unchanged.
        
    Returns
    -------
    x_out : ndarray
        Sorted (or unsorted) x values
    y_true_out : ndarray or None
        Sorted y_true (or None if input was None)
    y_noisy_out : ndarray or None
        Sorted y_noisy (or None if input was None)
    outlier_mask_out : ndarray or None
        Sorted outlier_mask (or None if input was None)
    sort_idx : ndarray
        Indices used for sorting (useful for back-transformation)
    """
    x = np.asarray(x)
    
    if sort:
        sort_idx = np.argsort(x)
    else:
        sort_idx = np.arange(len(x))
    
    x_out = x[sort_idx]
    
    y_true_out = None if y_true is None else np.asarray(y_true)[sort_idx]
    y_noisy_out = None if y_noisy is None else np.asarray(y_noisy)[sort_idx]
    outlier_mask_out = None if outlier_mask is None else np.asarray(outlier_mask)[sort_idx]
    
    return x_out, y_true_out, y_noisy_out, outlier_mask_out, sort_idx


def moving_statistic(y, window_size=15, statistic='median', half_step=False):
    """
    Compute moving median or mean.
    
    Parameters
    ----------
    y : array-like
        Input values
    window_size : int
        Window size (should be odd for symmetric window)
    statistic : str
        Either 'median' or 'mean'
    half_step : bool
        If True, compute statistic every window_size/2 steps instead of every step
        
    Returns
    -------
    x_indices : ndarray
        Indices at which statistics are computed
    result : ndarray
        Moving statistic values at those indices
    """
    y = np.asarray(y)
    n = len(y)
    
    half_window = window_size // 2
    stat_func = np.nanmedian if statistic == 'median' else np.nanmean
    
    if half_step:
        step = max(1, window_size // 2)
        indices = list(range(half_window - 1, n - half_window, step))
    else:
        indices = list(range(half_window, n - half_window))
    
    result = np.array([stat_func(y[max(0, i - half_window) : i + half_window + 1]) 
                       for i in indices])
    
    return np.array(indices), result


def detrend_data(x, y, method='model', y_true=None, x_moving=None, moving_values=None):
    """
    Detrend data by subtracting a trend.
    
    Parameters
    ----------
    x : array-like
        X values
    y : array-like
        Y values to detrend
    method : str
        Detrending method: 'model' (subtract y_true), 'median', or 'mean'
        (subtract interpolated moving statistic)
    y_true : array-like or None
        True model values (required if method='model')
    x_moving : array-like or None
        X values for moving statistic points (required if method='median' or 'mean')
    moving_values : array-like or None
        Moving statistic values (required if method='median' or 'mean')
        
    Returns
    -------
    y_detrended : ndarray
        Detrended y values
    trend : ndarray
        The trend that was subtracted
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if method == 'model':
        if y_true is None:
            raise ValueError("y_true required for method='model'")
        trend = np.asarray(y_true)
    elif method in ('median', 'mean'):
        if x_moving is None or moving_values is None:
            raise ValueError(f"x_moving and moving_values required for method='{method}'")
        x_moving = np.asarray(x_moving)
        moving_values = np.asarray(moving_values)
        
        # Linear interpolation in the middle (np.interp)
        trend = np.interp(x, x_moving, moving_values)
        
        # Linear extrapolation at edges using first/last two points
        if len(x_moving) >= 2:
            # Left edge: extrapolate using first two points
            left_mask = x < x_moving[0]
            if np.any(left_mask):
                slope_left = (moving_values[1] - moving_values[0]) / (x_moving[1] - x_moving[0])
                trend[left_mask] = moving_values[0] + slope_left * (x[left_mask] - x_moving[0])
            
            # Right edge: extrapolate using last two points
            right_mask = x > x_moving[-1]
            if np.any(right_mask):
                slope_right = (moving_values[-1] - moving_values[-2]) / (x_moving[-1] - x_moving[-2])
                trend[right_mask] = moving_values[-1] + slope_right * (x[right_mask] - x_moving[-1])
    else:
        raise ValueError(f"Unknown detrend method: {method}")
    
    y_detrended = y - trend
    return y_detrended, trend


def plot_simulation(x, y_true, y_noisy, outlier_mask, 
                    x_med=None, moving_med=None, 
                    x_avg=None, moving_avg=None,
                    window_size=15, half_step=False, output_file=None,
                    detrend=None):
    """
    Plot the simulated data with optional moving statistics and detrending.
    
    Parameters
    ----------
    x : array-like
        X values for data points
    y_true : array-like
        True quadratic values
    y_noisy : array-like
        Noisy data values
    outlier_mask : array-like
        Boolean mask for outliers
    x_med : array-like or None
        X values for moving median points
    moving_med : array-like or None
        Moving median values
    x_avg : array-like or None
        X values for moving mean points
    moving_avg : array-like or None
        Moving mean values
    window_size : int
        Window size used (for labeling)
    half_step : bool
        Whether half-stepping was used (for labeling)
    output_file : str or None
        Output filename, or None for interactive display
    detrend : str or None
        Detrending method: 'model', 'median', 'mean', or None
    """
    # Determine if we need two panels
    if detrend:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None
    
    # Count data points and outliers
    n_data = np.sum(~outlier_mask)
    n_outliers = np.sum(outlier_mask)
    
    # ---- Top panel: Original data ----
    ax1.scatter(x[~outlier_mask], y_noisy[~outlier_mask], 
                alpha=0.6, s=20, label=f'Data (n={n_data})', color='steelblue')
    ax1.scatter(x[outlier_mask], y_noisy[outlier_mask], 
                alpha=0.8, s=40, marker='x', label=f'Outliers (n={n_outliers})', color='red')
    ax1.plot(x, y_true, 'k-', lw=2, label='True quadratic')
    
    step_label = ', half-step' if half_step else ''
    
    if moving_med is not None and x_med is not None:
        label = f'Moving median (w={window_size}{step_label})'
        ax1.plot(x_med, moving_med, color='green', linestyle='-', linewidth=2,
                 marker='o', markerfacecolor='none', markeredgewidth=2, markersize=8,
                 label=label, zorder=10)
    
    if moving_avg is not None and x_avg is not None:
        label = f'Moving mean (w={window_size}{step_label})'
        ax1.plot(x_avg, moving_avg, color='magenta', linestyle='-', linewidth=2,
                 marker='s', markerfacecolor='none', markeredgewidth=2, markersize=8,
                 label=label, zorder=10)
    
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.set_title('Quadratic with Gaussian noise and outliers')
    
    # ---- Bottom panel: Detrended data ----
    if detrend and ax2 is not None:
        # Compute detrended data
        if detrend == 'model':
            y_detrended, trend = detrend_data(x, y_noisy, method='model', y_true=y_true)
            detrend_label = 'model'
        elif detrend == 'median':
            if x_med is None or moving_med is None:
                raise ValueError("Moving median required for detrend='median'. Use -M flag.")
            y_detrended, trend = detrend_data(x, y_noisy, method='median', 
                                               x_moving=x_med, moving_values=moving_med)
            detrend_label = 'moving median'
        elif detrend == 'mean':
            if x_avg is None or moving_avg is None:
                raise ValueError("Moving mean required for detrend='mean'. Use -M flag.")
            y_detrended, trend = detrend_data(x, y_noisy, method='mean',
                                               x_moving=x_avg, moving_values=moving_avg)
            detrend_label = 'moving mean'
        else:
            raise ValueError(f"Unknown detrend method: {detrend}")
        
        # Plot detrended data
        ax2.scatter(x[~outlier_mask], y_detrended[~outlier_mask], 
                    alpha=0.6, s=20, label=f'Detrended data (n={n_data})', color='steelblue')
        ax2.scatter(x[outlier_mask], y_detrended[outlier_mask], 
                    alpha=0.8, s=40, marker='x', label=f'Outliers (n={n_outliers})', color='red')
        
        # Plot zero line
        ax2.axhline(y=0, color='k', linestyle='-', lw=2, label='Zero')
        
        # Compute and plot detrended moving statistics
        if moving_med is not None and x_med is not None:
            if detrend == 'model':
                # Interpolate model to moving stat x positions
                trend_at_med = np.interp(x_med, x, y_true)
                moving_med_detrended = moving_med - trend_at_med
            elif detrend == 'median':
                moving_med_detrended = moving_med - moving_med  # Zero by definition
            else:  # mean
                trend_at_med = np.interp(x_med, x_avg, moving_avg)
                moving_med_detrended = moving_med - trend_at_med
            
            label = f'Moving median (w={window_size}{step_label})'
            ax2.plot(x_med, moving_med_detrended, color='green', linestyle='-', linewidth=2,
                     marker='o', markerfacecolor='none', markeredgewidth=2, markersize=8,
                     label=label, zorder=10)
        
        if moving_avg is not None and x_avg is not None:
            if detrend == 'model':
                trend_at_avg = np.interp(x_avg, x, y_true)
                moving_avg_detrended = moving_avg - trend_at_avg
            elif detrend == 'mean':
                moving_avg_detrended = moving_avg - moving_avg  # Zero by definition
            else:  # median
                trend_at_avg = np.interp(x_avg, x_med, moving_med)
                moving_avg_detrended = moving_avg - trend_at_avg
            
            label = f'Moving mean (w={window_size}{step_label})'
            ax2.plot(x_avg, moving_avg_detrended, color='magenta', linestyle='-', linewidth=2,
                     marker='s', markerfacecolor='none', markeredgewidth=2, markersize=8,
                     label=label, zorder=10)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y - trend')
        ax2.legend()
        ax2.set_title(f'Detrended by {detrend_label}')
    else:
        ax1.set_xlabel('x')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Simulate quadratic function with Gaussian noise and outliers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    
    # Quadratic coefficients
    parser.add_argument('-a', type=float, default=1.0,
                        help='Quadratic coefficient (a*x^2)')
    parser.add_argument('-b', type=float, default=0.0,
                        help='Linear coefficient (b*x)')
    parser.add_argument('-c', type=float, default=0.0,
                        help='Constant term')
    
    # Data generation
    parser.add_argument('-n', '--npoints', type=int, default=200,
                        help='Number of data points')
    parser.add_argument('-x', '--xmin', type=float, default=-5.0,
                        help='Minimum x value')
    parser.add_argument('-X', '--xmax', type=float, default=5.0,
                        help='Maximum x value')
    parser.add_argument('-s', '--noise', type=float, default=2.0,
                        help='Gaussian noise sigma')
    parser.add_argument('-f', '--outlier-fraction', type=float, default=0.1,
                        help='Fraction of outliers (0 to 1)')
    parser.add_argument('-S', '--outlier-scale', type=float, default=2.0,
                        help='Noise multiplier for outliers (relative to noise_sigma)')
    parser.add_argument('-m', '--outlier-m', type=float, default=0.0,
                        help='Slope of linear function for outliers')
    parser.add_argument('-C', '--outlier-c', type=float, default=10.0,
                        help='Intercept of linear function for outliers (uppercase because -c is taken)')
    parser.add_argument('-r', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Moving statistics
    parser.add_argument('-M', '--moving', action='store_true',
                        help='Compute and plot moving median and mean (uppercase because -m is taken)')
    parser.add_argument('-w', '--window', type=int, default=20,
                        help='Window size for moving statistics')
    parser.add_argument('-H', '--half-step', action='store_true',
                        help='Compute moving statistics every window/2 steps (uppercase because -h is help)')
    parser.add_argument('-d', '--detrend', type=str, default=None, choices=['model', 'median', 'mean'],
                        help='Detrend data and show second panel: model, median, or mean')
    parser.add_argument('-N', '--no-sort', action='store_true',
                        help='Skip sorting data by x (uppercase because -n is taken)')
    
    # Output
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename for plot (if not set, displays interactively)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    t_start = time.perf_counter()
    
    # Generate x values
    x = np.linspace(args.xmin, args.xmax, args.npoints)
    
    # Simulate data
    t_sim_start = time.perf_counter()
    y_true, y_noisy, outlier_mask = simulate_quadratic(
        x, 
        a=args.a, b=args.b, c=args.c,
        noise_sigma=args.noise,
        outlier_fraction=args.outlier_fraction,
        outlier_scale=args.outlier_scale,
        seed=args.seed,
        outlier_linear_m=args.outlier_m,
        outlier_linear_c=args.outlier_c
    )
    t_sim_end = time.perf_counter()
    print(f"Simulation: {t_sim_end - t_sim_start:.4f} s")
    
    # Preprocess: sort by x (default) or skip sorting
    t_sort_start = time.perf_counter()
    do_sort = not args.no_sort
    x, y_true, y_noisy, outlier_mask, sort_idx = preprocess_sort(
        x, y_true, y_noisy, outlier_mask, sort=do_sort
    )
    t_sort_end = time.perf_counter()
    print(f"Preprocessing (sort={do_sort}): {t_sort_end - t_sort_start:.4f} s")
    
    # Compute moving statistics if requested
    x_med, moving_med = None, None
    x_avg, moving_avg = None, None
    
    if args.moving:
        t_moving_start = time.perf_counter()
        
        indices_med, moving_med = moving_statistic(
            y_noisy, args.window, 'median', half_step=args.half_step
        )
        indices_avg, moving_avg = moving_statistic(
            y_noisy, args.window, 'mean', half_step=args.half_step
        )
        
        # Convert indices to x values
        x_med = x[indices_med]
        x_avg = x[indices_avg]
        
        t_moving_end = time.perf_counter()
        print(f"Moving statistics: {t_moving_end - t_moving_start:.4f} s")
    
    # Plot
    t_plot_start = time.perf_counter()
    plot_simulation(x, y_true, y_noisy, outlier_mask,
                    x_med=x_med, moving_med=moving_med,
                    x_avg=x_avg, moving_avg=moving_avg,
                    window_size=args.window,
                    half_step=args.half_step,
                    output_file=args.output,
                    detrend=args.detrend)
    t_plot_end = time.perf_counter()
    print(f"Plotting: {t_plot_end - t_plot_start:.4f} s")
    
    t_end = time.perf_counter()
    print(f"Total elapsed: {t_end - t_start:.4f} s")


if __name__ == '__main__':
    main()
