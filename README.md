# quadratic_sim.py

Simulate a quadratic function with Gaussian noise and outliers, with optional moving statistics and detrending.

## Features

- Generate quadratic data (y = ax² + bx + c) with configurable Gaussian noise
- Add outliers drawn from a separate linear function (simulating contamination)
- Compute moving median and moving mean with configurable window size
- Half-step mode for sparser moving statistic points
- Detrend data by model, moving median, or moving mean
- Two-panel plots showing original and detrended data
- Linear interpolation/extrapolation for detrending at edges

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

No installation required. Just ensure dependencies are available:

```bash
pip install numpy matplotlib
```

## Usage

### Basic Examples

```bash
# Basic plot with default parameters
python quadratic_sim.py -r 42 -o plot.png

# With moving statistics (median and mean)
python quadratic_sim.py -r 42 -M -o plot.png

# With half-stepped moving statistics (sparser points)
python quadratic_sim.py -r 42 -M -H -o plot.png

# With detrending (two-panel plot)
python quadratic_sim.py -r 42 -M -H -d model -o plot.png

# Custom quadratic coefficients (y = 0.5x² + 2x - 3)
python quadratic_sim.py -a 0.5 -b 2 -c -3 -M -o plot.png

# More data points with larger window
python quadratic_sim.py -n 1000 -w 200 -M -H -o plot.png

# Custom outlier linear function (y = 2x + 5)
python quadratic_sim.py -r 42 -M -m 2 -C 5 -o plot.png
```

### CLI Options

```
usage: quadratic_sim.py [-h] [-v] [-a A] [-b B] [-c C] [-n NPOINTS] [-x XMIN]
                        [-X XMAX] [-s NOISE] [-f OUTLIER_FRACTION]
                        [-S OUTLIER_SCALE] [-m OUTLIER_M] [-C OUTLIER_C]
                        [-r SEED] [-M] [-w WINDOW] [-H]
                        [-d {model,median,mean}] [-N] [-o OUTPUT]

Simulate quadratic function with Gaussian noise and outliers

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -a A                  Quadratic coefficient (a*x^2) (default: 1.0)
  -b B                  Linear coefficient (b*x) (default: 0.0)
  -c C                  Constant term (default: 0.0)
  -n NPOINTS, --npoints NPOINTS
                        Number of data points (default: 200)
  -x XMIN, --xmin XMIN  Minimum x value (default: -5.0)
  -X XMAX, --xmax XMAX  Maximum x value (default: 5.0)
  -s NOISE, --noise NOISE
                        Gaussian noise sigma (default: 2.0)
  -f OUTLIER_FRACTION, --outlier-fraction OUTLIER_FRACTION
                        Fraction of outliers (0 to 1) (default: 0.1)
  -S OUTLIER_SCALE, --outlier-scale OUTLIER_SCALE
                        Noise multiplier for outliers (relative to
                        noise_sigma) (default: 2.0)
  -m OUTLIER_M, --outlier-m OUTLIER_M
                        Slope of linear function for outliers (default: 0.0)
  -C OUTLIER_C, --outlier-c OUTLIER_C
                        Intercept of linear function for outliers (default: 10.0)
  -r SEED, --seed SEED  Random seed for reproducibility (default: None)
  -M, --moving          Compute and plot moving median and mean (default: False)
  -w WINDOW, --window WINDOW
                        Window size for moving statistics (default: 20)
  -H, --half-step       Compute moving statistics every window/2 steps (default: False)
  -d {model,median,mean}, --detrend {model,median,mean}
                        Detrend data and show second panel (default: None)
  -N, --no-sort         Skip sorting data by x (default: False)
  -o OUTPUT, --output OUTPUT
                        Output filename for plot (default: None, interactive)
```

### CLI Option Summary

| Short | Long | Description | Default |
|-------|------|-------------|---------|
| `-v` | `--version` | Show version | |
| `-a` | | Quadratic coefficient a | 1.0 |
| `-b` | | Quadratic coefficient b | 0.0 |
| `-c` | | Constant term c | 0.0 |
| `-n` | `--npoints` | Number of data points | 200 |
| `-x` | `--xmin` | Minimum x value | -5.0 |
| `-X` | `--xmax` | Maximum x value | 5.0 |
| `-s` | `--noise` | Gaussian noise sigma | 2.0 |
| `-f` | `--outlier-fraction` | Fraction of outliers | 0.1 |
| `-S` | `--outlier-scale` | Outlier noise multiplier | 2.0 |
| `-m` | `--outlier-m` | Outlier linear slope | 0.0 |
| `-C` | `--outlier-c` | Outlier linear intercept | 10.0 |
| `-r` | `--seed` | Random seed | None |
| `-M` | `--moving` | Enable moving statistics | False |
| `-w` | `--window` | Window size | 20 |
| `-H` | `--half-step` | Half-step mode | False |
| `-d` | `--detrend` | Detrend method | None |
| `-N` | `--no-sort` | Skip sorting | False |
| `-o` | `--output` | Output filename | None |

Note: Uppercase short options are used when lowercase is already taken (e.g., `-M` because `-m` is outlier slope, `-H` because `-h` is help).

## Code Structure

The script (548 lines) is organized into the following functions:

| Lines | Function | Description |
|-------|----------|-------------|
| 1-52 | Module header | Docstring, version, imports |
| 54-113 | `simulate_quadratic()` | Generate y = ax² + bx + c with noise and outliers |
| 116-159 | `preprocess_sort()` | Sort all arrays by x values |
| 162-199 | `moving_statistic()` | Compute moving median or mean |
| 202-262 | `detrend_data()` | Subtract trend with linear extrapolation at edges |
| 265-411 | `plot_simulation()` | Create one or two panel plot |
| 414-474 | `parse_arguments()` | CLI argument parsing |
| 477-547 | `main()` | Orchestrate simulation, stats, and plotting |

### Function Details

#### `simulate_quadratic()`
Generates the quadratic function with Gaussian noise. Outliers are drawn from a separate linear function (y = mx + c) rather than displaced from the quadratic, simulating contamination from a different population.

#### `preprocess_sort()`
Sorts all data arrays by x-values. This is required for proper moving window calculations and line plotting.

#### `moving_statistic()`
Computes moving median or mean over a sliding window. With `half_step=True`, statistics are computed every `window/2` steps instead of at every point, giving sparser output.

#### `detrend_data()`
Subtracts a trend from the data:
- `method='model'`: Subtract the true quadratic function
- `method='median'`: Subtract interpolated moving median
- `method='mean'`: Subtract interpolated moving mean

Uses linear interpolation between moving statistic points and linear extrapolation at the edges (using first/last two points).

#### `plot_simulation()`
Creates the visualization:
- Single panel: Original data with optional moving statistics
- Two panels (with `-d`): Original data + detrended residuals

## Output

The plot shows:
- **Blue dots**: Non-outlier data points
- **Red X markers**: Outliers (from the linear contamination function)
- **Black line**: True quadratic function
- **Green circles + line**: Moving median (robust to outliers)
- **Magenta squares + line**: Moving mean (affected by outliers)

With detrending (`-d`), a second panel shows residuals after subtracting the trend.

## Version History

### v0.2.0 (2025-01)
- Outliers generated from linear function instead of displaced quadratic
- Added `--outlier-m` and `--outlier-c` CLI arguments
- Fixed half-step indexing for correct point counts
- Added detrending with linear extrapolation at edges
- Added single-character CLI options
- Default window size changed to 20

### v0.1.0
- Initial version with quadratic simulation
- Moving median and mean with half-stepping
- Preprocessing sort function
- Elapsed time reporting

## Author

Claude

## License

Public domain
