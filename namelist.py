import os
import numpy as np
"""
Namelist file that serves as the configuration file for the TC-risk model.
"""
########################## File System Parameters ###########################
src_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = '/glade/derecho/scratch/abolivar/tc_risk/input/ERA5/sub-monthly/1deg'
output_directory = '/glade/campaign/univ/upsu0068/output/ERA5/6-hourly/second_run/vr/sub-monthly_thermo/5d_rm_0.8_v2'
exp_name = '1980-2021_Vbar_Spoint_med'
debug = True
# For now, we support either 'GCM' or 'ERA5'. Different file types and variable
# names can be added by modifying the "input.py" file and adding the appropriate
# variable key words in the structure var_keys.
dataset_type = 'ERA5' # 'GCM'
exp_prefix = 'era5'

# Variable naming based on dataset_type.
# 'sst' is sea-surface temperature (monthly-averaged)
# 'mslp' is mean sea-level pressure (monthly-averaged)
# 'temp' is temperature (monthly-averaged)
# 'sp_hum' is specific humidity (monthly-averaged)
# 'u' is zonal wind (daily)
# 'v' is meridional wind (daily)
var_keys = {'ERA5': {'sst': 'SSTK', 'mslp': 'MSL', 'temp': 'T',
                     'sp_hum': 'Q', 'u': 'U', 'v': 'V',
                     'lvl': 'level', 'lon': 'lon', 'lat': 'lat'},
            'GCM': {'sst': 'tas', 'mslp': 'psl', 'temp': 'ta',
                    'sp_hum': 'hus', 'u': 'ua', 'v': 'va',
                    'lvl': 'plev', 'lon': 'lon', 'lat': 'lat'}}

########################### Data Input Parameters ###########################
wind_ts = '6-hourly'     # timestep of input wind data, 'monthly' or '6-hourly'
thermo_ts = 'sub-monthly' # timestep of input thermo data, 'monthly' or 'sub-monthly'
window = 5
seeding = 'random'       # method of storm seeding, 'random' or 'manual'
# file containing genesis points (lon/lat/date/time)
gen_points = '/glade/work/abolivar/track_data/trajectories_ERA5_2005sample.csv'
# Variable naming for the above file. if 'time' exists, it will override 'year',
# 'month','day', and 'hour' when accessing the file.
gen_var_keys = {'trackid': 'sid', 'time': 'time', 'year': 'year', 
                'month': 'month', 'day': 'day', 'hour': 'hour', 'lon': 'lon', 
                'lat': 'lat', 'wind': 'wind'}

########################### Parallelism Parameters ##########################
n_procs = 1             # number of processes to use in dask
gnu_parallel = True    # when True, uses GNU parallel for parallelization

############################ TC Risk Parameters #############################
"""
These parameters configure the dates for the TC-risk model.
"""
start_year = 1980
start_month = 1                       # month of start_year to start downscaling
end_year = 2021
end_month = 12                        # month of end_year to stop downscaling

"""
These parameters configure the output.
"""
output_interval_s = 3600              # output interval of tracks, seconds (does not change time integration)
total_track_time_days = 15            # total time to integrate tracks, days
tracks_per_year = 100                 # total number of tracks to simulate per year (will be overwritten if seeding is set to 'manual')

"""
These parameters configure thermodynamics and thermodynamic constants.
"""
p_midlevel = 60000
PI_reduc = 0.9
Ck = 1.2e-3
Cd = 1.2e-3
select_thermo = 1   # 1 for pseudoadiabatic, 2 for reversible thermodynamics
select_interp = 2   # 1 for computation, 2 for interpolation
chi_radius = False
wind_radius = True
rwind = 1000

"""
These parameters configure track and intensity constants.
"""
# Defines the steering levels (hPa) of the storm (see paper)
# If 250- and 850-hPa, uses two steering levels.
# If 250-, 500-, and 850-hPa, uses three steering levels.
# The steering_coefficients ('steering_coefs') should have the same 
# length as the number of levels.
steering_levels = [500, 850]
steering_coefs = [0.2, 0.8]           # constant steering coefficients if not coupled
shear_levels = [250, 850]
coupled_track = False                 # track coupled to intensity; overrides alpha
y_alpha = [0.17, 0.83]                # value of steering coefficient at 0 knots
m_alpha = [0.0025, -0.0025]           # change of each coefficient per unit storm intensity, 1 / kts
alpha_max = [0.41, 0.78]              # maximum value of each steering coefficient (coupled track only)
alpha_min = [0.22, 0.59]              # minimum value of each steering coefficient (coupled track only)
u_beta = -1                           # zonal beta drift, m/s
v_beta = 2.5                          # meridional beta drift, m/s
T_days = 20                           # period of the fourier series, days
seed_v_init_ms = 5                    # initial seed v intensity, m/s
seed_v_2d_threshold_ms = 6.5          # seed v threshold after 2 days, m/s
seed_v_threshold_ms = 15              # seed v threshold over entire lifetime, m/s
seed_vmax_threshold_ms = 18           # seed vmax threshold over entire lifetime, m/s
# Atmospheric boundary layer depth (FAST), m
atm_bl_depth = {'NA': 1400.0, 'EP': 1400.0, 'WP': 1800.0, 'AU': 1800.0,
                'SI': 1600.0, 'SP': 2000.0, 'NI': 1500.0}
log_chi_fac = 0.5                     # addition to chi in log space
chi_fac = 1.3                         # addition to chi
lat_vort_fac = 2                      # sets where vorticity threshold decays toward equator
lat_vort_power = {'NA': 6, 'EP': 6,   # power decay towards the equator
                  'WP': 3.5, 'AU': 6,
                  'SI': 3, 'SP': 7, 'NI': 2.5}
# Initial m based on large-scale relative humidity
f_mInit = lambda rh: 0.20 / (1 + np.exp(-(rh - 0.55) * 10)) + 0.125

"""
Basins for which the model is enabled.
The basin bounds dictionary maps a basin identifier to the basin boundaries.
The bounds are ordered as (LL - Lower Left, UR - Upper Right):
[LL Longitude, LL Latitude, UR Longitude, UR Latitude].
Note the basins bounds are extended slightly to allow tracks to
extend slightly beyond the bounds, as this may be the case when a TC starts
in one basin and goes to another.
Identifiers: EP - Eastern Pacific
             NA - North Atlantic
             NI - North Indian
             SI - South Indian
             SP - South Pacific
             WP - Western Pacific
             GL - Global (no basin)
"""
basin_bounds = {'EP': ['180E', '0N', '290E', '60N'],
                'NA': ['260E', '0N', '360E', '60N'],
                'NI': ['30E', '0N', '100E', '50N'],
                'SI': ['20E', '45S', '100E', '0S'],
                'AU': ['100E', '45S', '180E', '0S'],
                'SP': ['180E', '45S', '250E', '0S'],
                'WP': ['100E', '0N', '180E', '60N'],
                'GL': ['0E', '90S', '360E', '90N']}

