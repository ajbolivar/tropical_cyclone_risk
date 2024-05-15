#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""
import dask
import datetime
import numpy as np
import os
import xarray as xr
import time
import pandas as pd

import namelist
from intensity import coupled_fast, ocean
from thermo import calc_thermo
from track import env_wind
from wind import tc_wind
from util import basins, input, mat

"""
Driver function to compute zonal and meridional wind monthly mean and
covariances, potential intensity, GPI, and saturation deficit.
"""
def compute_downscaling_inputs():
    data_ts = namelist.data_ts
    print('Computing monthly mean and variance of environmental wind...')
    s = time.time()
    env_wind.gen_wind_mean_cov(data_ts)
    e = time.time()
    print('Time Elapsed: %f s' % (e - s))

    print('Computing thermodynamic variables...')
    s = time.time()
    calc_thermo.gen_thermo()
    e = time.time()
    print('Time Elapsed: %f s' % (e - s))
    
"""
Splits the file containing seed genesis information into smaller files by year.
"""
def create_yearly_files():
    yearS = namelist.start_year
    yearE = namelist.end_year

    try:
        csv = pd.read_csv(namelist.gen_points,
                          usecols=[input.get_trackid_key(),input.get_year_key(),input.get_month_key(),
                                   input.get_day_key(),input.get_hour_key(),input.get_genlon_key(),
                                   input.get_genlat_key()], delimiter=',')
    except:
        csv = pd.read_csv(namelist.gen_points,
                          usecols=[input.get_trackid_key(),input.get_time_key(),
                                   input.get_genlon_key(),input.get_genlat_key()], delimiter=',')

        # Ensure time is in datetime format
        dt_time = pd.to_datetime(csv[input.get_time_key()])
        # Extract year, month, day, and hour from time
        csv[input.get_year_key()] = dt_time.dt.year
        csv[input.get_month_key()] = dt_time.dt.month
        csv[input.get_day_key()] = dt_time.dt.day
        csv[input.get_hour_key()] = dt_time.dt.hour
        # Drop time column before saving
        csv = csv.drop(input.get_time_key(),axis=1)

    # Make sure longitudes are in 0-360 form
    csv[input.get_genlon_key()] = csv[input.get_genlon_key()] % 360
    
    for year in range(yearS,yearE+1):
        if yearS == yearE:
            fn_out = '%s/gen_points_%s_%d%02d_%d%02d.csv' % (namelist.output_directory, namelist.exp_prefix, year,
                                                             namelist.start_month, year, namelist.end_month)
        elif year == yearS:
            fn_out = '%s/gen_points_%s_%d%02d_%d12.csv' % (namelist.output_directory, namelist.exp_prefix, year,
                                                           namelist.start_month, year)
        elif year == yearE:
            fn_out = '%s/gen_points_%s_%d01_%d%02d.csv' % (namelist.output_directory, namelist.exp_prefix, year,
                                                           year, namelist.end_month)
        else:
            fn_out = '%s/gen_points_%s_%d01_%d12.csv' % (namelist.output_directory, namelist.exp_prefix, year, year)
            
        if os.path.exists(fn_out):
            continue
        
        # Keep only the genesis point of each storm
        csv_yr = csv.loc[csv[input.get_year_key()] == year].drop_duplicates(subset=[input.get_trackid_key()])
        csv_yr.to_csv(fn_out)

"""
Returns the name of the file containing downscaled tropical cyclone tracks.
"""
def get_fn_tracks(b):
    fn_args = (namelist.output_directory, namelist.exp_name,
               b.basin_id, namelist.exp_prefix,
               namelist.start_year, namelist.start_month,
               namelist.end_year, namelist.end_month)
    fn_trk = '%s/%s/tracks_%s_%s_%d%02d_%d%02d.nc' % fn_args
    return(fn_trk)

"""
Adds a number to the end of fn_trk if the file exists.
Used when running multiple simulations under the same configuration.
"""
def fn_tracks_duplicates(fn_trk):
    f_int = 0
    fn_trk_out = fn_trk
    while os.path.exists(fn_trk_out):
        fn_trk_out = fn_trk.rstrip('.nc') + '_e%d.nc' % f_int
        f_int += 1
    return fn_trk_out

"""
Generates "n_tracks" number of tropical cyclone tracks, in basin
described by "b" (can be global), in the year.
"""
def run_tracks(year, n_tracks, b, data_ts):
    # Load thermodynamic and ocean variables.
    print(year)
    fn_th = calc_thermo.get_fn_thermo()
    ds = xr.open_dataset(fn_th)
    dt_year_start = datetime.datetime(year-1, 12, 31)
    dt_year_end = datetime.datetime(year, 12, 31)
    dt_bounds = input.convert_from_datetime(ds, [dt_year_start, dt_year_end])
    ds = ds.sel(time=slice(dt_bounds[0], dt_bounds[1])).load()
    lon = ds['lon'].data
    lat = ds['lat'].data
    mld = ocean.mld_climatology(year, basins.TC_Basin('GL'))
    strat = ocean.strat_climatology(year, basins.TC_Basin('GL'))    # Make sure latitude is increasing.
    vpot = ds['vmax'] * namelist.PI_reduc * np.sqrt(namelist.Ck / namelist.Cd)
    rh_mid = ds['rh_mid']
    chi = ds['chi']

    if (lat[0] - lat[1]) > 0:
        vpot = vpot.reindex({'lat': lat[::-1]})
        rh_mid = rh_mid.reindex({'lat': lat[::-1]})
        chi = chi.reindex({'lat': lat[::-1]})
        lat = lat[::-1]

    # Load the basin bounds and genesis points.
    basin_ids = np.array(sorted([k for k in namelist.basin_bounds if k != 'GL']))
    f_basins = {}
    for basin_id in basin_ids:
        ds_b = xr.open_dataset('land/%s.nc' % basin_id)
        basin_mask = ds_b['basin']
        f_basins[basin_id] = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)

    # In case basin is "GL", we load again.
    ds_b = xr.open_dataset('land/%s.nc' % b.basin_id)
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    b_bounds = b.get_bounds()

    # Overwrite n_tracks with number of seeds/year if seeding is set to 'manual'
    if namelist.seeding == 'manual':
        seed_info = pd.read_csv('%s/gen_points_%s_%d01_%d12.csv' % (namelist.output_directory, namelist.exp_prefix, 
                                                                    year, year))
        if seed_info.empty:
            print(f'No seed information found for {year}. Skipping')
            return
            
        lon_check = seed_info[input.get_genlon_key()].between(b_bounds[0], b_bounds[2])
        lat_check = seed_info[input.get_genlat_key()].between(b_bounds[1], b_bounds[3])
        seed_info = seed_info[lon_check & lat_check]
        n_tracks = len(seed_info)
        print(f'AJB: n_tracks == {n_tracks}')
                                  
    # To randomly seed in both space and time, load data for each month in the year.
    T_s = namelist.total_track_time_days * 24 * 60 * 60     # total time to run tracks
    fn_wnd_stat = env_wind.get_env_wnd_fn(data_ts, year)
    ds_wnd = xr.open_dataset(fn_wnd_stat)
    if data_ts == 'monthly':
        cpl_fast = [0] * 12
        m_init_fx = [0] * 12
        n_seeds = np.zeros((len(basin_ids), 12))

        for i in range(12):
            dt_month = datetime.datetime(year, i + 1, 15)
            ds_dt_month = input.convert_from_datetime(ds_wnd, [dt_month])[0]
            vpot_month = np.nan_to_num(vpot.interp(time = ds_dt_month).data, 0)
            rh_mid_month = rh_mid.interp(time = ds_dt_month).data
            chi_month = chi.interp(time = ds_dt_month).data
            chi_month[np.isnan(chi_month)] = 5
            m_init_fx[i] = mat.interp2_fx(lon, lat, rh_mid_month)
            chi_month = np.maximum(np.minimum(np.exp(np.log(chi_month + 1e-3) + namelist.log_chi_fac) + namelist.chi_fac, 5), 1e-5)
            mld_month = mat.interp_2d_grid(mld['lon'], mld['lat'], np.nan_to_num(mld[:, :, i]), lon, lat)
            strat_month = mat.interp_2d_grid(strat['lon'], strat['lat'], np.nan_to_num(strat[:, :, i]), lon, lat)
            cpl_fast[i] = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_month, data_ts,
                                                    namelist.output_interval_s, T_s)
            cpl_fast[i].init_fields(lon, lat, chi_month, vpot_month, mld_month, strat_month)

    if data_ts == '6-hourly':
        cpl_fast = [0] * 1460
        m_init_fx = [0] * 1460
        n_seeds = np.zeros((len(basin_ids), 1460))

        # Creating array of dates to draw from and removing leap day
        start_date = datetime.datetime(year, 1, 1, 0)
        end_date = datetime.datetime(year, 12, 31, 18)
        interval = datetime.timedelta(hours=6)  # 6-hourly time interval

        dates = [start_date + i * interval for i in range(int((end_date - start_date) / interval) + 1)
                 if (((start_date + i * interval).month != 2) or (((start_date + i * interval).day) != 29))]

        for i in range(1460):
            dt_6hr = dates[i]
            month_index = int(dt_6hr.month - 1)
            ds_dt_6hr = input.convert_from_datetime(ds_wnd, [dt_6hr])[0]
            vpot_6hr = np.nan_to_num(vpot.interp(time = ds_dt_6hr).data, 0)
            rh_mid_6hr = rh_mid.interp(time = ds_dt_6hr).data
            chi_6hr = chi.interp(time = ds_dt_6hr).data
            chi_6hr[np.isnan(chi_6hr)] = 5
            m_init_fx[i] = mat.interp2_fx(lon, lat, rh_mid_6hr)
            chi_6hr = np.maximum(np.minimum(np.exp(np.log(chi_6hr + 1e-3) + namelist.log_chi_fac) + namelist.chi_fac, 5), 1e-5)

            mld_month = mat.interp_2d_grid(mld['lon'], mld['lat'], np.nan_to_num(mld[:, :, month_index]), lon, lat)
            strat_month = mat.interp_2d_grid(strat['lon'], strat['lat'], np.nan_to_num(strat[:, :, month_index]), lon, lat)
            cpl_fast[i] = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_6hr, data_ts,
                                                    namelist.output_interval_s, T_s)
            cpl_fast[i].init_fields(lon, lat, chi_6hr, vpot_6hr, mld_month, strat_month)

    # Output vectors.
    nt = 0
    n_steps = cpl_fast[0].total_steps
    tc_lon = np.full((n_tracks, n_steps), np.nan)
    tc_lat = np.full((n_tracks, n_steps), np.nan)
    tc_v = np.full((n_tracks, n_steps), np.nan)
    tc_m = np.full((n_tracks, n_steps), np.nan)
    tc_vmax = np.full((n_tracks, n_steps), np.nan)
    tc_env_wnds = np.full((n_tracks, n_steps, cpl_fast[0].nWLvl), np.nan)
    tc_month = np.full(n_tracks, np.nan)
    tc_basin = np.full(n_tracks, "", dtype = 'U2')
    
    seeds_df_list = []
    tcs_df_list = []

    while nt < n_tracks:
        print(f"Attempting track number = {nt}")
        # Skip seed check for manual seeding
        if namelist.seeding == 'manual':
            print('manual seeding')
            gen_lon = seed_info[input.get_genlon_key()][nt]
            gen_lat = seed_info[input.get_genlat_key()][nt]
            gen_t   = pd.to_datetime('2001' + str(seed_info[input.get_month_key()][nt]) +
                                     str(seed_info[input.get_day_key()][nt]) +
                                     str(seed_info[input.get_hour_key()][nt]),
                                     format='%Y%m%d%H')

            time_seed = int(((gen_t.dayofyear - 1) * 4) + gen_t.hour/6)
            print(gen_lon, gen_lat, gen_t, time_seed)
            fast = cpl_fast[time_seed]

            # Find basin of genesis location and switch H_bl.
            basin_val = np.zeros(len(basin_ids))
            for (b_idx, basin_id) in enumerate(basin_ids):
                basin_val[b_idx] = f_basins[basin_id].ev(gen_lon, gen_lat)
            basin_idx = np.argmax(basin_val)
            
        if namelist.seeding == 'random':
            seed_passed = False
            while not seed_passed:
                # Random genesis location for the seed (weighted by area).
                # Ensure that it is located within the basin and over ocean.
                # Genesis is [3, 45] latitude for each basin.
                lat_min = 3 if np.sign(b_bounds[1]) >= 0 else -45
                lat_max = 45 if np.sign(b_bounds[3]) >= 0 else -3
                y_min = np.sin(np.pi / 180 * lat_min)
                y_max = np.sin(np.pi / 180 * lat_max)
                gen_lon = np.random.uniform(b_bounds[0], b_bounds[2], 1)[0]
                gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1)[0]) * 180 / np.pi
                while f_b.ev(gen_lon, gen_lat) < 1e-2:
                    gen_lon = np.random.uniform(b_bounds[0], b_bounds[2], 1)[0]
                    gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1)[0]) * 180 / np.pi
                
                if data_ts == 'monthly':
                    # Randomly seed the month.
                    time_seed = np.random.randint(1, 13)
                if data_ts == '6-hourly':
                    # Randomly seed the 6-hour timestep.
                    time_seed = np.random.randint(1,1461)
    
                fast = cpl_fast[time_seed - 1]
    
                # Find basin of genesis location and switch H_bl.
                basin_val = np.zeros(len(basin_ids))
                for (b_idx, basin_id) in enumerate(basin_ids):
                    basin_val[b_idx] = f_basins[basin_id].ev(gen_lon, gen_lat)
                basin_idx = np.argmax(basin_val)
    
                # Discard seeds with increasing probability equatorwards.
                # If PI is less than 35 m/s, do not integrate, but treat as a seed.
                pi_gen = float(fast.f_vpot.ev(gen_lon, gen_lat))
                lat_vort_power = namelist.lat_vort_power[basin_ids[basin_idx]]
                prob_lowlat = np.power(np.minimum(np.maximum((np.abs(gen_lat) - namelist.lat_vort_fac) / 12.0, 0), 1), lat_vort_power)
                rand_lowlat = np.random.uniform(0, 1, 1)[0]
                seed_df = pd.DataFrame([[gen_lat,gen_lon,time_seed,year]],columns=['lat','lon','month','year'])
                seeds_df_list.append(seed_df)
                if (np.nanmax(basin_val) > 1e-3) and (rand_lowlat < prob_lowlat):
                    n_seeds[basin_idx, time_seed-1] += 1
                    if (pi_gen > 35):
                        seed_passed = True

        # Set the initial value of m to a function of relative humidity.
        v_init = namelist.seed_v_init_ms + np.random.randn(1)[0]
        rh_init = float(m_init_fx[time_seed-1].ev(gen_lon, gen_lat))
        rh_init = float(m_init_fx[time_seed-1].ev(gen_lon, gen_lat))
        m_init = np.maximum(0, namelist.f_mInit(rh_init))
        fast.h_bl = namelist.atm_bl_depth[basin_ids[basin_idx]]
        res = fast.gen_track(gen_lon, gen_lat, v_init, m_init, data_ts)

        is_tc = False
        if res != None:
            track_lon = res.y[0]
            track_lat = res.y[1]
            v_track = res.y[2]
            m_track = res.y[3]

            # If the TC has not reached the threshold m/s after 2 days, throw it away.
            # The TC must also reach the genesis threshold during it's entire lifetime.
            v_thresh = namelist.seed_v_threshold_ms
            v_thresh_2d = np.interp(2*24*60*60, res.t, v_track.flatten())
            is_tc = np.logical_and(np.any(v_track >= v_thresh), v_thresh_2d >= namelist.seed_v_2d_threshold_ms)

        # Skip TC threshold check for manual seeding
        # No stochastic wind generation, so track integration will be identical for repeated attempts
        if namelist.seeding == 'manual': is_tc = True
        if is_tc:
            n_time = len(track_lon)
            tc_lon[nt, 0:n_time] = track_lon
            tc_lat[nt, 0:n_time] = track_lat
            tc_v[nt, 0:n_time] = v_track
            tc_m[nt, 0:n_time] = m_track

            # Redudant calculation, but since environmental winds are not part
            # of the time-integrated state (a parameter), we recompute it.
            # TODO: Remove this redudancy by pre-caclulating the env. wind.
            for i in range(len(track_lon)):
                tc_env_wnds[nt, i, :] = fast._env_winds(track_lon[i], track_lat[i], fast.t_s[i], data_ts)     
            vmax = tc_wind.axi_to_max_wind(track_lon, track_lat, fast.dt_track,
                                           v_track, tc_env_wnds[nt, 0:n_time, :])
            if np.nanmax(vmax) >= namelist.seed_vmax_threshold_ms:
                tc_vmax[nt, 0:n_time] = vmax
                tc_month[nt] = time_seed
                tc_basin[nt] = basin_ids[basin_idx]
                nt += 1

        tcs_df = pd.DataFrame([[gen_lat,gen_lon,time_seed,year]],columns=['lat','lon','month','year'])
        tcs_df_list.append(tcs_df)

    # If no spatial seed info retained, create empty DataFrame
    if not seeds_df_list: seed_tries = pd.DataFrame(columns=['lat','lon','month','year'])
    else: seed_tries = pd.concat(seeds_df_list)
    # This shouldn't trigger, but just in case.
    # If no spatial TC attempt info retained, create empty DataFrame
    if not tcs_df_list: tc_tries = pd.DataFrame(columns=['lat','lon','month','year'])
    else: tc_tries = pd.concat(tcs_df_list)

    return((tc_lon, tc_lat, tc_v, tc_m, tc_vmax, tc_env_wnds, tc_month, tc_basin, n_seeds, seed_tries, tc_tries))

"""
Runs the downscaling model in basin "basin_id" according to the
settings in the namelist.txt file.
"""
def run_downscaling(basin_id):
    data_ts = namelist.data_ts
    n_tracks = namelist.tracks_per_year   # number of tracks per year
    n_procs = namelist.n_procs
    b = basins.TC_Basin(basin_id)
    yearS = namelist.start_year
    yearE = namelist.end_year
    
    if (namelist.seeding == 'manual') & (namelist.data_ts == 'monthly'):
        print('Error: manual seeding only supported for 6-hourly data!')
        return
        
    lazy_results = []; f_args = [];
    for yr in range(yearS, yearE+1):
        lazy_result = dask.delayed(run_tracks)(yr, n_tracks, b, data_ts)
        f_args.append((yr, n_tracks, b))
        lazy_results.append(lazy_result)

    s = time.time()
    out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = n_procs)

    # Process the output and save as a netCDF file.
    tc_lon = np.concatenate([x[0] for x in out], axis = 0)
    tc_lat = np.concatenate([x[1] for x in out], axis = 0)
    tc_v = np.concatenate([x[2] for x in out], axis = 0)
    tc_m = np.concatenate([x[3] for x in out], axis = 0)
    tc_vmax = np.concatenate([x[4] for x in out], axis = 0)
    tc_env_wnds = np.concatenate([x[5] for x in out], axis = 0)
    tc_months = np.concatenate([x[6] for x in out], axis = 0)
    tc_basins = np.concatenate([x[7] for x in out], axis = 0)
    tc_years = np.concatenate([[i+yearS]*out[i][0].shape[0] for i in range(len(out))], axis = 0)
    n_seeds = np.array([x[8] for x in out])

    seed_tries = pd.concat([x[9] for x in out], axis = 0)
    tc_tries = pd.concat([x[10] for x in out], axis = 0)

    total_time_s = namelist.total_track_time_days*24*60*60
    n_steps_output = int(total_time_s / namelist.output_interval_s) + 1
    ts_output = np.linspace(0, total_time_s, n_steps_output)
    yr_trks = np.stack([[x[0]] for x in f_args]).flatten()
    basin_ids = sorted([k for k in namelist.basin_bounds if k != 'GL'])

    if data_ts == "monthly":
        name = "month"
        t = list(range(1,13))
    if data_ts == "6-hourly":
        name = "timestep"
        t = list(range(1,1461))

    ds = xr.Dataset(data_vars = dict(lon_trks = (["n_trk", "time"], tc_lon),
                                     lat_trks = (["n_trk", "time"], tc_lat),
                                     u250_trks = (["n_trk", "time"], tc_env_wnds[:, :, 0]),
                                     v250_trks = (["n_trk", "time"], tc_env_wnds[:, :, 1]),
                                     u850_trks = (["n_trk", "time"], tc_env_wnds[:, :, 2]),
                                     v850_trks = (["n_trk", "time"], tc_env_wnds[:, :, 3]),
                                     v_trks = (["n_trk", "time"], tc_v),
                                     m_trks = (["n_trk", "time"], tc_m),
                                     vmax_trks = (["n_trk", "time"], tc_vmax),
                                     tc_month = (["n_trk"], tc_months),
                                     tc_basins = (["n_trk"], tc_basins),                                     
                                     tc_years = (["n_trk"], tc_years),
                                     seeds_per_month = (["year", "basin", name], n_seeds)),
                    coords = {'n_trk': range(tc_lon.shape[0]), 'time': ts_output,
                              'year': yr_trks, 'basin': basin_ids, name: t})
 
    os.makedirs('%s/%s' % (namelist.base_directory, namelist.exp_name), exist_ok = True)
    fn_trk_out = fn_tracks_duplicates(get_fn_tracks(b))
    seed_tries.to_csv('%s/%s/seed_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix), mode = 'w')
    tc_tries.to_csv('%s/%s/tc_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix), mode = 'w')
    ds.to_netcdf(fn_trk_out, mode = 'w')
    print('Saved %s' % fn_trk_out)
    print(time.time() - s)
