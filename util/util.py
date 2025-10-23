import os
import subprocess
import namelist
import numpy as np
import xarray as xr
import time
from util import constants
from scipy import interpolate
from scipy.ndimage import generic_filter
import warnings
warnings.filterwarnings('ignore')
"""
Inverse transform sampling.
"""
def inv_trans_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


"""
Seed the generator. Advantage of this method is that processes that
run close to each other will have very different seeds.
"""
def random_seed():
    t = int(time.time() * 1000.0)
    np.random.seed(((t & 0xff000000) >> 24) +
                   ((t & 0x00ff0000) >>  8) +
                   ((t & 0x0000ff00) <<  8) +
                   ((t & 0x000000ff) << 24))

def map_to_fx(source_idx, fxs):
    if source_idx > len(fxs):
        raise ValueError('Source index is not valid. See namelist configuration.')
    else:
        return(fxs[source_idx])

def haversine(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return (constants.earth_R / 1000) * c  # distance in km

def circular_footprint(radius_cells):
    y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
    mask = x**2 + y**2 <= radius_cells**2
    return mask.astype(int)

def nanpercentile_filter(arr, percentile=90):
    # arr is flattened footprint
    return np.nanpercentile(arr, percentile)

def apply_nanpercentile_filter(data, footprint):
    return generic_filter(
        data,
        function=lambda arr: nanpercentile_filter(arr, percentile=90),
        footprint=footprint,
        mode='constant',  # constant mode avoids wrapping edges
        cval=np.nan       # fill edges with NaN, will be ignored
    )

def radial_percentile(lat, lon, data, basin='GL'):
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    grid_dx_km = haversine(lat0, lon0, lat0, lon0 + data.lon.diff('lon').mean().item())
    grid_dy_km = haversine(lat0, lon0, lat0 + data.lat.diff('lat').mean().item(), lon0)
    
    r_lat = int(namelist.rchi / grid_dy_km)
    r_lon = int(namelist.rchi / grid_dx_km)
    
    land_mask = xr.open_dataset(f'land/{basin}.nc')
    land_mask = land_mask.interp(
            lat=data.lat,
            lon=data.lon,
            method='nearest'
            ).basin
    
    data = data.where(land_mask)
    footprint = circular_footprint(r_lat)
    data = xr.apply_ufunc(
            lambda arr: apply_nanpercentile_filter(arr, footprint),
            data,
            input_core_dims=[['lat','lon']],
            output_core_dims=[['lat','lon']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[data.dtype]
            )
    
    data = data.where(land_mask, 0) + namelist.chi_fac
    data = np.maximum(np.minimum(data, 5), 1e-5)
    return data

def is_nc_file_valid(fn):
    is_valid = True
    if not os.path.exists(fn):
        is_valid = False
    else:
        try:
            root = Dataset(fn, 'r')
        except:
            is_valid = False
    return(is_valid)

def link_valid(link):
    # Check that file exists.
    cmd = "%s/realtime/validate.sh %s" % (namelist.src_directory, link)
    x = subprocess.check_output(cmd.split(' ')).decode("utf-8").rstrip('\n')
    return(x.lower() in ['true'])
    
def try_download(link, fn_out):
    # Sleep until the link becomes valid.
    while not link_valid(link):
        time.sleep(60)

    # Try to download 3 times.
    for i in range(3):
        try:
            cmd = "wget -q -O %s %s >/dev/null" % (fn_out, link)
            out = subprocess.check_output(cmd.split(' '), timeout=90)
            print(out)
        except:
            continue
        break
