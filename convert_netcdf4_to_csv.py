import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from netCDF4 import num2date

f=nc.Dataset('/home/emirs/blocking-research/american_reanalysis_data/hgt.1979.nc')
hgt= f.variables['hgt']

time_dim, lat_dim, lon_dim, level_dim = hgt.get_dims()
time_var = f.variables[time_dim.name]
times = num2date(time_var[:], time_var.units)
latitudes = f.variables[lat_dim.name][:]
longitudes = f.variables[lon_dim.name][:]
levels= f.variables[level_dim.name][:]

output_dir = './'

filename = os.path.join(output_dir, 'table.csv')
print(f'Writing data in tabular form to {filename} (this may take some time)...')
times_grid, latitudes_grid, longitudes_grid, levels_grid = [
    x.flatten() for x in np.meshgrid(times, latitudes, longitudes, levels, indexing='ij')]
df = pd.DataFrame({
    'time': [t.isoformat() for t in times_grid],
    'latitude': latitudes_grid,
    'longitude': longitudes_grid,
    'levels' : levels_grid,
    'hgt': hgt[:].flatten()})
df.to_csv(filename, index=False)
print('Done')











#file='/home/emirs/blocking-research/american_reanalysis_data/hgt.1979.nc'
#ds=nc.Dataset(file,mode='r')


#lons = ds.variables['lon'][:]
#lats = ds.variables['lat'][:]
#levels= ds.variables['level'][:]
#hgt=ds.variables['hgt'][:]
#times=ds.variables['time'][:]

#output_dir = './' 

















