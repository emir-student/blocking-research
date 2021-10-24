import netCDF4 as nc 
import numpy as np
import pandas as pd
import xarray as xr
import os

ds = xr.open_dataset('/home/emirs/blocking-research/american_reanalysis_data/hgt.1979.nc')
df = ds.to_dataframe()
path= '/home/emirs/blocking-research/'
df.to_csv(os.path.join(path,r'test_conversion'))






