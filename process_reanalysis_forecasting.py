import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from netCDF4 import num2date
import datetime

# make new folder called reanalysis_data_processed
os.makedirs('./reanalysis_data_forecasting_processed', exist_ok=True)

dates = []
start_date = datetime.datetime(1979, 1, 1) 
end_date = datetime.datetime(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    dates.append(day)

for date in dates:
    print(f"date: {date}")
    f_nc = nc.Dataset(f'./american_reanalysis_data/hgt.{date.year}.nc')
    times = num2date(f_nc["time"][:], f_nc["time"].units)
    times = [t.isoformat() for t in times]
    times = [datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in times]
    t_idx = times.index(date)
    hgt_array_event = f_nc["hgt"][t_idx,:,:,:].filled(fill_value=0)
    event_array_file_path = f"./reanalysis_data_forecasting_processed/{date.year}-{date.month}-{date.day}.npy"
    print(f"saveing data for day to: {event_array_file_path}")
    np.save(event_array_file_path, hgt_array_event)