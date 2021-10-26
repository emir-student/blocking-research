import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from netCDF4 import num2date
import datetime

df=pd.read_csv('./preprocessed_blocking_data.csv')

# make new folder called reanalysis_data_processed
os.makedirs('./reanalysis_data_processed',exist_ok=True)

for index, row in df.iterrows():
    year = int(row["blocking_year"])
    month = int(row["month"])
    day = int(row["day_begin"])
    print(year, month, day)
    event_dt = datetime.datetime(year=year, month=month, day=day)
    print(f"event date: {event_dt}")
    f_nc = nc.Dataset(f'./american_reanalysis_data/hgt.{year}.nc')
    times = num2date(f_nc["time"][:], f_nc["time"].units)
    times = [t.isoformat() for t in times]
    times = [datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in times]
    t_idx = times.index(event_dt)
    hgt_array_event = f_nc["hgt"][t_idx,:,:,:].filled(fill_value=0)
    event_array_file_path = f"./reanalysis_data_processed/{ int(row['event_id']) }.npy"
    print(f"saveing data for event to: {event_array_file_path}")
    np.save(event_array_file_path, hgt_array_event)


