import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import datetime
from datetime import timedelta
import os 


forecasting_df = pd.DataFrame()

dates = []
start_date = datetime.date(1979, 1, 1) 
end_date = datetime.date(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    day=day.strftime("%Y-%-m-%-d") 
    dates.append(day)

forecasting_df["date"] = dates

#reanalysis_fp= []
#for date in dates:
#    array = np.load(f"./reanalysis_data_forecasting_processed/{date}.npy")
#    reanalysis_fp.append(array)

#reanalysis_fp = np.array(reanalysis_fp)

#forecasting_df["reanalysis_fp"] = reanalysis_fp

dates = []
start_date = datetime.date(1979, 1, 1) 
end_date = datetime.date(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    dates.append(day)

forecasting_df["date"] = dates


blocking_df = pd.read_csv("preprocessed_blocking_data.csv")

blocking_dates = []

for i in blocking_df.index:
    blocking_date_range = pd.date_range(start=blocking_df.loc[i,'date_begin'], end=blocking_df.loc[i,'date_end'])
    blocking_dates.append(blocking_date_range)

blocking_dates= [x for l in blocking_dates for x in l] #Flattens data

for i in forecasting_df.index:
    if forecasting_df.loc[i,'date'] in blocking_dates:
        forecasting_df.loc[i,'event_t0'] = 1
    else:
        forecasting_df.loc[i,'event_t0'] = 0

forecasting_df['event_t+1'] = forecasting_df['event_t0'].shift(periods=-1)
forecasting_df['event_t+2'] = forecasting_df['event_t0'].shift(periods=-2)
forecasting_df['event_t+3'] = forecasting_df['event_t0'].shift(periods=-3)
forecasting_df['event_t+4'] = forecasting_df['event_t0'].shift(periods=-4)
forecasting_df['event_t+5'] = forecasting_df['event_t0'].shift(periods=-5)

print(forecasting_df)
print(forecasting_df.info())

path='/home/emirs/blocking-research/'
forecasting_df.to_csv(os.path.join(path,r'forecasting_df.csv'),index=False)








