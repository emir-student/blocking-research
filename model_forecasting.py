import numpy as np
import pandas as pd
import datetime

from skimage.feature import hog
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor 

from sklearn.metrics import mean_absolute_error as MAE 

forecasting_df = pd.DataFrame()

print("Adding dates to date column of dataframe...")
dates = []
start_date = datetime.date(1979, 1, 1) 
end_date = datetime.date(2020, 12, 31)    # perhaps date.now()
delta = end_date - start_date   # returns timedelta
for i in range(delta.days + 1):
    day = start_date + datetime.timedelta(days=i)
    day=day.strftime("%Y-%-m-%-d")
    dates.append(day)

forecasting_df["date"] = dates

print("Adding reanalysis data to reanalysis_fp column of dataframe...")
reanalysis_fp= []
for date in dates:
    array = np.load(f"./reanalysis_data_forecasting_processed/{date}.npy")
    reanalysis_fp.append(array)

reanalysis_fp = np.array(reanalysis_fp)

#forecasting_df["reanalysis_fp"] = reanalysis_fp

print("Done")

#forecasting_df["reanalysis_fp"] = 

#forecasting_df['reanalysis_fp'] = ... forecasting_df["date"] ...
#forecasting_df['event'] = 0

# load the processed blocking data
# check per date in there

#Do all the time shifting to get x and y times series data
#forecasting_df['event_t0'] = forecasting_df['event']
#forecasting_df['event_t+1'] = forecasting_df['event'].shift()

#print(forecasting_df)
