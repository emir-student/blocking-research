import numpy as np
import pandas as pd
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

forcasting_df = pd.DataFrame()

forcasting_df["date"] = ...
forcasting_df['reanalysis_fp'] = ... forcasting_df["date"] ...
forcasting_df['event'] = 0

# load the processed blocking data
# check per date in there

# Do all the time shifting to get x and y times series data
forcasting_df['event_t0'] = forcasting_df['event']
forcasting_df['event_t+1'] = forcasting_df['event'].shift()


