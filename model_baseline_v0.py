import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler

import xgboost

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor 

from sklearn.metrics import mean_absolute_error as MAE 
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('./preprocessed_blocking_data.csv')

event_ids = df['event_id'].to_numpy()
y_data = df['block_intensity'].to_numpy() 

x_reanalysis_data = []

print("Loading reanalysis data...")
for event_id in event_ids:
    event_array = np.load(f"./reanalysis_data_processed/{event_id}.npy")
    x_reanalysis_data.append(event_array)

x_reanalysis_data = np.array(x_reanalysis_data)

x_reanalysis_data_500_mb = x_reanalysis_data[:,5,:,:]

# (686, 73, 144)
# for each event: (73, 144) -> hog -> (n)
# (686, n)
print("Extracting HOG data...")
x_reanalysis_data_hog = []
for i in range(x_reanalysis_data_500_mb.shape[0]):
    # print(f"extracting hog features: {event_ids[i]}")
    event_image = x_reanalysis_data_500_mb[i,:,:]
    event_image_hog = hog(event_image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(3,3), block_norm='L2-Hys', feature_vector=True)
    x_reanalysis_data_hog.append(event_image_hog)

x_reanalysis_data_hog = np.array(x_reanalysis_data_hog)
HOG_FEATURE_SIZE = x_reanalysis_data_hog.shape[1]


# event_ids: (686,)
# x_reanalysis_data_hog: (686, 1008)
# y_block_intensity: (686,)

event_ids_train, event_ids_test, \
x_reanalysis_data_hog_train, x_reanalysis_data_hog_test, \
y_data_train, y_data_test = train_test_split(event_ids, x_reanalysis_data_hog, y_data, test_size=0.3, random_state=0)

print(f"event_ids_train: {event_ids_train.shape}")
print(f"event_ids_test: {event_ids_test.shape}")

print(f"x_reanalysis_data_hog_train: {x_reanalysis_data_hog_train.shape}")
print(f"x_reanalysis_data_hog_test: {x_reanalysis_data_hog_test.shape}")

print(f"y_train: {y_data_train.shape}")
print(f"y_test: {y_data_test.shape}")

scaler= RobustScaler()
scaler.fit(x_reanalysis_data_hog_train)

x_reanalysis_data_hog_train_scaled = scaler.transform(x_reanalysis_data_hog_train)
x_reanalysis_data_hog_test_scaled = scaler.transform(x_reanalysis_data_hog_test)


#model = LinearRegression()
#model = Ridge(alpha=1, random_state=0)
#model = Lasso(alpha=1, random_state=0)
#model = SVR(kernel='rbf', verbose=1)
#model = GradientBoostingRegressor(random_state=0, n_estimators=100, verbose=11)       # Each tree feeds into the next, hoping to improve results. 
model = RandomForestRegressor(random_state=0)             # Takes average of a bunch of decision trees.
#model = xgboost.XGBRegressor(n_estimators= 200, verbosity= 2, max_depth=6, gamma=5)
#model=MLPRegressor(hidden_layer_sizes=(256,128,64),
                    #activation='relu',
                    #solver='adam',
                    #batch_size=128,
                    #learning_rate_init=1e-3,
                    #max_iter=200,
                    #verbose=True,
                    #alpha=1e-3)


model.fit(x_reanalysis_data_hog_train_scaled, y_data_train)
#model.fit(x_reanalysis_data_hog_train_scaled, y_data_train)
r2_train = model.score(x_reanalysis_data_hog_train_scaled, y_data_train)
r2_test = model.score(x_reanalysis_data_hog_test_scaled, y_data_test)

print(f"r2_train: {r2_train}")
print(f"r2_test: {r2_test}")

y_train_predicted = model.predict(x_reanalysis_data_hog_train_scaled)
y_test_predicted = model.predict(x_reanalysis_data_hog_test_scaled)

mae_train = MAE(y_train_predicted, y_data_train)
mae_test = MAE(y_test_predicted, y_data_test)
print(f"MAE Train: {mae_train}")
print(f"MAE Test: {mae_test}") 
mape_train = np.average( np.abs( (y_train_predicted - y_data_train) /y_data_train ) )
mape_test = np.average( np.abs( (y_test_predicted - y_data_test) /y_data_test) )
print(f"MAPE Train: {mape_train}")
print(f"MAPE Test: {mape_test}") 

error_dataset = pd.DataFrame()
error_dataset['y_test_predicted'] = y_test_predicted
error_dataset['y_test_actual'] = y_data_test

path = '/home/emirs/blocking-research/'
error_dataset.to_csv(os.path.join(path,r'error.csv'),index=False)


# error_train= y_train_predicted - y_data_train 
# error_test= y_test_predicted - y_data_test

# sns.displot([error_train,error_test], kind='kde')
# plt.savefig('model_error.png')
