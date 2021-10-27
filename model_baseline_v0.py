import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('./preprocessed_blocking_data.csv')

event_ids = df['event_id'].to_numpy()
y_block_intensity = df['block_intensity'].to_numpy()

x_reanalysis_data = []

print("Loading reanalysis data...")
for event_id in event_ids:
    # print(f"loading reanalysis data: {event_id}")
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
    event_image_hog = hog(event_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(7, 7), block_norm='L2-Hys', feature_vector=True)
    x_reanalysis_data_hog.append(event_image_hog)

x_reanalysis_data_hog = np.array(x_reanalysis_data_hog)
HOG_FEATURE_SIZE = x_reanalysis_data_hog.shape[1]


# event_ids: (686,)
# x_reanalysis_data_hog: (686, 1008)
# y_block_intensity: (686,)

event_ids_train, event_ids_test, \
x_reanalysis_data_hog_train, x_reanalysis_data_hog_test, \
y_block_intensity_train, y_block_intensity_test = train_test_split(event_ids, x_reanalysis_data_hog, y_block_intensity, test_size=0.3, random_state=0)

print(f"event_ids_train: {event_ids_train.shape}")
print(f"event_ids_test: {event_ids_test.shape}")

print(f"x_reanalysis_data_hog_train: {x_reanalysis_data_hog_train.shape}")
print(f"x_reanalysis_data_hog_test: {x_reanalysis_data_hog_test.shape}")

print(f"y_block_intensity_train: {y_block_intensity_train.shape}")
print(f"y_block_intensity_test: {y_block_intensity_test.shape}")

min_max_scaler = RobustScaler()
min_max_scaler.fit(x_reanalysis_data_hog_train)

x_reanalysis_data_hog_train_scaled = min_max_scaler.transform(x_reanalysis_data_hog_train)
x_reanalysis_data_hog_test_scaled = min_max_scaler.transform(x_reanalysis_data_hog_test)

# model = SVR(kernel='rbf')
model = GradientBoostingRegressor(verbose=11, random_state=0)
model.fit(x_reanalysis_data_hog_train_scaled, y_block_intensity_train)
r2_train = model.score(x_reanalysis_data_hog_train_scaled, y_block_intensity_train)
r2_test = model.score(x_reanalysis_data_hog_test_scaled, y_block_intensity_test)

print(f"r2_train: {r2_train}")
print(f"r2_test: {r2_test}")