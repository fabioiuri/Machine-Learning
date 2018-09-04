# Gather data and choosing features
import pandas as pd

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
#print(melbourne_data.columns)
#print(melbourne_data.describe())
melbourne_data = melbourne_data.dropna() # drop rows with NaNs

y = melbourne_data.Price # define prediction target
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
#print("X (Features):\n", X)
#print("y (Prediction Target):\n", y)

#print(X.describe())
#print(X.head())

# Build learning model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(train_X, train_y)

print("Making predictions for the following 5 houses:")
print(val_X.head())
print("The predictions are")
print(melbourne_model.predict(val_X.head()))
print("Real values are")
print(val_y.head())

# Predict MAE - Mean Absolute Error
from sklearn.metrics import mean_absolute_error

predicted_in_sample_prices = melbourne_model.predict(train_X)
in_sample_score = mean_absolute_error(train_y, predicted_in_sample_prices)
print("MAE (Mean Absolute Error) - In Sample Score =", in_sample_score)

predicted_out_sample_prices = melbourne_model.predict(val_X)
out_sample_score = mean_absolute_error(val_y, predicted_out_sample_prices)
print("MAE (Mean Absolute Error) - Out of Sample Score =", out_sample_score)