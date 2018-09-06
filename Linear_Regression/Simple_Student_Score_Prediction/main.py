# Linear regression (real valued output) using sklearn implementation
# Predict student scores given hours studied
# Simple linear regression problem with 1 feature only

import pandas as pd
import pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 0. Read data
dataset = pd.read_csv("studentscores.csv")
X = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values

# 1. Handle misisng data
# drop rows that contain some NaN value
X = dataset.dropna().iloc[ : , :-1].values # drop rows with NaNs
y =  dataset.dropna().iloc[ : , -1].values

# 2. Split train and test data (80 / 20)    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Fit regressor
regressor = LinearRegression()
regressor = regressor.fit(X, y)

# 4. Make predictions
y_pred = regressor.predict(X_test)

# 5. Visualize the results
plt.title("Student scores by hours studied")
plt.xlabel("x = study hours")
plt.ylabel("y = scores")
plt.plot(X, y, "bo")
plt.plot(X, regressor.predict(X), "r", lw=2, label="g(x) : final hypothesis")
plt.legend()
in_sample_err = mean_absolute_error(y_train, regressor.predict(X_train))
out_sample_err = mean_absolute_error(y_test, y_pred)
print("In sample err:", in_sample_err)
print("Out of sample err:", out_sample_err)