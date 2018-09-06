import pandas as pd
import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 0. Read dataset
dataset = pd.read_csv("50_Startups.csv")
dataset = dataset.dropna()
X = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values
#print(X)
#print(y)

# 1. Preprocess data
label_encoder_state = LabelEncoder()
X[ : , 3] = label_encoder_state.fit_transform(X[ : , 3])
one_hot_encoder_state = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder_state.fit_transform(X).toarray()

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 3. Fit model
multi_regressor = LinearRegression()
multi_regressor = multi_regressor.fit(X_train, y_train)

# 4. Make predictions
in_sample_score = mean_absolute_error(y_train, multi_regressor.predict(X_train))
out_sample_score = mean_absolute_error(y_test, multi_regressor.predict(X_test))
print("in_sample_score", in_sample_score)
print("out_sample_score", out_sample_score)
print("Predictions for test set:")
print(multi_regressor.predict(X_test))