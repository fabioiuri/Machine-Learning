import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 0. Read data
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values

# 1. Handle misisng data
#   1st approach: replace NaNs with mean of column
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # 0 axis for column
imputer = imputer.fit(X[ : , 1:3]) # fit on age and salary
X[ : , 1:3] = imputer.transform(X[ : , 1:3]) # compute and add to dataset
#print("1st missing data approach:\n", X)

#   2nd approach: drop rows that contain some NaN value
X = dataset.dropna().iloc[ : , :-1].values # drop rows with NaNs
y =  dataset.dropna().iloc[ : , -1].values
#print("2nd missing data approach:\n", X)

# 2. Enconding categorical data (encode fields to be used within modle calcs)
#   one hot encoding is needed because if we just leave it after the label
#   encoding, different countries would have different numbers (e.g. France
#   would be 0, Spain = 1, Germany = 2), and that would make our model say:
#   "The higher the category value, the better the category. Therefore Spain
#   "is better than France, and Germany is better than France and Spain"
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0]) # encode country name
onehotencoder = OneHotEncoder(categorical_features = [0]) # hot encode contry categories 
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y =  labelencoder_Y.fit_transform(y) # encode purshaced field
#print("isFrance", "isSpain", "isGermany")
#print(X, y)


# 3. Split train and test data (80 / 20)    
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Scale data 
#    so that the nature of a feature does not define its weight in the model
#    Mean and standard deviation normalization:
#    1. Compute mean (mew) = 1/N * sum(x's)
#    2. Compute variance (signa**2) = sum((x's-mew)**2)/N
#    3. Compute std (sigma) = sqrt(variance)
#    4. Normalized x = (x - mew) / sigma
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("X_train:\n", X_train)
print("X_test:\n", X_test)