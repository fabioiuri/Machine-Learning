# Linear Regressor simple implementation
# Assumes x0 artificial coordinate is NOT setup yet (LinRegressor sets it up)

import numpy as np

class LinRegressor:
    
    # Init empty weight array - not needed
    def __init__(self):
        self.w = []
    
    # To add x0 = 1 so that we can have w0 to assume intersection value
    def add_artificial_x0(self, X):
        newX = []
        for x in X:
            x = np.insert(x, 0, 1)
            newX.append(x)
        return np.array(newX)
      
    # Fits linear regressor - works for multiple linear regression problems too
    def fit(self, X, y):
        X = self.add_artificial_x0(X)
        X_dagger = np.linalg.pinv(X)
        self.w = np.dot(X_dagger, y)
        return self
    
    # Predict ys given X
    def predict(self, X):
        X = self.add_artificial_x0(X)
        y_pred = np.dot(X, self.w)
        return y_pred