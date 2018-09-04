# Simple perceptron binary classifier implementation

import numpy as np

class Perceptron:
    def __init__(self, n_features):
        assert n_features == 2, \
               "Perceptron class is not ready for n_features != 2 (check get_line)"
        self.w = np.array([0] * (n_features+1))
        self.iters = 0
        
    def __str__(self):
        return ", ".join(["x" + str(i) + "=" + str(self.w[i]) \
                          for i in range(len(self.w))])
        
    # Auxiliary method
    def sign(self, val):
        if val > 0:
            return 1
        else:
            return -1
    
    # Calculates the sum of the multiplications between weights and params
    def decision_function(self, X):
        scores = []
        for x in X:
            score = np.dot(self.w.T, np.insert(x, 0, 1))
            scores.append(score)
        return scores
        
    # Predict the classification of set of inputs (classify -1 or 1)
    def predict(self, X):
        classif = []
        scores = self.decision_function(X)
        for score in scores:
            classif.append(self.sign(score))
        return classif
    
    def fit(self, X, y):
        changes = -1
        while changes != 0:
            changes = 0
            self.iters += 1
            for x, _y in zip(X, y):
                if self.predict([x])[0] != _y:
                    changes += 1
                    self.w += _y * np.insert(x, 0, 1)
    
    def get_line(self):
        # TODO: generalize (only works for 2 features)
        slope = -self.w[1] / self.w[2]
        x_intersect = -self.w[0] / self.w[2]
        return slope, x_intersect