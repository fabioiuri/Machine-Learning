import numpy as np

class Perceptron:
    def __init__(self, n_features):
        self.w = np.array([0] * (n_features+1))
        self.iters = 0
        
    def __str__(self):
        return ", ".join(["x" + str(i) + "=" + str(self.w[i]) \
                          for i in range(len(self.w))])
        
    def sign(self, val):
        if val > 0:
            return 1
        else:
            return -1
        
    def predict(self, x):
        score = np.dot(self.w.T, x)
        return self.sign(score)
    
    def fit(self, X, y):
        changes = -1
        while changes != 0:
            changes = 0
            self.iters += 1
            for x, _y in zip(X, y):
                if self.predict(x) != _y:
                    changes += 1
                    self.w += _y * x
    
    def get_line(self):
        # TODO: generalize (only works for 2 features)
        slope = -self.w[1] / self.w[2]
        intersect = -self.w[0] / self.w[2]
        return slope, intersect
    
    def test_predictions(self, X, y):
        correct, wrong = 0, 0
        for x, y in zip(X, y):
            predicted = self.predict(x)
            if predicted == y:
                correct += 1
            else:
                wrong += 1
        return correct, wrong