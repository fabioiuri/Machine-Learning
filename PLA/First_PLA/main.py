# Perceptron Learning Algorithm 
# Sample Usage and convergence test

import random
import pylab
import numpy as np
from sklearn.model_selection import train_test_split
from perceptron import *

class Point:
    lo_lim = -100
    up_lim = 100
    
    def __init__(self, x1=None, x2=None):
        if x1 == None or x2 == None:
            self.x1 = random.randint(self.lo_lim, self.up_lim)
            self.x2 = random.randint(self.lo_lim, self.up_lim)
        else:
            self.x1 = x1
            self.x2 = x2            

    def __str__(self):
        return "(" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def __repr__(self):
        return "Point at (" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def coords(self):
        return [self.x1, self.x2]

def generate_points(n):
    points = []
    for i in range(n):
        points.append(Point())
    return points

def classify_points(D, f):
    positiveD = []
    negativeD = []
    for pt in D:
        x1, x2 = pt.x1, pt.x2
        f_o = f(x1)
        if x2 > f_o:
            positiveD.append(pt)
        else:
            negativeD.append(pt)
    return positiveD, negativeD

# Driver
if __name__ == "__main__":
    # Generate points and plot them
    random.seed(0)
    D = generate_points(100)
    pylab.title("Single Perceptron Convergence")
    pylab.xlabel("x1")
    pylab.ylabel("x2")
    #pylab.plot([pt.x1 for pt in D], \
    #           [pt.x2 for pt in D], \
    #           "bo", label = "D = Sample data points")
    
    # Fix target function (unkown in real-life application)
    f = lambda x: -0.3 * x + 35 # target function (hand picked)
    pylab.plot([Point.lo_lim, Point.up_lim], \
               [f(Point.lo_lim), f(Point.up_lim)], \
               "grey", label = "f(x) = target function")
    pylab.legend()
    
    # Calculate D points classification (positive or negative)
    # Positive points: Above target function
    # Negative points: Bellow target function
    posD, negD = classify_points(D, f)
    pylab.plot([pt.x1 for pt in posD], [pt.x2 for pt in posD], "bo")
    pylab.plot([pt.x1 for pt in negD], [pt.x2 for pt in negD], "rx")
    X = np.array([(1, pt.x1, pt.x2) for pt in posD + negD])
    y = np.array([1] * len(posD) + [-1] * len(negD))
    
    # Start PLA
    # (1) Split dataset into training and test datasets 80/20
    # (2) Fit perceptron using training dataset
    # (3) Test predictions using test dataset + plot g final hypothesis
    # Stage 1
    perc = Perceptron(2)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1)
    # Stage 2
    perc.fit(X_train, y_train)
    print("Model fit:", perc)
    print("Iterations made:", perc.iters)
    # Stage 3
    # Predict on train data
    correct, wrong = perc.test_predictions(X_train, y_train)
    print("Correct predictions on train data:", correct)
    print("Wrong predictions on train data:", wrong)
    # Predict test data
    correct, wrong = perc.test_predictions(X_test, y_test)
    print("Correct predictions on test data:", correct)
    print("Wrong predictions on test data:", wrong)
    # PLot g
    slope, intersect = perc.get_line()
    g = lambda x: slope * x + intersect
    pylab.plot([Point.lo_lim, Point.up_lim], \
               [g(Point.lo_lim), g(Point.up_lim)], \
               "purple", label = "g(x) = final hypothesis")
    pylab.legend()