# Perceptron Learning Algorithm 
# Sample Usage and convergence test of perceptron on classification problem
# Compare results of built perceptron vs sklearn perceptron

import random
import pylab
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import mean_absolute_error
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

def get_sk_line(skperc):
    assert isinstance(skperc, SKPerceptron),\
        "Wrong perceptron type (must be sklearn.linear_model.perceptron)"
    w = skperc.coef_[0]
    slope = -w[0] / w[1]
    x_intercept = -skperc.intercept_ / w[1]
    return slope, x_intercept

# Driver
if __name__ == "__main__":
    # Generate points and plot them
    random.seed(3)
    D = generate_points(50)
    fig, mainplt = pylab.subplots(figsize=(7,7))
    pylab.ylim(Point.lo_lim, Point.up_lim)
    pylab.xlim(Point.lo_lim, Point.up_lim)
    mainplt.set_title("Single Perceptron Binary Classifier")
    mainplt.set_xlabel("x1")
    mainplt.set_ylabel("x2")
    #mainplt.plot([pt.x1 for pt in D], \
    #           [pt.x2 for pt in D], \
    #           "bo", label = "D = Sample data points")
    
    # Fix target function (unkown in real-life application)
    f = lambda x: 1.1 * x + 71 # target function (hand picked)
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [f(Point.lo_lim), f(Point.up_lim)], \
               "grey", label = "f(x) = target function")
    mainplt.legend()
        
    # Calculate D points classification (positive or negative)
    # Positive points: Above target function
    # Negative points: Bellow target function
    posD, negD = classify_points(D, f)
    mainplt.scatter([pt.x1 for pt in posD], [pt.x2 for pt in posD], c="b", marker="o")
    mainplt.scatter([pt.x1 for pt in negD], [pt.x2 for pt in negD], c="r", marker="x")
    X = np.array([(pt.x1, pt.x2) for pt in posD + negD])
    y = np.array([1] * len(posD) + [-1] * len(negD))
    
    # Start PLA (use built Perceptron vs. sklearn Perceptron)
    # (1) Split dataset into training and test datasets 80/20
    # (2) Fit perceptron using training dataset
    # (3) Test predictions using test dataset + plot g final hypothesis
    # Stage 1
    perc = Perceptron(n_features=2, plot=mainplt, \
                      plot_points=[Point.lo_lim, Point.up_lim])
    skperc = SKPerceptron(max_iter=10000)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1)
    # Stage 2
    perc.fit(X_train, y_train)
    skperc.fit(X_train, y_train)
    print("Model fit (built):", perc, "| Model fit (sklearn):", skperc.coef_[0])
    print("Iterations made (built):", perc.iters)
    # Stage 3
    # Predict on train data
    built_predicted_in_sample = perc.predict(X_train)
    built_in_sample_score = mean_absolute_error(y_train, built_predicted_in_sample)
    print("In Sample Score (built):", built_in_sample_score)
    sk_predicted_in_sample = skperc.predict(X_train)
    sk_in_sample_score = mean_absolute_error(y_train, sk_predicted_in_sample)
    print("In Sample Score (sklearn):", sk_in_sample_score)
    # Predict test data
    built_predicted_out_sample = perc.predict(X_test)
    built_out_sample_score = mean_absolute_error(y_test, built_predicted_out_sample)
    print("Out of Sample Score (built):", built_out_sample_score)
    sk_predicted_out_sample = skperc.predict(X_test)
    sk_out_sample_score = mean_absolute_error(y_test, sk_predicted_out_sample)
    print("Out of Sample Score (sklearn):", sk_out_sample_score)

    # PLot g and g'
    slope, x_intersect = perc.get_line()
    g = lambda x: slope * x + x_intersect
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [g(Point.lo_lim), g(Point.up_lim)], \
               "purple", label = "g(x) = final hypothesis by built")
    sk_slope, sk_x_intercept = get_sk_line(skperc)
    g2 = lambda x: sk_slope * x + sk_x_intercept
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [g2(Point.lo_lim), g2(Point.up_lim)], \
               "green", label = "g'(x) = final hypothesis by sklearn")
    mainplt.set_title("Single Perceptron Binary Classifier (built vs sklearn convergence)")
    mainplt.legend()
    fig.savefig('final_plot.png', dpi = 300)