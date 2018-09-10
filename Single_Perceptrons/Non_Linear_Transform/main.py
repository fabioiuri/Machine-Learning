# Perceptron Learning Algorithm 
# Sample Usage and convergence test of perceptron using PLA
# Compare results of built perceptron vs sklearn perceptron

import random
import pylab
from matplotlib.patches import Ellipse
import math
import numpy as np
import numpy.random as rnd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')
from perceptron import Perceptron

class Point:    
    lo_lim = 0
    up_lim = 2
    
    def __init__(self):
        self.x1 = random.choice([-1, 1]) * random.uniform(self.lo_lim, self.up_lim)
        self.x2 = random.choice([-1, 1]) * random.uniform(self.lo_lim, self.up_lim)          

    def __str__(self):
        return "(" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def __repr__(self):
        return "Point at (" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def coords(self):
        return [self.x1, self.x2]

# Generate random linearly seperable data (and return target function)
def generate_points(n):
    X = []
    y = []
    for i in range(n): 
        point = Point()
        X.append(point.coords())
        dist_orig = math.sqrt(point.x1**2 + point.x2**2)
        if dist_orig < (Point.up_lim + Point.lo_lim) / 2:
            y.append(-1)
        else:
            y.append(1)
    return X, y

# Check if y > 0 in [x,y]
def isPositive(xy):
    return xy[1] > 0

# Check if y < 0 in [x,y]
def isNegative(xy):
    return xy[1] < 0

# Driver
if __name__ == "__main__":
    # Generate random linearly separable points and plot target function
    random.seed(0)
    X, y = generate_points(50)
    fig, mainplt = pylab.subplots(figsize=(7,7))
    mainplt.set_title("Original Dataset w/ original target function")
    mainplt.set_xlabel("x1")
    mainplt.set_ylabel("x2")
    mainplt.add_artist(Ellipse(xy=[Point.lo_lim, Point.lo_lim], \
                               width=Point.up_lim + Point.lo_lim, \
                               height=Point.up_lim + Point.lo_lim, \
                               edgecolor='black', lw=2, facecolor='none'))
        
    # Split points by category (positive or negative) and plot them
    # Positive points: Above target function
    # Negative points: Bellow target function
    posD = list(map(lambda xy: xy[0], filter(isPositive, zip(X,y))))
    negD = list(map(lambda xy: xy[0], filter(isNegative, zip(X,y))))
    mainplt.scatter([pt[0] for pt in posD], [pt[1] for pt in posD], c="b", marker="o")
    mainplt.scatter([pt[0] for pt in negD], [pt[1] for pt in negD], c="r", marker="x")
    X = np.array(X)
    y = np.array(y)
    fig.savefig('original_plot.png', dpi = 300)
    
    # Non Linear Transformation (phi)
    phi = lambda xs: list(np.array(xs)**2)
    fig, transfplot = pylab.subplots(figsize=(7,7))
    transfplot.set_title("Transformed Dataset ($\phi(x)=x^2$)")
    transfplot.set_xlabel("$x1^2$")
    transfplot.set_ylabel("$x2^2$")
    transfplot.set_ylim(Point.lo_lim, Point.up_lim**2)
    transfplot.set_xlim(Point.lo_lim, Point.up_lim**2)
    posD = list(map(phi, posD))
    negD = list(map(phi, negD))
    transfplot.scatter([pt[0] for pt in posD], [pt[1] for pt in posD], c="b", marker="o")
    transfplot.scatter([pt[0] for pt in negD], [pt[1] for pt in negD], c="r", marker="x")
    X = phi(X) # Transform Non linear seperable X to a linear seperable X' through phi
    
    # Start PLA (use built Perceptron)
    # (1) Split dataset into training and test datasets 80/20
    # (2) Fit perceptron using training dataset
    # (3) Test predictions using test dataset + plot g final hypothesis
    # Stage 1
    perc = Perceptron(n_features=2, plot=transfplot, gif_framerate=300, \
                      plot_points=[-Point.up_lim**2, Point.up_lim**2])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1)
    # Stage 2
    perc.fit(X_train, y_train)
    print("Model fit:", perc)
    print("Iterations made:", perc.iters)
    # Stage 3
    # Predict on train data
    built_predicted_in_sample = perc.predict(X_train)
    built_in_sample_score = mean_absolute_error(y_train, built_predicted_in_sample)
    print("In Sample Score (built):", built_in_sample_score)
    # Predict test data
    built_predicted_out_sample = perc.predict(X_test)
    built_out_sample_score = mean_absolute_error(y_test, built_predicted_out_sample)
    print("Out of Sample Score (built):", built_out_sample_score)

    # PLot g and g'
    slope, x_intersect = perc.get_line()
    g = lambda x: slope * x + x_intersect
    transfplot.plot([-Point.up_lim**2, Point.up_lim**2], \
               [g(-Point.up_lim**2), g(Point.up_lim**2)], \
               "purple", label = "g(x) = final hypothesis by built")
    transfplot.legend()
    fig.savefig('final_plot.png', dpi = 300)