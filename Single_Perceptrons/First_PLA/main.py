# Perceptron Learning Algorithm 
# Sample Usage and convergence test of perceptron using PLA
# Compare results of built perceptron vs sklearn perceptron

import random
import pylab
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import mean_absolute_error

import sys
sys.path.append('../')
from perceptron import Perceptron

class Point:
    lo_lim = -5
    up_lim = 5
    
    def __init__(self, x1=None, x2=None):
        if x1 == None or x2 == None:
            self.x1 = random.uniform(self.lo_lim, self.up_lim)
            self.x2 = random.uniform(self.lo_lim, self.up_lim)
        else:
            self.x1 = x1
            self.x2 = x2            

    def __str__(self):
        return "(" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def __repr__(self):
        return "Point at (" + str(self.x1) + "," + str(self.x2) + ")" 
    
    def coords(self):
        return [self.x1, self.x2]

# Generate random linearly seperable data (and return target function)
def generate_points(n):
    xA,yA,xB,yB = [random.uniform(Point.lo_lim, Point.up_lim) for i in range(4)]
    f = lambda x: (yB-yA)/(xB-xA)*x - (xB*yA-xA*yB)/(xA-xB)
    w = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
    X = []
    y = []
    for i in range(n):
        point = Point()
        X.append(point.coords())
        x = np.array([1] + point.coords())
        y.append(int(np.sign(w.T.dot(x))))
    return X, y, f

# Check if y > 0 in [x,y]
def isPositive(xy):
    return xy[1] > 0

# Check if y < 0 in [x,y]
def isNegative(xy):
    return xy[1] < 0

# Get line params for tor the sklearn perceptron model (so we can plot it)
def get_sk_line(skperc):
    assert isinstance(skperc, SKPerceptron),\
        "Wrong perceptron type (must be sklearn.linear_model.perceptron)"
    w = skperc.coef_[0]
    slope = -w[0] / w[1]
    y_intercept = -skperc.intercept_ / w[1]
    return slope, y_intercept

# Test #iterations to converge using a dataset of size N
def test_iters(N):
    X, y, _ = generate_points(N)
    perc = Perceptron(n_features=2, max_iters=100000)
    perc.fit(X, y)
    return perc.iters
    
# Driver
if __name__ == "__main__":
    # Generate random linearly separable points and plot target function
    random.seed(0)
    X, y, f = generate_points(40)
    fig, mainplt = pylab.subplots(figsize=(7,7))
    pylab.ylim(Point.lo_lim, Point.up_lim)
    pylab.xlim(Point.lo_lim, Point.up_lim)
    mainplt.set_title("Single Perceptron Binary Classifier")
    mainplt.set_xlabel("x1")
    mainplt.set_ylabel("x2")
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [f(Point.lo_lim), f(Point.up_lim)], \
               "grey", label = "f(x) = target function")
        
    # Split points by category (positive or negative) and plot them
    # Positive points: Above target function
    # Negative points: Bellow target function
    posD = list(map(lambda xy: xy[0], filter(isPositive, zip(X,y))))
    negD = list(map(lambda xy: xy[0], filter(isNegative, zip(X,y))))
    mainplt.scatter([pt[0] for pt in posD], [pt[1] for pt in posD], c="b", marker="o")
    mainplt.scatter([pt[0] for pt in negD], [pt[1] for pt in negD], c="r", marker="x")
    X = np.array(X)
    y = np.array(y)
    
    # Start PLA (use built Perceptron vs. sklearn Perceptron)
    # (1) Split dataset into training and test datasets 80/20
    # (2) Fit perceptron using training dataset
    # (3) Test predictions using test dataset + plot g final hypothesis
    # Stage 1
    perc = Perceptron(n_features=2, plot=None, gif_framerate=300, \
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

    # Plot g and g'
    slope, x_intersect = perc.get_line()
    g = lambda x: slope * x + x_intersect
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [g(Point.lo_lim), g(Point.up_lim)], \
               "purple", label = "g(x) = final hypothesis by built")
    sk_slope, sk_y_intercept = get_sk_line(skperc)
    g2 = lambda x: sk_slope * x + sk_y_intercept
    mainplt.plot([Point.lo_lim, Point.up_lim], \
               [g2(Point.lo_lim), g2(Point.up_lim)], \
               "green", label = "g'(x) = final hypothesis by sklearn")
    mainplt.set_title("Single Perceptron Binary Classifier (built vs sklearn convergence)")
    mainplt.legend()
    fig.savefig('final_plot.png', dpi = 300)
    
    # Test multiple dataset sizes (N) in multiple conditions
    Ns = [10, 50, 100, 200, 350, 500, 750, 1000]
    random_seeds = [0, 1, 2, 3, 4, 5]
    results = np.zeros(len(Ns))
    for seed in random_seeds:
        print("Testing iterations on random seed", seed)
        random.seed(seed)
        for i in range(len(Ns)):
            results[i] += test_iters(Ns[i])
    results = results / len(random_seeds)
    print(results)
    fig, itersplot = pylab.subplots(figsize=(7,7))
    itersplot.bar(Ns, results, width=30)
    itersplot.set_xlabel("|D| : Dataset size")
    itersplot.set_ylabel("Number of Iterations")
    fig.savefig('iterations_bar_plot.png', dpi = 300)
    