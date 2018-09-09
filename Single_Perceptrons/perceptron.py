# Simple perceptron implementation as binary classifier
# Assumes data is always linearly seperable (does not rely on #iterations)

import numpy as np
import pylab
import os
from PIL import Image
import random

class Perceptron:
    plot_image_dir = "iteration_plots.temp/"
    
    def __init__(self, n_features, plot=None, plot_points=[], gif_framerate=100):
        assert n_features == 2, \
               "Perceptron class is not ready for n_features != 2 (check get_line)"
        self.w = np.zeros(n_features+1)
        self.iters = 0
        self.plot = plot
        self.plot_points = plot_points
        self.plot_names = []
        self.gif_framerate = gif_framerate
        if plot != None and not os.path.exists(self.plot_image_dir):
            os.makedirs(self.plot_image_dir)
        
    def __str__(self):
        return ", ".join(["x" + str(i) + "=" + str(self.w[i]) \
                          for i in range(len(self.w))])
    
    # Calculates the sum of the multiplications between weights and params
    def decision_function(self, X):
        scores = []
        for x in X:
            score = np.dot(self.w.T, np.insert(x, 0, 1)) # add artificial x0 = 1
            scores.append(score)
        return scores
        
    # Predict the classification of set of inputs (classify -1 or 1)
    def predict(self, X):
        classif = []
        scores = self.decision_function(X)
        for score in scores:
            classif.append(np.sign(score))
        return classif
    
    # Gets relative frequency of wrong predictions 
    def prediction_error(self, X, y):
        M = len(X)
        n_wrong = 0
        for x, y in zip(X, y):
            if self.predict([x])[0] != y:
                n_wrong += 1
        error = n_wrong / M
        return error
    
    # Get random point among the misclassified
    def choose_missclass_pt(self, X, y):
        wrong_pts = []
        for x, y in zip(X, y):
            if self.predict([x])[0] != y:
                wrong_pts.append((x, y))
        return wrong_pts[random.randrange(0,len(wrong_pts))]
    
    # Fit the model to the given X, y dataset
    def fit(self, X, y):
        save_plots = self.plot != None 
        if save_plots:
            print("INFO: Perceptron is printing plots as it converges...")
            self.save_plot()
        while self.prediction_error(X, y) != 0:
            self.iters += 1
            x, _y = self.choose_missclass_pt(X, y)
            self.w += _y * np.insert(x, 0, 1)
            if save_plots:
                self.save_plot()
        if save_plots:
            self.save_gif()
            
    # Returns the slope and intersects needed to draw the model line
    def get_line(self):
        # TODO: generalize (only works for 2 features)
        if self.w[2] == 0:
            return 0, 0
        slope = -self.w[1] / self.w[2]
        y_intercept = -self.w[0] / self.w[2]
        return slope, y_intercept
    
    # Saves plot of model's current state to plot_image_dir
    def save_plot(self):
        assert self.plot != None, \
            "No plot object was given - can't print plots"
        slope, y_intercept = self.get_line()
        h = lambda x: slope * x + y_intercept
        self.plot.plot(self.plot_points, \
                   list(map(lambda x: h(x), self.plot_points)), \
                   "black", label = "h(x) = current hypothesis by built") 
        self.plot.legend()
        name = self.plot_image_dir + "plot_" + str(self.iters) + ".png"
        pylab.gcf().savefig(name, dpi = 150)
        self.plot_names.append(name)
        self.plot.lines.pop()  # Delete last drawn line
        
    # Saves gif of all plots used to fit the dataset
    def save_gif(self):
        assert type(self.plot_names) == list and len(self.plot_names) > 0, \
            "No plots to convert to gif"
        # Open all the frames
        images = []
        for n in self.plot_names:
            frame = Image.open(n)
            images.append(frame)
        images += [frame] * 20 # To stay on last frame for a bit
        # Save the frames as an animated GIF
        name = self.plot_image_dir + "iteration_plots.gif"
        images[0].save(name,
                       save_all=True,
                       append_images=images[1:],
                       duration=self.gif_framerate,
                       loop=0)
        print("Saved", name)
        