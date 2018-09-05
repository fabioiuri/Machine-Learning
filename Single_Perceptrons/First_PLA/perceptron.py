# Simple perceptron binary classifier implementation

import numpy as np
import pylab
import os
from PIL import Image
from random import shuffle

class Perceptron:
    plot_image_dir = "iteration_plots.temp/"
    
    def __init__(self, n_features, plot=None, plot_points=[]):
        assert n_features == 2, \
               "Perceptron class is not ready for n_features != 2 (check get_line)"
        self.w = np.array([0] * (n_features+1))
        self.iters = 0
        self.plot = plot
        self.plot_points = plot_points
        self.plot_names = []
        if plot != None and not os.path.exists(self.plot_image_dir):
            os.makedirs(self.plot_image_dir)
        
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
        print_plots = self.plot != None 
        if print_plots:
            print("INFO: Perceptron is printing plots as it converges...")
            self.print_plot()
        while changes != 0:
            changes = 0
            self.iters += 1
            inputs_class = list(zip(X, y))
            shuffle(inputs_class)
            for x, _y in inputs_class:
                if self.predict([x])[0] != _y:
                    changes += 1
                    self.w += _y * np.insert(x, 0, 1)
            if print_plots:
                self.print_plot()
        if print_plots:
            self.save_gif()
        
    def get_line(self):
        # TODO: generalize (only works for 2 features)
        if self.w[2] == 0:
            return 0, 0
        slope = -self.w[1] / self.w[2]
        x_intersect = -self.w[0] / self.w[2]
        return slope, x_intersect
    
    def print_plot(self):
        assert self.plot != None, \
            "No plot object was given - can't print plots"
        slope, x_intersect = self.get_line()
        h = lambda x: slope * x + x_intersect
        self.plot.plot(self.plot_points, \
                   list(map(lambda x: h(x), self.plot_points)), \
                   "black", label = "h(x) = current hypothesis by built") 
        self.plot.legend()
        name = self.plot_image_dir + "plot_" + str(self.iters) + ".png"
        pylab.gcf().savefig(name, dpi = 150)
        self.plot_names.append(name)
        self.plot.lines.pop()  # Delete last drawn line
        
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
                       duration=70,
                       loop=0)
        print("Saved", name)
        