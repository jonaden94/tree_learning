import numpy as np
import pptk

path = "C:/Users/jonat/Documents/Studium/Angewandte Statistik/4.Semester/MA/KI-Segmentation/beech_plots_automatic/A1N/non_trees/non_trees.npy"

points = np.load(path)
points = points[::20]

colors = np.ones((len(points), 3)) * 255
v = pptk.viewer(points)
v.attributes(colors)
v.set(point_size=0.01)
