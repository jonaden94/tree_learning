import numpy as np
import pptk
import os


# Error class to catch if no points in radius
class EmptyFilter(Exception):
    def __str__(self):
        return "Filter does not contain any data"


def explore(path, treenumber=0, radius=1000, remove_non_trees=False, subset=50):

    print(os.getcwd())

    # load all forest points
    points = np.load(path)
    points = points[::subset]

    # calculate tree position
    tree_indices = points[:, 3] == treenumber
    tree = points[tree_indices]
    tree_position = np.mean(tree, axis=0)
    tree_position = tree_position[[0, 1]]

    # remove non_tree points if desired
    if remove_non_trees:
        index = (points[:, -1]) != 9999
        points = points[index]

    # collect all points in radius around treenumber
    xypoints = points[:, [0, 1]]
    distances = np.linalg.norm(xypoints - tree_position, ord=np.inf, axis=1)
    within_radius = distances <= radius

    if not any(within_radius):
        raise EmptyFilter

    points = points[within_radius]

    # define a color palette and map colors to trees
    np.random.seed(3)
    n_color_palette = len(np.unique(points[:, 3]))
    color_palette = pptk.rand(n_color_palette, 3)

    color_palette_mapping = {j: i for i, j in enumerate(np.sort(np.unique(points[:, 3])))}
    num_points = len(points)
    colors = np.empty((num_points, 3))

    for i in range(num_points):
        ind = int(points[i][-1])
        colors[i] = color_palette[color_palette_mapping[ind]]

    # call plot
    v = pptk.viewer(points[:, :-1])
    v.attributes(colors)
    v.set(point_size=0.01)
    tree_position = np.hstack([tree_position, 0])
    v.set(lookat=tree_position)

plot = "G2W"
path = f"data/raw_data/beech_plots/{plot}/all_points/all_points.npy"
# path = "C:/Users/jonat/Documents/Studium/Angewandte Statistik/4.Semester/MA/KI-Segmentation_new/beech_plots_automatic/mixed_forest_manually_labeled/all_points/all_points.npy"

explore(path=path, treenumber=0, radius=1000, remove_non_trees=False)





