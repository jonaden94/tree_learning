import numpy as np
import pptk
import plotly_express as px
import pandas as pd


# Error class to catch if no points in radius
class EmptyFilter(Exception):
    def __str__(self):
        return "Filter does not contain any data"


def explore(path, treenumber=0, radius=1000, remove_non_trees=False, subset=50, point_size=0.1):

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
    v.set(point_size=point_size)
    tree_position = np.hstack([tree_position, 0])
    v.set(lookat=tree_position)


def explore_simple(path, subset=50, point_size=0.01):
    points = np.load(path)
    points = points[::subset]

    colors = np.repeat(np.array([204, 0, 102]).reshape(1, 3), len(points), axis=0)
    v = pptk.viewer(points)
    v.attributes(colors)
    v.set(point_size=point_size)


def explore_plotly(coords, col=None, shift=None, size=None, width=800, height=800, show=True, write=False, savename=None):

    color_discrete_sequence = ["white", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "magenta", "grey", "maroon", "brown", "teal", "olive", "red", "navy", "pink", "beige"]

    if col is None and shift is None:
        df = pd.DataFrame(data=coords, columns=["x", "y", "z"])
    elif (not col is None) and shift is None:
        df = pd.DataFrame(data=zip(coords[:, 0], coords[:, 1], coords[:, 2], col.astype("str")), columns=["x", "y", "z", "col"])
    else:
        df = pd.DataFrame(data=zip(coords[:, 0] + shift[:, 0], coords[:, 1] + shift[:, 1], coords[:, 2] +  + shift[:, 2], col.astype("str")), columns=["x", "y", "z", "col"])

    if size == None:
        size = pd.DataFrame(data=np.ones(len(df)), columns=['size'])
    else:
        size = pd.DataFrame(data=size, columns=['size'])

    df = pd.concat([df, size], axis=1)
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color="col" if not col is None else None, size="size"\
                            ,opacity=0, template="plotly_dark", size_max=6, width=width, height=height, color_discrete_sequence=color_discrete_sequence)


    fig.update_layout(paper_bgcolor='rgba(50,50,50,50)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(title_text='Your title', title_x=0.5)
    fig.update_traces(marker=dict(line=dict(width=4, color='Black')), selector=dict(mode='markers'))

    if show:
        fig.show()
    if write:
        fig.write_html(savename + ".html")








