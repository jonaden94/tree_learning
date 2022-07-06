import numpy as np
import plotly_express as px
import pandas as pd


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








