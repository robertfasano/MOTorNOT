import numpy as np
import plotly.graph_objs as go
from MOTorNOT.plotting import plot_2D, plot_phase_space_force

def plot_phase_space_trajectories(acceleration, X, V, axis='x'):
    i = ord(axis)-120

    surf = plot_phase_space_force(acceleration, axis, limits=[(X.min(), X.max()), (-V.max(), V.max())], numpoints=100)
    traces = [surf]
    for p in range(X.shape[1]):
        traces.append(go.Scatter(x=X[:, p, i], y=V[:, p, i], line=dict(color='#ffffff')))
    fig = go.Figure(traces)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=r'${}$'.format(axis))),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=r'$v_{}$'.format(axis)))
        )
    fig.update_xaxes(range=[X[:, :, i].min(), X[:, :, i].max()])
    fig.update_yaxes(range=[-V[:, :, i].max(), V[:, :, i].max()])
    fig.show()


def plot_trajectories(acceleration, X, t, plane='xy'):
    i = ord(plane[0])-120
    j = ord(plane[1])-120
    x = X[:, :, i]
    y = X[:, :, j]
    fig = plot_2D(acceleration, plane, limits=[(x.min(), x.max()), (y.min(), y.max())], numpoints=30)

    for p in range(X.shape[1]):
        fig.add_trace(go.Scatter(x=x[:, p], y=y[:, p], line=dict(color='#ffffff')))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=r'${}$'.format(plane[0]))),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text=r'${}$'.format(plane[1])))
        )
    fig.update_xaxes(range=[x.min(), x.max()])
    fig.update_yaxes(range=[y.min(), y.max()])
    fig.show()
