import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

def plot_2D(func, plane='xy', limits=[(-20e-3, 20e-3), (-20e-3, 20e-3)], numpoints=30, quiver=True):
    ''' Generates a 2D plot of the passed vector function in the given plane. '''

    i = ord(plane[0]) - 120    # ordinate index
    j = ord(plane[1]) - 120    # abscissa index

    ## set up coordinate arrays
    xi = np.linspace(limits[0][0], limits[0][1], numpoints)
    xj = np.linspace(limits[1][0], limits[1][1], numpoints)

    ## create meshgrid and convert to coordinate array form for vectorized calculation
    k = 3-(i+j)             # index of orthogonal axis
    pairs = np.transpose(np.meshgrid(xi, xj)).reshape(-1,2)
    X = np.hstack((pairs[:,:k], np.zeros((len(pairs), 1)), pairs[:,k:]))
    V = np.zeros(X.shape)

    ## compute function at coordinates
    a = func(X, V)
    agrid = np.linalg.norm(a, axis=1).reshape((numpoints, numpoints)).T

    ## create heatmap for norm of function
    fig = go.Figure()
    surf = go.Heatmap(x=xi,
                      y=xj,
                      z=agrid,
                      colorscale="Viridis",
                      zsmooth='best',
                      colorbar={'title': 'Acceleration', 'titleside': 'right'})
    if quiver:
        ## create quiver plot
        xg = X[:, i].reshape(numpoints, numpoints)
        yg = X[:, j].reshape(numpoints, numpoints)
        ax = a[:, i].reshape(numpoints, numpoints)
        ay = a[:, j].reshape(numpoints, numpoints)

        ax /= ax.max() / 10
        ay /= ay.max() / 10
        # scale = 1e-4
        scale = (limits[0][1] - limits[0][0])/500
        fig = ff.create_quiver(xg, yg, ax, ay, scale=scale)
        fig['data'][0]['line']['color'] = 'rgb(255,255,255)'

    fig.add_trace(surf)
    fig.update_xaxes(range=[xi.min(), xi.max()])
    fig.update_yaxes(range=[xj.min(), xj.max()])

    return fig

def plot_phase_space_force(func, axis='x', limits=[(-20e-3, 20e-3), (-20e-3, 20e-3)], numpoints=30):
    i = ord(axis) - 120             # 0, 1, or 2 corresponding to x, y, or z

    ## set up coordinate arrays
    x = np.linspace(limits[0][0], limits[0][1], numpoints)
    v = np.linspace(limits[1][0], limits[1][1], numpoints)
    pairs = np.transpose(np.meshgrid(x, v)).reshape(-1,2)
    X = np.zeros((numpoints**2, 3))
    V = np.zeros((numpoints**2, 3))
    X[:, i] = pairs[:, 0]
    V[:, i] = pairs[:, 1]

    a = func(X, V)
    agrid = a[:, i].reshape(numpoints, numpoints).T

    surf = go.Heatmap(x=x,
                      y=v,
                      z=agrid,
                      colorscale="Viridis",
                      colorbar={'title': 'Acceleration', 'titleside': 'right'})

    return surf
