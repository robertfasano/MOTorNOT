import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

def radial_projection(x, v):
    ''' Returns the radial projection of a vector v at a position x'''
    X = np.atleast_2d(x)
    V = np.atleast_2d(v)


    phi = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2*np.pi)
    xproj = np.transpose([np.cos(phi), np.zeros(len(phi)), np.zeros(len(phi))])
    yproj = np.transpose([np.zeros(len(phi)), np.sin(phi), np.zeros(len(phi))])
    rhat = xproj + yproj

    return (rhat*V).sum(axis=1)

def axial_projection(x, v):
    ''' Returns the axial projection of a vector v at position x '''
    X = np.atleast_2d(x)
    V = np.atleast_2d(v)

    zhat = np.array([0, 0, np.sign(X[:, 2])])

    return (zhat*V).sum(axis=1)

def sample_2d(func, plane, limits, numpoints, component='all'):
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
    if component == 'all':
        agrid = np.linalg.norm(a, axis=1).reshape((numpoints, numpoints)).T
    elif component == 'radial':
        agrid = radial_projection(X, a).reshape((numpoints, numpoints)).T
    elif component == 'axial':
        agrid = axial_projection(X, a).reshape((numpoints, numpoints)).T

    return X, agrid, a

def plane_indices(plane):
    return ord(plane[0])-120, ord(plane[1])-120

def plot_2D(func, plane='xy', limits=[(-20e-3, 20e-3), (-20e-3, 20e-3)], numpoints=40, quiver=True, quiver_scale=5e-4, component='all'):
    ''' Generates a 2D plot of the passed vector function in the given plane. '''

    i = ord(plane[0]) - 120    # ordinate index
    j = ord(plane[1]) - 120    # abscissa index

    X, agrid, a = sample_2d(func, plane, limits, numpoints, component=component)
    xi = np.unique(X[:, i])
    xj = np.unique(X[:, j])
    ## create heatmap for norm of function
    fig = go.Figure()

    surf = go.Heatmap(x=xi,
                      y=xj,
                      z=agrid,
                      colorscale="Rainbow",
                      zsmooth='best',
                      colorbar={'title': 'Acceleration', 'titleside': 'right'})
    if quiver:
        n = numpoints
        X2, agrid2, a2 = X, agrid, a
        ## create quiver plot
        xg = X2[:, i].reshape(n, n)
        yg = X2[:, j].reshape(n, n)
        ax = (a2[:, i] / np.linalg.norm(a2, axis=1)).reshape(n, n)
        ay = (a2[:, j] / np.linalg.norm(a2, axis=1)).reshape(n, n)

        fig = ff.create_quiver(xg, yg, ax, ay, scale=quiver_scale)
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
                      colorscale="Rainbow",
                      colorbar={'title': 'Acceleration', 'titleside': 'right'})

    return surf
