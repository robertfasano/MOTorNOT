import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from MOTorNOT.parameters import plot_params
from matplotlib import colors

def subplots(func, numpoints = 100, label = None, units = None, scale = 1):
    fig, ax = plt.subplots(2, 3)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    for axis in [0, 1, 2]:
        plot_1D(func, axis = axis, numpoints = numpoints**2, ax = ax[0][axis], label = label, units = units, scale=scale)
    i = 0
    for plane in ['xy', 'xz', 'yz']:
        plot_2D(func, plane=plane, numpoints = numpoints, ax = ax[1][i], label = label, units = units, scale = scale)
        i += 1
    plt.show()
    
def plot_1D(func, axis = 0, numpoints = 100, ax = None, label = None, units = None, scale = 1):
    ax_label = ['x', 'y', 'z'][axis]
    x = np.linspace(plot_params['axlim'][ax_label][0], plot_params['axlim'][ax_label][1], numpoints)
    X = np.zeros((numpoints, 3))
    X[:,axis] = x
    V = np.zeros((numpoints, 3))

    f = func(X,V)
    if f.shape[0] == 3:
        f = f.T
    f = f[:,axis]
    f *= scale

    if label is not None:
        label += r'_%s'%ax_label
        if units is not None:
            label +=  ' (%s)'%units

    if ax is None:
        plt.figure()
        plt.plot(x*1000, f)
        plt.xlabel('%s (mm)'%ax_label)
        plt.ylabel(r'$%s$)'%(label))
    else:
        ax.plot(x*1000, f)
        ax.set_xlabel('%s (mm)'%ax_label)
        ax.set_ylabel(r'$%s$'%(label))

def plot_field(B, coil_axis = 2, plot_axis = 0, numpoints = 100):
    ax_label = ['x', 'y', 'z'][plot_axis]
    plt.figure()
    x = np.linspace(plot_params['axlim'][ax_label][0], plot_params['axlim'][ax_label][1], numpoints)
    X = np.zeros((numpoints, 3))
    X[:,plot_axis] = x
    b = B(X, axis = coil_axis)

    plt.plot(x*1000, b[plot_axis]*1e4)
    plt.xlabel('%s (mm)'%ax_label)
    plt.ylabel('Magnetic field (G)')

def plot_2D(func, plane='xy', numpoints = 100, ax = None, label = None, units = None, scale = 1):
    ''' choose ordinate/abscissa based on user choice for variable plane '''
    ordinate = plane[0]
    abscissa = plane[1]
    index_dict = {'x':0, 'y':1, 'z':2}
    ordinate_index = index_dict[ordinate]
    abscissa_index = index_dict[abscissa]

    ''' set up coordinate arrays '''
    ordinate = np.linspace(plot_params['axlim'][ordinate][0], plot_params['axlim'][ordinate][1], numpoints)
    abscissa = np.linspace(plot_params['axlim'][abscissa][0], plot_params['axlim'][abscissa][1], numpoints)

    ''' Create meshgrid and convert to coordinate array form for vectorized calculation '''
    other_index = np.setdiff1d([0,1,2],[ordinate_index,abscissa_index])[0]
    coordinates = []
    pairs = np.transpose(np.meshgrid(ordinate,abscissa)).reshape(-1,2)
    for pair in pairs:
        coord = [0, 0, 0]
        coord[ordinate_index] = pair[0]
        coord[abscissa_index] = pair[1]
        coord[other_index] = 0
        coordinates.append(coord)

    X = np.array(coordinates)

    v = np.zeros(len(coordinates))
    V = np.array([v, v, v]).T

    a = func(X, V)
    if a.shape[0] == 3:
        a = a.T
    a *= scale

    ''' Create density and stream plots '''
    X = X[:,[ordinate_index, abscissa_index]]
    a_total = np.sqrt(a[:,0]**2+a[:,1]**2+a[:,2]**2)
    ordinate_mesh, abscissa_mesh = np.meshgrid(ordinate, abscissa)
    agrid = griddata(X, a_total, (ordinate_mesh,abscissa_mesh), method = 'linear')

    aogrid = griddata(X, a[:,ordinate_index], (ordinate_mesh,abscissa_mesh), method = 'linear')
    aagrid = griddata(X, a[:,abscissa_index], (ordinate_mesh,abscissa_mesh), method = 'linear')
    if label is not None:
        label = '|%s|'%label
        if units is not None:
            label += ' (%s)'%units
    if ax is None:
        plt.figure()
        plt.streamplot(ordinate_mesh*1000,abscissa_mesh*1000,aogrid,aagrid, density=2)
        plot = plt.contourf(ordinate_mesh*1000, abscissa_mesh*1000, agrid, plot_params['contours'], cmap='gist_rainbow')
        plt.colorbar(plot, label=label)
        plt.xlim([np.min(ordinate)*1000, np.max(ordinate)*1000])
        plt.ylim(np.min(abscissa)*1000, np.max(abscissa)*1000)
        plt.xlabel(plane[0] + ' (mm)')
        plt.ylabel(plane[1] + ' (mm)')

    else:
        ax.streamplot(ordinate_mesh*1000,abscissa_mesh*1000,aogrid,aagrid, density=2)
        plot = ax.contourf(ordinate_mesh*1000, abscissa_mesh*1000, agrid, plot_params['contours'], cmap='gist_rainbow')
        plt.colorbar(plot, label=label, ax=ax)
        ax.set_xlim([np.min(ordinate)*1000, np.max(ordinate)*1000])
        ax.set_ylim(np.min(abscissa)*1000, np.max(abscissa)*1000)
        ax.set_xlabel(plane[0] + ' (mm)')
        ax.set_ylabel(plane[1] + ' (mm)')


def plot_trajectory(df, N = None, axis = 0):
    ax_label = ['x', 'y', 'z'][axis]
    ''' Plots the phase-space trajectory of atom n '''
    n_atoms = int(len(df.columns)/6)
    if N is None:
        N = range(n_atoms)
    else:
        N = [N]

    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    i = 0
    for n in N:
        X = np.array([df[df.columns[i]], df[df.columns[i+1]], df[df.columns[i+2]]]).T
        V = np.array([df[df.columns[3*n_atoms+i]], df[df.columns[3*n_atoms+i+1]], df[df.columns[3*n_atoms+i+2]]]).T
        ax[0].plot(df.index*1000, X[:,axis]*1000)
        ax[0].set_ylabel(r'$%s$ (mm)'%ax_label)
        ax[0].set_xlabel(r'$t$ (ms)')
        ax[0].set_ylim([-1000*np.abs(X[0,axis]),1000*np.abs(X[0,axis])])

        ax[1].plot(df.index*1000, V[:,axis])
        ax[1].set_ylabel(r'$v_%s$ (m/s)'%ax_label)
        ax[1].set_xlabel(r'$t$ (ms)')
        ax[1].set_ylim([-np.abs(V[0,axis]),np.abs(V[0,axis])])

        ax[2].plot(1000*X[:,axis], V[:,axis])
        ax[2].set_xlabel(r'$%s$ (mm)'%ax_label)
        ax[2].set_ylabel(r'$v_%s$ (m/s)'%ax_label)
        ax[2].set_xlim([-1000*np.abs(X[0,axis]),1000*np.abs(X[0,axis])])
        ax[2].set_ylim([-np.abs(V[0,axis]),np.abs(V[0,axis])])
        i += 3

    vx = df[[x for x in df.columns if 'vx' in x]]
    vmin = vx.iloc[0].min()
    vmax = vx.iloc[0].max()
    fig.suptitle('Capture velocity range: %.1f-%.1f m/s'%(vmin, vmax))
