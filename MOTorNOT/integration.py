import numpy as np
import pandas as pd
import attr
from scipy.integrate import solve_ivp
from scipy.constants import physical_constants

amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters, rotate
atom = load_parameters()['atom']

def generate_initial_conditions(x0, v0, theta=0, phi=0):
    ''' Generates atomic positions and velocities along the z
        axis, then rotates to the spherical coordinates theta and
        phi.
    '''
    lenx = 0
    if hasattr(x0, '__len__'):
        lenx = len(x0)
    lenv = 0
    if hasattr(v0, '__len__'):
        lenv = len(v0)

    length = np.maximum(lenx, lenv)

    if length == 0:
        X = np.atleast_2d([0, 0, x0])
        V = np.atleast_2d([0, 0, v0])
    else:
        X = np.zeros((length, 3))
        X[:, 2] = x0

        V = np.zeros((length, 3))
        V[:, 2] = v0

    Rx = rotate(0, theta)
    Rz = rotate(2, phi)
    X = X.dot(Rx).dot(Rz)
    V = V.dot(Rx).dot(Rz)
    return X, V

from MOTorNOT.analysis import *

@attr.s
class Solver:
    acceleration = attr.ib()
    X0 = attr.ib(converter=np.atleast_2d)
    V0 = attr.ib(converter=np.atleast_2d)

    def run(self, duration, dt=None):
        self.y, self.X, self.V, self.t = solve(self.acceleration,
                                       self.X0,
                                       self.V0,
                                       duration, dt=dt)
        return self

    def get_particle(self, i):
        ''' Returns a DataFrame containing the dynamics of the ith particle. '''
        df = pd.DataFrame.from_dict({'x': self.X[:, i, 0],
                                     'y': self.X[:, i, 1],
                                     'z': self.X[:, i, 2],
                                     'vx': self.V[:, i, 0],
                                     'vy': self.V[:, i, 1],
                                     'vz': self.V[:, i, 2]})

        return df.set_index(self.t)

    def get_position(self, i):
        ''' Returns an array containing the positions at the ith timestep. '''
        return self.X[i, :, :]

    def get_velocity(self, i):
        ''' Returns an array containing the velocities at the ith timestep. '''
        return self.V[i, :, :]

    def plot(self, plane='xy', limits=None, trapped_only=False, numpoints=50, quiver_scale=3e-4):
        X = self.X
        if trapped_only:
            X = X[:, self.trapped(), :]
        plot_trajectories(self.acceleration, X, self.t, plane=plane, limits=limits, numpoints=numpoints, quiver_scale=quiver_scale)

    def phase_plot(self, axis='x', trapped_only=False):
        X = self.X
        V = self.V
        if trapped_only:
            X = X[:, self.trapped(), :]
            V = V[:, self.trapped(), :]

        plot_phase_space_trajectories(self.acceleration, X, V, axis=axis)

    def capture_velocity(self, rmax=1e-3, vmax=1e-3):
        vi = self.get_velocity(0)[self.trapped(rmax, vmax)].max(axis=1)
        if len(vi) > 0:
            return vi.min(), vi.max()
        else:
            return 0, 0


    def trapped(self, rmax=1e-3, vmax=1e-3):
        ''' Return indices of atoms within a velocity threshold at the end of the simulation.
            Argument rmax currently does nothing.
        '''
        vf = np.linalg.norm(self.get_velocity(-1), axis=1)
        return np.where(vf < vmax)[0]

def solve(acceleration, X0, V0, duration, dt=None):
    ''' Integrates the equations of motion given by the specified force,
        starting from given initial conditions.
    '''

    def dydt(t, y):
        ''' Args:
                y (array-like): Array of length N where the first N/2 values correspond
                                to position and the last N/2 to velocity.
            Returns:
                '''
        N = int(len(y)/6)
        ''' Reconstruct arrays in (N,3) shape from flattened arrays '''
        X = y[0:3*N].reshape(N,3)
        V = y[3*N::].reshape(N,3)

        a = acceleration(X, V)

        return np.append(V.flatten(), a.flatten())

    y0 = np.append(np.array(X0).flatten(), np.array(V0).flatten())
    if dt is not None:
        t_eval = np.arange(0, duration, dt)
        r = solve_ivp(dydt, (0, duration), y0, t_eval=t_eval, vectorized=True)
    else:
        r = solve_ivp(dydt, (0, duration), y0, vectorized=True)

    t = r.t
    y = r.y
    N = int(len(y)/6)

    X = y[0:3*N, :].T.reshape(-1, N, 3)
    V = y[3*N:6*N, :].T.reshape(-1, N, 3)

    return y, X, V, t
