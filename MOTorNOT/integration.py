import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.constants import physical_constants
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']

class Atom:
    def __init__(self, X, V, t):
        self.x = pd.DataFrame(X, columns=['x', 'y', 'z'], index=t)
        self.x.index.rename('t', inplace=True)

        self.v = pd.DataFrame(V, columns=['vx', 'vy', 'vz'], index=t)
        self.v.index.rename('t', inplace=True)

def generate_initial_conditions(x0, v0, theta=0, phi=0):
    ''' Generates atomic positions and velocities along the z
        axis, then rotates to the spherical coordinates theta and
        phi.
    '''
    theta *= np.pi/180
    phi *= np.pi/180

    lenx = 0
    if hasattr(x0, '__len__'):
        lenx = len(x0)
    lenv = 0
    if hasattr(v0, '__len__'):
        lenv = len(v0)

    length = np.maximum(lenx, lenv)

    X = np.zeros((length, 3))
    X[:, 2] = x0

    V = np.zeros((length, 3))
    V[:, 2] = v0

    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    X = X.dot(Rx).dot(Rz)
    V = V.dot(Rx).dot(Rz)
    return X, V

class Solver():
    def __init__(self, X0, V0, force, duration, dt=None):
        self.position = X0.copy()
        self.velocity = V0.copy()
        self.force = force
        self.duration = duration
        self.last_percent = '0'
        self.dt = dt

    def acceleration(self, X, V):
        mass = atom['mass'] * amu
        return self.force(X, V)/mass

    def dydt(self, t, y):
        ''' Args:
                y (array-like): Array of length N where the first N/2 values correspond
                                to position and the last N/2 to velocity.
            Returns:
                '''
        N = int(len(y)/6)
        ''' Reconstruct arrays in (N,3) shape from flattened arrays '''
        X = y[0:3*N].reshape(N,3)
        V = y[3*N::].reshape(N,3)

        a = self.acceleration(X, V)

        ''' Flatten result to pass back into solver '''
        a = np.append(V.flatten(), a.flatten())
        return a

    def solve(self):
        tspan=(0,self.duration)
        y0 = np.append(self.position.flatten(), self.velocity.flatten())
        self.timestep = []
        self.dv = []
        if self.dt is not None:
            t_eval = np.arange(0, self.duration, self.dt)
            r = solve_ivp(self.dydt, tspan, y0, t_eval=t_eval, vectorized=True, dense_output=True)
        else:
            r = solve_ivp(self.dydt, tspan, y0)

        t = r.t
        y = r.y
        N = int(len(y)/6)

        # return in pandas dataframes
        atoms = []
        for i in range(3*N):
            if not i % 3:
                X = np.vstack([y[i], y[i+1], y[i+2]]).T
                V = np.vstack([y[i+3*N], y[i+1+3*N], y[i+2+3*N]]).T
                atom = Atom(X, V, t)
                atoms.append(atom)
        return atoms
