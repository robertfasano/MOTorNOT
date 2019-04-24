import numpy as np
import pandas as pd
from MOTorNOT.parameters import atom
import sys
import time
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt
import sdeint

class Atoms():
    def __init__(self, X0, V0):
        ''' Initialize a cloud of atoms at position X0 and velocity V0. '''
        self.X0 = X0
        self.V0 = V0

class Solver():
    def __init__(self, X0, V0, force, recoil, duration, dt=None):
        self.position = X0.copy()
        self.velocity = V0.copy()
        self.force = force
        self.recoil = recoil
        self.last_time = None
        self.duration = duration
        self.last_percent = '0'
        self.time = 0
        self.dt = dt

    def acceleration(self, X, V):
        return self.force(X, V)/atom['m']

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

        self.status(t/self.duration)
        return a

    def f(self, y, t):
        return self.dydt(t,y)

    def func(self, y, t):

        N = int(len(y)/6)
        X = y[0:3*N].reshape(N,3)
        V = y[3*N::].reshape(N,3)
        k=3
        beta=1
        a = -k*X-beta*V
        a = np.append(V.flatten(), a.flatten())
#        print(a)
        return a

    def G(self, y, t):
        N = int(len(y)/6)
        X = y[0:3*N].reshape(N,3)
        V = y[3*N::].reshape(N,3)
        kick = self.recoil(X,V, self.duration/self.steps)[0]
        kick = np.array([0,0,0,kick,kick,kick])
        return np.diag(kick)

#        return np.diag(np.zeros(6*N))

    def solve_sde(self):
        self.start_time = time.time()
        self.steps = 10000
        tspan = np.linspace(0.0, self.duration, self.steps)
        y0 = np.append(self.position.flatten(), self.velocity.flatten()).astype(float)
#        print(y0)

#        return tspan, sdeint.itoint(self.f, self.G, y0, tspan)
        return tspan, sdeint.itoSRI2(self.f, self.G, y0, tspan)

    def solve(self):
        tspan=(0,self.duration)
        y0 = np.append(self.position.flatten(), self.velocity.flatten())
        self.start_time = time.time()
        self.timestep = []
        self.dv = []
        if self.dt is not None:
            t_eval = np.arange(0, self.duration, self.dt)
            r = solve_ivp(self.dydt, tspan, y0, t_eval=t_eval)
        else:
            r = solve_ivp(self.dydt, tspan, y0)
        # t = np.arange(0,self.duration, 1e-6)
        # y = odeint(self.dydt, y0, t)

        t = r.t
        y = r.y

        x = pd.DataFrame(index = t)
        j=0
        for i in range(int(len(y)/2)):
            ax_label = ['x', 'y', 'z'][i%3]
            x[ax_label+str(j)] = y[i]
            if i%3 == 2:
                j += 1

        v = pd.DataFrame(index = t)
        j = 0
        for i in range(int(len(y)/2), len(y)):
            ax_label = ['vx', 'vy', 'vz'][i%3]
            v[ax_label+str(j)] = y[i]
            if i%3 == 2:
                j += 1
        return x, v

    def plot(self, x, axis='x'):
        x = x[[i for i in x.columns if axis in i]]
        x.plot(legend=False)
        xmin = x[x.columns[0]].iloc[0]
        plt.ylim([-np.abs(xmin), np.abs(xmin)])


    def trapped_atoms(self, x, v, radius, min_height):
        x0 = x[[col for col in x.columns if 'x' in col]]
        z0 = x[[col for col in x.columns if 'z' in col]]

        trapped_indices = [col for col in x0.columns if np.abs(x0[col].iloc[-1]) < radius and z0[col.replace('x','z')].iloc[-1] > min_height]
        trapped_numeric_indices = [int(x[1::]) for x in trapped_indices]

        trapped_v_indices = []
        trapped_x_indices = []
        for i in trapped_numeric_indices:
            for ax in ['x', 'y', 'z']:
                trapped_x_indices.append('%s%i'%(ax,i))
                trapped_v_indices.append('v%s%i'%(ax,i))
        return trapped_x_indices, trapped_v_indices

    def trapping_efficiency(self, x):
        ''' Check the number of atoms inside the overlap region of the beam '''
        count = 0
        i = 0
        N_atoms = int(len(x.columns)/3)
        for i in range(N_atoms):
            if x['x%i'%i].iloc[-1] < trapping['radius'] and x['y%i'%i].iloc[-1] < trapping['radius'] and x['z%i'%i].iloc[-1] < trapping['radius']:
                count += 1
            i += 1
        return count/N_atoms

    def velocity_distribution(self, V, axis):
        ax_label = ['vx', 'vy', 'vz'][axis]
        for i in range(len(self.history)):
            plt.hist(self.history[[x for x in self.history.columns if ax_label in x]].iloc[i])
            plt.savefig('./data/%i.png'%i)
            plt.close()

    def histogram(self, V, bins=50):
        plt.figure()
#        V = np.linalg.norm(V, axis=1)
        plt.hist(V, bins=bins)
        plt.xlabel(r'$v$ (m/s)')

    def status(self, percent_done):
        if '%.2f'%percent_done == self.last_percent:
            return
        if float(percent_done) == 0:
            return
        self.last_percent = '%.2f'%percent_done
        time_elapsed = time.time()-self.start_time
        estimated_total_time = time_elapsed/percent_done
        time_remaining = estimated_total_time - time_elapsed
        print('\r'+'%.1f%% done; estimated time remaining: %.0f s'%(percent_done*100, time_remaining), end=' '*20, flush=True)
#        sys.stdout.flush()
