import numpy as np
import attr
from scipy.integrate import quad
from scipy.constants import k
from scipy.constants import physical_constants
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']
mass = atom['mass'] * amu

@attr.s
class MaxwellBoltzmann:
    T = attr.ib(converter=float)

    def pdf(self, v):
        return np.sqrt((mass/(2*np.pi*k*self.T))**3)*4*np.pi*v**2*np.exp(-mass*v**2/(2*k*self.T))

    def cdf(self, vmin, vmax):
        return quad(self.pdf, vmin, vmax)[0]*1e6    # in ppm

    def generate(self, N, vmax = None):
        ''' Rejection sampling routine '''
        v = []
        vmean = np.sqrt(8*k*self.T/np.pi/mass)
        if vmax == None:
            vmax = vmean+5*np.sqrt(k*self.T/mass)
        vprob = np.sqrt(2*k*self.T/mass)
        pmax = self.pdf(vprob)

        while len(v) < N:
            v0 = np.random.uniform(0,vmax)
            p0 = np.random.uniform(0, pmax)
            if p0 <= self.pdf(v0):
                v = np.append(v, v0)

        return v
