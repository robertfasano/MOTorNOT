import numpy as np
from MOTorNOT.parameters import atom
from scipy.integrate import quad
from scipy.constants import k
from scipy.constants import physical_constants
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']
mass = atom['mass'] * amu

''' Atomic initial conditions '''
def maxwell_boltzmann(v, T):
    return np.sqrt((mass/(2*np.pi*k*T))**3)*4*np.pi*v**2*np.exp(-mass*v**2/(2*k*T))

def trapping_fraction(vmin, vmax, T):
    v0 = np.sqrt(8*k*T/np.pi/mass)
    v = np.linspace(0, 100, 1000)
    efficiency = quad(maxwell_boltzmann, vmin, vmax, args=(T))[0]*1e6    # in ppm
    return efficiency

def generate_maxwell_boltzmann(T, N, vmax = None):
    ''' Rejection sampling routine to generate a Maxwell-Boltzmann distribution at temperature T.'''
    v = []
    vmean = np.sqrt(8*k*T/np.pi/mass)
    if vmax == None:
        vmax = vmean+5*np.sqrt(k*T/mass)
    vprob = np.sqrt(2*k*T/mass)
    pmax = maxwell_boltzmann(vprob, T)


    while len(v) < N:
        v0 = np.random.uniform(0,vmax)
        p0 = np.random.uniform(0, pmax)
        if p0 <= maxwell_boltzmann(v0,T):
            v.append(v0)

    return np.array(v)


if __name__ == '__main__':
    N = 100
    v = generate_maxwell_boltzmann(450, N)
