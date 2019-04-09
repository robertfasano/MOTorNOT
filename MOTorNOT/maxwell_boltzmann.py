import numpy as np
from MOTorNOT.parameters import constants, atom
from scipy.integrate import quad

''' Atomic initial conditions '''
def maxwell_boltzmann(v, T):
    return np.sqrt((atom['m']/(2*np.pi*constants['kB']*T))**3)*4*np.pi*v**2*np.exp(-atom['m']*v**2/(2*constants['kB']*T))

def trapping_fraction(vmin, vmax, T):
    v0 = np.sqrt(8*constants['kB']*T/np.pi/atom['m'])
    v = np.linspace(0, 100, 1000)
    efficiency = quad(maxwell_boltzmann, vmin, vmax, args=(T))[0]*1e6    # in ppm
    return efficiency

def generate_maxwell_boltzmann(T, N, vmax = None):
    ''' Rejection sampling routine to generate a Maxwell-Boltzmann distribution at temperature T.'''
    v = []
    vmean = np.sqrt(8*constants['kB']*T/np.pi/atom['m'])
    if vmax == None:
        vmax = vmean+5*np.sqrt(constants['kB']*T/atom['m'])
    vprob = np.sqrt(2*constants['kB']*T/atom['m'])
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
