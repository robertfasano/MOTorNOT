from parameters import state, atom
from gmot import gratingMOT
from coils import Coil, Coils
import matplotlib.pyplot as plt
import numpy as np
from maxwell_boltzmann import trapping_fraction

def prepare(state):
    ''' Prepares a MOT with settings defined by the passed state dict.
        You can visualize the coil field or MOT acceleration profile by calling
        coils.plot() or mot.plot() respectively. '''

    ''' INSTANTIATE COILS '''
    Coil1 = Coil(state['coil1']['R'], state['coil1']['z'], state['coil1']['N'], -state['coil1']['I'], axis=2)
    Coil2 = Coil(state['coil2']['R'], state['coil2']['z'], state['coil2']['N'], state['coil2']['I'], axis=2)

    coils = Coils([Coil1, Coil2])

    ''' INSTANTIATE BEAMS '''
    alpha = np.arcsin(atom['wavelength']/state['grating']['d'])
    mot = gratingMOT(position = state['grating']['position'],
                       alpha=alpha,
                       radius = state['trapping']['radius'],
                       field = coils.field,
                       power = state['trapping']['power'],
                       detuning = state['trapping']['detuning'],
                       polarization = state['trapping']['polarization'],
                       R1=state['grating']['R1'],
                       show_negative=True,
                       show_incident=True)

    return mot, coils

def simulate(X, V, mot):
    ''' Run a simulation'''
    from integration import Solver
    tmax = 20*np.abs(np.min(X)/np.min(V[:,0]))
    sol = Solver(X, V, mot.force, None, duration = tmax)
    x, v = sol.solve()
    xi,vi = sol.trapped_atoms(x, v, state['trapping']['radius'], -state['grating']['position'])

    x = x[xi]
    v = v[vi]

    return x, v

def separate_axes(v):
    vx = v[[x for x in v.columns if 'x' in x]]
    vy = v[[x for x in v.columns if 'y' in x]]
    vz = v[[x for x in v.columns if 'z' in x]]

    return vx, vy, vz

def analyze_capture_velocity(v):
    vx, vy, vz = separate_axes(v)
    vmin = vx.iloc[0].min()
    vmax = vx.iloc[0].max()
    if vmax is np.nan:
        vmax = 0
    if vmin is np.nan:
        vmin = 0
    return vmin, vmax


def cost(state, params={'atoms':100, 'vmin':5, 'vmax':30}):
    ''' Prepares a MOT and runs a simulation to determine maximum capture velocity '''
    mot, coils = prepare(state)

    ''' DEFINE INITIAL CONDITIONS '''
    N_atoms = params['atoms']
    X = np.zeros((N_atoms, 3))
    X[:,0] = -2*state['trapping']['radius']*np.ones(N_atoms)       # initial position
    V = np.zeros((N_atoms, 3))
    V[:,0] = np.linspace(params['vmin'],params['vmax'], N_atoms)

    ''' Integrate atomic trajectories '''
    x, v = simulate(X, V, mot)
#        df = x.join(v)
#        plot_trajectory(df, axis=0)
    vmin, vmax = analyze_capture_velocity(v)
    efficiency = trapping_fraction(vmin, vmax, state['oven']['T'])
    print(vmin, vmax, efficiency)
    return efficiency
    
    
if __name__ == '__main__':
    mot, coils = prepare(state)