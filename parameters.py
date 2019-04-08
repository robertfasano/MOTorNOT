''' This script contains all parameters used for a calculation. In order to load them in, just save
    this script and then include "from parameters import *" in whatever other script the calculation takes place in. '''
import numpy as np
#from atoms import *
from scipy.special import ellipeinc, ellipk

'''  CONSTANTS '''
constants = {
    'm': 171*1.67e-27,
    'hbar': (6.626/2/np.pi)*1e-34,
    'kB': 1.381e-23,
    'muB': 9.274e-24,
    'mu0': 4*np.pi * 1e-7
        }

''' ATOM PARAMETERS '''
atom = {
    'm': 171*1.67e-27,
    'gamma': 2*np.pi*29e6,
    'wavelength': 399e-9,
    'gF': 1,
    'Isat': 600             # saturation intensity is 60 mW/cm^2 or 600 W/m^2
        }

atom['k'] = 2*np.pi/atom['wavelength']
atom['mu']= atom['gF']*constants['muB']


''' BEAM PARAMETERS'''
trapping = {'radius': 6e-3, 
            'power': 10e-3, 
            'detuning': -2*np.pi*15e6, 
            'polarization': -1}
slowing = {'radius': 1.5e-3, 
           'power': 3e-3, 
           'detuning': -2*np.pi*115e6, 
           'polarization': -1}

#intensity = {}
#beta = {}
#for beam in [trapping, slowing]:
#    beam['intensity'] = beam['power']/np.pi/beam['radius']**2
#    beam['beta'] = beam['intensity'] / atom['Isat']

''' GRATING PARAMETERS '''
grating = {'d': 600e-9, 'R1': 0.33}
alpha = np.arcsin(atom['wavelength']/grating['d'])
grating['position'] = 2*trapping['radius']/np.tan(alpha)/4

''' PLOTTING PARAMETERS'''
factor = 1
plot_params = {
    'axlim': {'x':[-factor*trapping['radius'], factor*trapping['radius']], 
             'y':[-factor*trapping['radius'], factor*trapping['radius']], 
             'z':[-grating['position'], factor*trapping['radius']]},
    'numpoints': 100,
    'contours': 100
    }

''' MAGNETIC FIELD PARAMETERS '''
coil1 = {'R': 8.0587/100, 'z': -3.6605/100, 'N': 51.1661, 'I': 61}
coil2 = {'R': 8.3645/100, 'z': 3.7488/100, 'N': 48.8691, 'I': 66}

''' OTHER PARAMETERS '''
oven = {'T': 475}

''' PREPARE OVERALL STATE '''
state = {'trapping': trapping, 'slowing': slowing, 'grating': grating, 'coil1': coil1, 'coil2': coil2, 'oven': oven}



if __name__ == '__main__':
    from gmot import gratingMOT
    from coils import Coil, Coils

    import matplotlib.pyplot as plt


    def prepare(state):
        ''' Prepares a MOT with settings defined by the passed state dict.
            You can visualize the coil field or MOT acceleration profile by calling 
            coils.plot() or mot.plot() respectively. '''
            
        ''' INSTANTIATE COILS '''
        Coil1 = Coil(state['coil1']['R'], state['coil1']['z'], state['coil1']['N'], -state['coil1']['I'], axis=2)
        Coil2 = Coil(state['coil2']['R'], state['coil2']['z'], state['coil2']['N'], state['coil2']['I'], axis=2)
        
        coils = Coils([Coil1, Coil2])
    
        ''' INSTANTIATE BEAMS '''
        alpha = np.arcsin(atom['wavelength']/grating['d'])
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
        
        return mot
        
    def simulate(X, V, mot):
        ''' Run a simulation'''
        from integration import Solver
        tmax = 20*np.abs(np.min(X)/np.min(V[:,0]))
        sol = Solver(X, V, mot.force, None, duration = tmax)
        x, v = sol.solve()
        xi,vi = sol.trapped_atoms(x, v)
    
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
        return vmin, vmax
    
    
    def cost(state):
        ''' Prepares a MOT and runs a simulation to determine maximum capture velocity '''
        mot = prepare(state)
        
        ''' DEFINE INITIAL CONDITIONS '''
        N_atoms = 100
        X = np.zeros((N_atoms, 3))
        X[:,0] = -2*trapping['radius']*np.ones(N_atoms)       # initial position
        V = np.zeros((N_atoms, 3))
        V[:,0] = np.linspace(5,30, N_atoms)
        
        ''' Integrate atomic trajectories '''
        x, v = simulate(X, V, mot)
#        df = x.join(v)
#        plot_trajectory(df, axis=0)
        vmin, vmax = analyze_capture_velocity(v)
        
        return vmax

    eff = np.linspace(.3, .4, 10)
    vmax = []
    for e in eff:
        state['grating']['R1'] = e
        vmax.append(cost(state))
    plt.plot(eff, vmax)




#    vx, vy, vz = separate_axes(v)
#    x, y, z = separate_axes(x)

#    sol.histogram(vx.iloc[-1].values)
#    sol.plot(x)
#    print('Trapping efficiency: %.0f%%'%(sol.trapping_efficiency(x)*100))

#    t, y = sol.solve_sde()
#    plt.plot(t,y[:,0])
    
