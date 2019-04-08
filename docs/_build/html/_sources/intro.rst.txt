#################
Getting started
#################

Defining parameters
---------------------
All parameters for the simulation are packaged into several different dictionaries
in the ``parameters.py`` file. The ``constants`` dict stores physical constants. The
``atoms`` dict stores atom-specific properties, such as the mass and transition
properties like the wavelength and linewidth. All other parameters are packaged into
the ``state`` dict, which nests all variables into a format compatible
with EMERGENT (more on that later). An example of instantiating a virtual MOT object
is given by the ``prepare()`` method in ``run.py``:

.. code-block :: python

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

You can see above how the state dict variables are used to create the MOT.
In particular, here we instantiate two Coil objects, which provide methods for
calculating the magnetic field surrounding a wire loop. Next, we create an instance
of the ``gratingMOT`` class from ``gmot.py``, which simulates the beam geometry
and scattering forces for a grating MOT with the given state.

Visualizing the MOT
--------------------
After creating the ```coils`` objects, either with the ``prepare`` method
above or with your own definitions, you can call ``coils.plot()`` to display the
magnetic field in the vicinity of the MOT. Similarly, ``mot.plot()`` shows the
spatial Zeeman acceleration profile of the MOT.

Running a simulation
---------------------
In order to run a simulation, you should define a method with two arguments: the
state dict as defined in ``parameters.py`` and a params dict containing any other
arguments you want to pass into the simulation (such as timestep or total duration).
You'll find a basic virtual experiment in
``run.py``:

.. code-block :: python

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
      vmin, vmax = analyze_capture_velocity(v)
      efficiency = trapping_fraction(vmin, vmax, state['oven']['T'])
      print(vmin, vmax, efficiency)
      return efficiency

The first thing this method does is to set up a grating MOT with the ``prepare()`` method.
Next, ``cost()`` sets up an ensemble of atoms at a location defined by the beam geometry
in the ``state`` dict and with velocities linearly spaced between two values contained
in the ``params`` dict. Next, it calls ``simulate()`` to integrate the atomic
trajectories and determine the minimum and maximum capture velocities, before
finally determining the capture fraction as an integral over the Maxwell-Boltzmann
distribution at a user-specified temperature.

In order to run the simulation, you can call the method at the bottom of the
``run.py`` file:

.. code-block :: python

  if __name__ == '__main__':
    efficiency = cost(state)
    print(efficiency)

The ``state`` dict is automatically imported from your ``parameters.py`` file
before launching the simulation.
