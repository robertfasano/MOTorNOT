import numpy as np
import attr
from scipy.constants import hbar, physical_constants
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
from MOTorNOT.beams import *
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/cm^2
linewidth = 2*np.pi*atom['gamma']
wavenumber = 2*np.pi/(atom['wavelength']*1e-9)

@attr.s
class MOT:
    ''' This class calculates the scattering rate and optical forces arising
        from a set of beams and a magnetic field.
    '''
    beams = attr.ib(converter=list)
    field = attr.ib()

    def acceleration(self, X, V):
        return self.force(X,V)/(atom['mass'] * amu)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def beam_scattering_rate(self, beam, X, V):
        ''' The scattering rate of the beam at a given position and velocity.
            Args:
                X (ndarray): position array with shape (N, 3)
                V (ndarray): velocity array with shape (N, 3)
                b (ndarray): magnetic field evaluated at the position
                betaT (ndarray): total saturation fraction evaluated at X
        '''
        wavevector = beam.direction * wavenumber
        prefactor = linewidth/2 * beam.intensity(X)/Isat
        summand = 0
        b = self.field(X)
        eta = self.eta(b, beam.direction, beam.handedness)
        betaT = self.total_intensity(X)/Isat
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/linewidth**2*(beam.detuning-np.dot(wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate

    def scattering_rate(self, X, V, i=None):
        rate = 0
        if i is not None:
            return self.beam_scattering_rate(self.beams[i], X, V)
        for beam in self.beams:
            rate += self.beam_scattering_rate(beam, X, V)
        return rate

    def force(self, X, V):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/Isat
        b = self.field(X)
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        for beam in self.beams:
            # force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), wavenumber * beam.direction)
            force += hbar* np.outer(self.beam_scattering_rate(beam, X, V), wavenumber * beam.direction)

        return force

    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50, quiver_scale=30):
        from MOTorNOT.plotting import plot_2D
        fig = plot_2D(self.acceleration, plane=plane, limits=limits, numpoints=numpoints, quiver=True, quiver_scale=quiver_scale)
        fig.show()

    def phase_plot(self, axis='x', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50):
        from MOTorNOT.plotting import plot_phase_space_force
        import plotly.graph_objs as go

        surf = plot_phase_space_force(self.acceleration, axis=axis, limits=limits, numpoints=numpoints)
        fig = go.Figure([surf])

        fig.show()

    @staticmethod
    def eta(b, khat, s):
        ''' Transition amplitude to states [-1, 0, 1].
            Args:
                b (ndarray): magnetic field, shape (N,3) array
                khat (ndarray): beam unit vector
                s (float): polarization handedness
        '''
        bT = b.T
        bnorm = np.linalg.norm(bT, axis=0)
        # Bhat = np.divide(bT, bnorm, where=bnorm!=0)
        Bhat = (bT/bnorm)
        xi = khat.dot(Bhat)
        return np.array([(1+s*xi)**2/4, (1-xi**2)/2, (1-s*xi)**2/4]).T



class SixBeam(MOT):
    def __init__(self, power, radius, detuning, handedness, field):
        beams = []
        directions = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]
        for d in directions:
            beam = Beam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = handedness)
            beams.append(beam)

        directions = [[0, 0, 1], [0, 0, -1]]
        for d in directions:
            beam = Beam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = -handedness)
            beams.append(beam)
        super().__init__(beams, field=field)

class GratingMOT(MOT):
    def __init__(self, position, alpha, detuning, radius, power, handedness, R1, field, sectors=3, grating_radius=None, beam_type = 'uniform'):
        ''' Creates a virtual laser beam. Params dict should contain the following fields:
                position (float): grating offset from z=0
                alpha (float): diffraction angle in degrees
                radius (float): radius of incident beam
                detuning (float): detuning of incident beam
                field (method): function returning the magnetic field at a position vector X
                power (float): power of the incident beam
                handedness (float): +/-1 for circular polarization
                R1 (float): diffraction efficiency
        '''
        self.field = field
        alpha *= np.pi/180
        self.beams = []
        for n in np.linspace(1, sectors, sectors):
            self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius))
        for n in np.linspace(-1, -sectors, sectors):
            self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius))

        if grating_radius is None:
            grating_radius = 2*radius
        beam_params = {'direction': np.array([0,0,-1]),
                       'power': power,
                        'radius': radius,
                        'detuning': detuning,
                        'handedness': handedness,
                        'cutoff': grating_radius}
        if beam_type == 'uniform':
            self.beams.append(UniformBeam(**beam_params))
        elif beam_type == 'gaussian':
            self.beams.append(GaussianBeam(**beam_params))

        super().__init__(self.beams, self.field)
