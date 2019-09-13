import numpy as np
import attr
from scipy.constants import hbar, physical_constants
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/cm^2

@attr.s
class Beam():
    ''' Creates a virtual laser beam.
        Args:
            direction (array-like): a unit vector representing the beam's direction
            power (float)
            radius (float): radius of the beam. Beams are currently treated as uniform intensity within the radius.
            detuning (float)
            handedness (int): +/- 1 for circular polarization.
    '''
    direction = attr.ib(converter=np.array)
    power = attr.ib(converter=float)
    radius = attr.ib(converter=float)
    detuning = attr.ib(converter=float)
    handedness = attr.ib(converter=int)
    origin = attr.ib(default=np.array([0, 0, 0]))
    cutoff = attr.ib(default=None)

    def exists_at(self, X):
        ''' A boolean check for whether or not the beam exists at position X. Only works for beams along the x, y, or z axes; arbitrary directions will be supported later. Also assumes that the beam passes through the origin. '''
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        if self.cutoff is None:
            return r < self.radius
        else:
            return (r < self.radius) & (r < self.cutoff)

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

    def intensity(self, X):
        return self.exists_at(X) * self.power/np.pi/self.radius**2

    def scattering_rate(self, X, V, b, betaT):
        ''' The scattering rate of the beam at a given position and velocity.
            Args:
                X (ndarray): position array with shape (N, 3)
                V (ndarray): velocity array with shape (N, 3)
                b (ndarray): magnetic field evaluated at the position
                betaT (ndarray): total saturation fraction evaluated at X
        '''
        linewidth = 2*np.pi*atom['gamma']
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        wavevector = self.direction * wavenumber
        prefactor = linewidth/2 * self.intensity(X)/Isat
        summand = 0
        eta = self.eta(b, self.direction, self.handedness)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/linewidth**2*(self.detuning-np.dot(wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate

class GaussianBeam(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def intensity(self, X):
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)
        w = self.radius
        I = self.power/np.pi/self.radius**2
        return I*np.exp(-2*r**2/w**2) * (r <= self.cutoff)

@attr.s
class Beams():
    beams = attr.ib(converter=list)
    field = attr.ib()

    def acceleration(self, X, V):
        return self.force(X,V)/(atom['mass'] * amu)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def scattering_rate(self, X, V, i=None):
        rate = 0
        if i is not None:
            return self.beams[i].scattering_rate(X, V, self.field(X), self.total_intensity(X))

        for beam in self.beams:
            rate += beam.scattering_rate(X, V, self.field(X), self.total_intensity(X))
        return rate

    def force(self, X, V):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/Isat
        b = self.field(X)
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        for beam in self.beams:
            force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), wavenumber * beam.direction)
        return force

    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)]):
        from MOTorNOT.plotting import plot_2D
        fig = plot_2D(self.acceleration, plane=plane, limits=limits, numpoints=40, quiver=True)
        fig.show()

class SixBeamMOT(Beams):
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
