import numpy as np
import attr
from scipy.constants import hbar, physical_constants
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/cm^2

@attr.s
class Beam:
    ''' Prototype class for describing laser beams. Subclasses for Uniform,
        Gaussian, or Diffracted beams will implement their own methods describing
        their spatial intensity variation.
    '''
    direction = attr.ib(converter=np.array)
    power = attr.ib(converter=float)
    radius = attr.ib(converter=float)
    detuning = attr.ib(converter=float)
    handedness = attr.ib(converter=int)
    cutoff = attr.ib(default=None)

@attr.s
class UniformBeam(Beam):
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

    def intensity(self, X):
        return self.exists_at(X) * self.power/np.pi/self.radius**2

def between_angles(theta, a, b):
    theta = theta % (2*np.pi)
    a = a % (2*np.pi)
    b = b % (2*np.pi)
    if a < b:
        return np.logical_and(a <= theta, theta <= b)
    return np.logical_or(a <= theta, theta <= b)

class DiffractedBeam(Beam):
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0]), beam_type='uniform', sectors=3, grating_radius=None):
        self.beam_type = beam_type
        self.n = n
        self.sectors = sectors
        self.phi = (np.pi * (1 + (2*np.abs(n)+1)/sectors)) % (2*np.pi)
        self.direction = np.array([np.sign(n)*np.cos(self.phi)*np.sin(alpha),
                                   np.sign(n)*np.sin(self.phi)*np.sin(alpha),
                                   np.cos(alpha)])
        super().__init__(self.direction, power, radius, detuning, handedness, cutoff=grating_radius)
        self.alpha = alpha
        self.I = power/np.pi/radius**2
        self.beta = self.I / Isat
        self.z0 = -position
        self.grating_radius = grating_radius
        if grating_radius is None:
            self.grating_radius = radius

    def intensity(self, X):
        x = X.T[0] - self.direction[0] * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.direction[1] * (X.T[2]-self.z0)/np.cos(self.alpha)
        r = np.sqrt(x**2+y**2)
        phi = np.mod(np.arctan2(y, x),2*np.pi)
        min_angle = self.phi + np.pi - np.pi/self.sectors
        max_angle = self.phi + np.pi + np.pi/self.sectors
        radial_inequality = (r <= self.radius) & (r <= self.grating_radius)
        angular_inequality = between_angles(phi, min_angle, max_angle)
        vertical_inequality = (X.T[2]-self.z0) > 0
        if self.beam_type == 'gaussian':
            I = self.I*np.exp(-2*r**2/self.radius**2)*angular_inequality
        else:
            I = radial_inequality * angular_inequality * vertical_inequality * self.I
        return (r <= self.grating_radius) * I

class GaussianBeam(Beam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def intensity(self, X):
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)
        w = self.radius
        I = self.power/np.pi/self.radius**2
        return I*np.exp(-2*r**2/w**2) * (r <= self.cutoff)
