import numpy as np
import attr

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
    origin = attr.ib(default=np.array([0, 0, 0]))
    cutoff = attr.ib(default=None)
    infinite = attr.ib(default=False)

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
    def exists_at(self, X):
        ''' A boolean check for whether or not the beam exists at position X. Only works for beams along the x, y, or z axes; arbitrary directions will be supported later. Also assumes that the beam passes through the origin. '''
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0, self.direction), (self.direction)), axis=1)
        if self.cutoff is None:
            return r < self.radius
        else:
            return (r < self.radius) & (r < self.cutoff)

    def intensity(self, X):
        if self.infinite:
            return np.ones(len(X))*self.power/np.pi/self.radius**2
        return self.exists_at(X) * self.power/np.pi/self.radius**2

def between_angles(theta, a, b):
    theta = theta % (2*np.pi)
    a = a % (2*np.pi)
    b = b % (2*np.pi)
    if a < b:
        return np.logical_and(a <= theta, theta <= b)
    return np.logical_or(a <= theta, theta <= b)

class DiffractedBeam(Beam):
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0]), beam_type='uniform', sectors=3, grating_radius=None, infinite=False):
        self.beam_type = beam_type
        self.n = n
        self.sectors = sectors
        self.phi = (np.pi * (1 + (2*np.abs(n)+1)/sectors)) % (2*np.pi)
        self.direction = np.array([np.sign(n)*np.cos(self.phi)*np.sin(alpha),
                                   np.sign(n)*np.sin(self.phi)*np.sin(alpha),
                                   np.cos(alpha)])

        super().__init__(self.direction, power, radius, detuning, handedness, cutoff=grating_radius, infinite=infinite)
        self.alpha = alpha
        self.I = power/np.pi/radius**2
        self.z0 = -position
        self.grating_radius = grating_radius
        if grating_radius is None:
            self.grating_radius = radius

    def intensity(self, X):
        x = X.T[0]
        y = X.T[1]
        z = X.T[2]
        z0 = self.z0
        ## trace wavevector backwards to grating plane and calculate intensity
        x0 = x - (z-z0)*self.direction[0]/self.direction[2]
        y0 = y - (z-z0)*self.direction[1]/self.direction[2]
        r0 = np.sqrt(x0**2+y0**2)

        I = self.I
        if self.beam_type == 'gaussian':
            I *= np.exp(-2*r0**2/self.radius**2)

        if self.infinite: return np.ones(len(X))*I

        ## check if the in-plane point is in the sector; return 0 if not
        radial_inequality = (r0 <= self.radius) & (r0 <= self.grating_radius)
        phi = np.mod(np.arctan2(y0, x0), 2*np.pi)
        angular_inequality = between_angles(phi, self.phi + np.pi - np.pi/self.sectors, self.phi + np.pi + np.pi/self.sectors)
        axial_inequality = z > z0

        return I * radial_inequality * angular_inequality * axial_inequality

class PyramidBeam(Beam):
    def __init__(self, leg_length, separation, direction, power, radius, detuning, handedness):
        super().__init__(direction, power, radius, detuning, handedness)
        self.d = leg_length
        self.D = separation

    def intensity(self, X):
        x = X.T[0]
        y = X.T[1]
        z = X.T[2]
        I = self.power/np.pi/self.radius**2
        in_radius = (z+(self.d+self.D)/2)**2+y**2<self.radius**2
        in_z_range = (-self.d/2 < z) * (z < self.d/2)

        return I * in_radius * in_z_range

@attr.s
class GaussianBeam(Beam):
    def intensity(self, X):
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)
        w = self.radius
        I = self.power/np.pi/self.radius**2
        return I*np.exp(-2*r**2/w**2) * (r <= self.cutoff)
