import numpy as np
import attr
from MOTorNOT import rotate
from scipy.special import ellipeinc, ellipk
from scipy.constants import mu_0

def assembleCoil(wire_diameter, turns, R, Z0, I, axis):
    coils = []
    for t in range(turns):
        coils.append(Coil(R, Z0, 1, I, axis))
        Z0 += np.sign(Z0)*wire_diameter
    return Coils(coils)

@attr.s
class Coils():
    coils = attr.ib(default=[])

    def append(self, coil):
        self.coils.append(coil)

    def field(self, X, V = None):
        ''' Compute the total field of all coils '''
        field = np.zeros(X.shape)
        for coil in self.coils:
            field += coil.field(X, V)
        return field

    def plot(self):
        from MOTorNOT.plotting import subplots
        subplots(self.field, numpoints=plot_params['numpoints'], label = 'B', units = 'G', scale = 1e4)

    def gradient(self, X, axis='z'):
        ''' Evaluates the gradient at a point X along a given axis using a
            finite difference approximation. '''
        X = np.atleast_2d(X)
        dX = np.zeros(X.shape)
        dX[:, {'x': 0, 'y': 1, 'z': 2}[axis]] = 1e-4
        return (self.field(X+dX) - self.field(X-dX)) / 2e-4   # units: T/m

class QuadrupoleCoils(Coils):
    def __init__(self, radius, offset, turns, current, axis, deltaI=0):
        ''' Creates a pair of coils with equal and opposite offsets and currents. '''
        coil1 = Coil(radius, offset, turns, current+deltaI/2, axis)
        coil2 = Coil(radius, -offset, turns, -current+deltaI/2, axis)
        super().__init__(coils=[coil1, coil2])

def div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.divide(a, b)
        res[b==0] = 0
    return res

@attr.s
class Coil():
    ''' Creates a virtual coil.
        Args:
            radius (float): coil radius
            offset (float): offset from the origin in mm
            turns (int): number of turns
            current (float): current
            axis (int): 0, 1, or 2 to point the coil along the x, y, or z axis
    '''
    radius = attr.ib(converter=float)
    offset = attr.ib(converter=float)
    turns = attr.ib(converter=float)
    current = attr.ib(converter=float)
    axis = attr.ib(converter=int)

    def power(self, d):
        ''' Returns the power required to operate this coil as a function of the diameter d '''
        length = 2*np.pi*self.radius*self.turns
        resistivity = 1.68e-8
        resistance = resistivity*length/np.pi/(d/2)**2
        return np.abs(self.current)**2*resistance

    def field(self, X, V = None):
        ''' Numerically evaluates the field for a coil placed a distance self.Z0 from the origin along the axis of choice. Axes other than z are
            handled by rotating the coordinate system, solving along the symmetry axis, then rotating back. '''
        X = np.atleast_2d(X)
        if self.axis == 0:
            ''' Apply -90 degree rotation around y '''
            X = np.dot(rotate(1, -np.pi/2), X.T).T
        elif self.axis == 1:
            ''' Apply 90 degree rotation around x '''
            X = np.dot(rotate(0, np.pi/2), X.T).T
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        r = np.sqrt(x**2+y**2)

        field = np.zeros(X.shape)
        alpha = r/self.radius
        beta = (z-self.offset)/self.radius
        Q = (1+alpha)**2+beta**2
        m = 4*div(alpha, Q)

        gamma = div(z-self.offset, r)
        E_integral = ellipeinc(np.pi/2, m)
        K_integral = ellipk(m)

        prefactor = mu_0*self.turns*self.current/(2*np.pi*self.radius*Q)
        transverse_field = prefactor*gamma*((1+alpha**2+beta**2)/(Q-4*alpha)*E_integral-K_integral)
        axial_field = prefactor*((1-alpha**2-beta**2)/(Q-4*alpha)*E_integral+K_integral)

        field[:, 0] = transverse_field * div(x, r)
        field[:, 1] = transverse_field * div(y, r)
        field[:,2] = axial_field

        ''' Rotate to correct axis '''
        if self.axis == 0:
            return np.dot(rotate(1, np.pi/2), field.T).T
        elif self.axis == 1:
            return np.dot(rotate(0, -np.pi/2), field.T).T
        return field
