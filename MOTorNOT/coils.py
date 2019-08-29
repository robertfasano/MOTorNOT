import numpy as np
from scipy.special import ellipeinc, ellipk
from scipy.constants import mu_0
def assembleCoil(wire_diameter, turns, R, Z0, I, axis):
    coils = []
    for t in range(turns):
        coils.append(Coil(R, Z0, 1, I, axis))
        Z0 += np.sign(Z0)*wire_diameter
    return Coils(coils)

class Coils():
    def __init__(self, coils):
        self.coils = coils

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

class Coil():
    def __init__(self, radius, offset, turns, current, axis):
        ''' Creates a virtual coil.
            Args:
                radius (float): coil radius
                offset (float): offset from the origin in mm
                turns (int): number of turns
                current (float): current
                axis (int): 0, 1, or 2 to point the coil along the x, y, or z axis
        '''
        self.R = radius
        self.Z0 = offset
        self.N = turns
        self.I = current
        self.axis = axis


    def power(self, d):
        ''' Returns the power required to operate this coil as a function of the diameter d '''
        length = 2*np.pi*self.R*self.N
        resistivity = 1.68e-8
        resistance = resistivity*length/np.pi/(d/2)**2
        return np.abs(self.I)**2*resistance

    def field(self, X, V = None):
        ''' Numerically evaluates the field for a coil placed a distance self.Z0 from the origin along the axis of choice. Axes other than z are
            handled by rotating the coordinate system, solving along the symmetry axis, then rotating back. '''
        X = np.atleast_2d(X)
        if self.axis == 0:
            ''' Apply -90 degree rotation around y '''
            Ry = np.array([[0,0,-1],[0,1,0], [1,0,0]])
            X = np.dot(Ry, X.T).T
        elif self.axis == 1:
            ''' Apply 90 degree rotation around x '''
            Rx = np.array([[1,0,0],[0,0,-1], [0,1,0]])
            X = np.dot(Rx, X.T).T
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        r = np.sqrt(x**2+y**2)

        field = np.zeros(X.shape)
        alpha=r/self.R
        beta=(z-self.Z0)/self.R
        Q=(1+alpha)**2+beta**2
        m=4*alpha/Q

        gamma = np.zeros(X.shape[0])
        nonzero_indices = np.where(r != 0)[0]
        gamma[nonzero_indices] = (z[nonzero_indices]-self.Z0)/r[nonzero_indices]


        E_integral = ellipeinc(np.pi/2, m)
        K_integral = ellipk(m)

        prefactor = mu_0*self.N*self.I/(2*np.pi*self.R*Q)
        transverse_part = gamma*((1+alpha**2+beta**2)/(Q-4*alpha)*E_integral-K_integral)
        axial_part = ((1-alpha**2-beta**2)/(Q-4*alpha)*E_integral+K_integral)

        transverse_field = prefactor*transverse_part
        axial_field = prefactor*axial_part

        if len(nonzero_indices) > 0:
            field[:,0] = (transverse_field[nonzero_indices] * x[nonzero_indices]/r[nonzero_indices])
            field[:,1] = (transverse_field[nonzero_indices] * y[nonzero_indices]/r[nonzero_indices])
        field[:,2] = axial_field

        ''' Rotate to correct axis '''
        if self.axis == 0:
            ''' Apply 90 degree rotation around y '''
            Ry = np.array([[0,0,1],[0,1,0], [-1,0,0]])
            return np.dot(Ry,field.T).T
        elif self.axis == 1:
            ''' Apply -90 degree rotation around x '''
            Rx = np.array([[1,0,0],[0,0,1], [0,-1,0]])
            return np.dot(Rx,field.T).T
        return field

    def plot(self):
        from MOTorNOT.plotting import subplots
        subplots(self.field, numpoints=plot_params['numpoints'], label = 'B', units = 'G', scale = 1e4)
