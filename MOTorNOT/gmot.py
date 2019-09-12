import numpy as np
from MOTorNOT.beams import Beam, GaussianBeam
from scipy.constants import hbar
from MOTorNOT import load_parameters
from scipy.constants import hbar, physical_constants
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/cm^2

class gratingMOT():
    def __init__(self, position, alpha, detuning, radius, power, handedness, R1, field, sectors=3, beam_type = 'uniform'):
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
            self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors))
        for n in np.linspace(-1, -sectors, sectors):
            self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors))

        beam_params = {'direction': np.array([0,0,-1]),
                                'power': power,
                                'radius': radius,
                                'detuning': detuning,
                                'handedness': handedness}
        if beam_type == 'uniform':
            self.beams.append(Beam(**beam_params))
        elif beam_type == 'gaussian':
            self.beams.append(GaussianBeam(**beam_params))

    def acceleration(self, X, V):
        return self.force(X,V)/(atom['mass'] * amu)

    def overlap(self, X):
        overlap = True
        for beam in self.beams:
            overlap = np.logical_and(overlap, beam.exists_at(X))
        return int(overlap)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def force(self, X, V):
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/Isat
        b = self.field(X)
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        for beam in self.beams:
            wavevector = wavenumber * beam.direction
            force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), wavevector)
        return force


    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50, quiver_scale=30):
        from MOTorNOT.plotting import plot_2D
        fig = plot_2D(self.acceleration, plane=plane, limits=limits, numpoints=numpoints, quiver=True, quiver_scale=quiver_scale)
        fig.show()

def between_angles(theta, a, b):
    theta = theta % (2*np.pi)
    a = a % (2*np.pi)
    b = b % (2*np.pi)
    if a < b:
        return np.logical_and(a <= theta, theta <= b)
    return np.logical_or(a <= theta, theta <= b)

class diffractedBeam():
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0]), beam_type='uniform', sectors=3):
        self.beam_type = beam_type
        self.n = n
        self.sectors = sectors
        # self.phi = np.pi/3*(4*np.abs(n)-5)
        self.phi = (np.pi * (1 + (2*np.abs(n)+1)/sectors)) % (2*np.pi)
        # self.direction = np.array([-np.sign(n)*np.cos(self.phi)*np.sin(alpha),
        #                            np.sign(n)*np.sin(self.phi)*np.sin(alpha),
        #                            np.cos(alpha)])
        self.direction = np.array([np.sign(n)*np.cos(self.phi)*np.sin(alpha),
                                   np.sign(n)*np.sin(self.phi)*np.sin(alpha),
                                   np.cos(alpha)])
        self.power = power
        self.radius = radius
        self.detuning = detuning
        self.handedness = handedness
        self.alpha = alpha     # diffraction angle
        self.I = power/np.pi/radius**2
        self.beta = self.I / Isat
        self.z0 = -position

    def exists_at(self, X):
        x = X.T[0] - self.direction[0] * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.direction[1] * (X.T[2]-self.z0)/np.cos(self.alpha)
        r = np.sqrt(x**2+y**2)
        phi = np.mod(np.arctan2(y, x),2*np.pi)
        min_angle = self.phi + np.pi - np.pi/self.sectors
        max_angle = self.phi + np.pi + np.pi/self.sectors
        angular_inequality = between_angles(phi, min_angle, max_angle)
        radial_inequality = (r <= self.radius)
        # angular_inequality = (2*np.pi/self.sectors*(np.abs(self.n)-1) < phi) & (phi < 2*np.pi/self.sectors*np.abs(self.n))
        vertical_inequality = (X.T[2]-self.z0) > 0
        return radial_inequality & angular_inequality & vertical_inequality

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
        x = X.T[0] - self.direction[0] * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.direction[1] * (X.T[2]-self.z0)/np.cos(self.alpha)
        r = np.sqrt(x**2+y**2)
        phi = np.mod(np.arctan2(y, x),2*np.pi)
        min_angle = self.phi + np.pi - np.pi/self.sectors
        max_angle = self.phi + np.pi + np.pi/self.sectors
        angular_inequality = between_angles(phi, min_angle, max_angle)
        radial_inequality = (r <= self.radius)
        # angular_inequality = (2*np.pi/self.sectors*(np.abs(self.n)-1) < phi) & (phi < 2*np.pi/self.sectors*np.abs(self.n))
        vertical_inequality = (X.T[2]-self.z0) > 0

        return self.I*np.exp(-2*r**2/self.radius**2)*angular_inequality

    def scattering_rate(self, X, V, b, betaT):
        linewidth = 2*np.pi*atom['gamma']
        if self.beam_type == 'gaussian':
            prefactor = linewidth/2 * self.intensity(X)/Isat
        else:
            prefactor = linewidth/2 * self.exists_at(X)*self.beta
        summand = 0
        eta = self.eta(b, self.direction, self.handedness)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
            wavevector = self.direction * wavenumber
            denominator = (1+betaT+4/linewidth**2*(self.detuning-np.dot(wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate
