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
    def __init__(self, position, alpha, detuning, radius, power, handedness, R1, field, beam_type = 'uniform'):
        ''' Creates a virtual laser beam. Params dict should contain the following fields:
                position (float): grating offset from z=0
                alpha (float): diffraction angle
                radius (float): radius of incident beam
                detuning (float): detuning of incident beam
                field (method): function returning the magnetic field at a position vector X
                power (float): power of the incident beam
                handedness (float): +/-1 for circular polarization
                R1 (float): diffraction efficiency
        '''
        self.field = field

        self.beams = []
        for n in [1,2,3]:
            self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type))
        for n in [-1, -2, -3]:
            self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type))
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)

        beam_params = {'wavevector': wavenumber*np.array([0,0,-1]),
                                'power': power,
                                'radius': radius,
                                'detuning': detuning,
                                'handedness': handedness}
        if beam_type == 'uniform':
            self.beams.append(Beam(**beam_params))
        elif beam_type == 'gaussian':
            self.beams.append(GaussianBeam(**beam_params))



    def acceleration(self, X, V):
        mass = atom['mass'] * amu
        return self.force(X,V)/mass

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
        for beam in self.beams:
            force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), beam.wavevector)
        return force

class diffractedBeam():
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0]), beam_type='uniform'):
        self.beam_type = beam_type
        self.n = n
        self.phi = np.pi/3*(4*np.abs(n)-5)
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        self.wavevector = wavenumber*np.array([-np.sign(n)*np.cos(self.phi)*np.sin(alpha), np.sign(n)*np.sin(self.phi)*np.sin(alpha), np.cos(alpha)])
        self.power = power
        self.radius = radius
        self.detuning = detuning
        self.handedness = handedness
        self.alpha = alpha     # diffraction angle
        self.I = power/np.pi/radius**2
        self.beta = self.I / Isat
        self.z0 = -position

    def exists_at(self, X):
        x = X.T[0] - self.wavevector[0]/np.linalg.norm(self.wavevector) * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.wavevector[1]/np.linalg.norm(self.wavevector) * (X.T[2]-self.z0)/np.cos(self.alpha)
        r = np.sqrt(x**2+y**2)
        phi = np.mod(np.arctan2(y, x),2*np.pi)

        radial_inequality = (r <= self.radius)
        angular_inequality = (2*np.pi/3*(np.abs(self.n)-1) < phi) & (phi < 2*np.pi/3*np.abs(self.n))
        vertical_inequality = (X.T[2]-self.z0) > 0
        return radial_inequality & angular_inequality & vertical_inequality

    def eta(self, b):
        xi = 0
        b = b.copy().T
        if np.linalg.norm(b) != 0:
            khat = self.wavevector/np.linalg.norm(self.wavevector)
            Bhat = (b/np.linalg.norm(b,axis=0)).T
            khat = np.stack([khat]*Bhat.shape[0])
            xi = (khat*Bhat).sum(1)
        eta = {}
        eta[0]=(1-xi**2)/2
        eta[-1] = (1+self.handedness*xi)**2/4
        eta[1] = (1-self.handedness*xi)**2/4

        return np.array([eta[-1], eta[0], eta[1]]).T

    def intensity(self, X):
        x = X.T[0] - self.wavevector[0]/np.linalg.norm(self.wavevector) * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.wavevector[1]/np.linalg.norm(self.wavevector) * (X.T[2]-self.z0)/np.cos(self.alpha)
        r = np.sqrt(x**2+y**2)
        phi = np.mod(np.arctan2(y, x),2*np.pi)

        radial_inequality = (r <= self.radius)
        angular_inequality = (2*np.pi/3*(np.abs(self.n)-1) < phi) & (phi < 2*np.pi/3*np.abs(self.n))
        vertical_inequality = (X.T[2]-self.z0) > 0

        return self.I*np.exp(-2*r**2/self.radius**2)*angular_inequality

    def scattering_rate(self, X, V, b, betaT):
        linewidth = 2*np.pi*atom['gamma']
        if self.beam_type == 'gaussian':
            prefactor = linewidth/2 * self.intensity(X)/Isat
        else:
            prefactor = linewidth/2 * self.exists_at(X)*self.beta
        summand = 0
        eta = self.eta(b)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/linewidth**2*(self.detuning-np.dot(self.wavevector, V.T)-mF*atom['gF']*muB*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate
