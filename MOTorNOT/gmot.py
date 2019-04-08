import numpy as np
from MOTorNOT.parameters import constants, atom, plot_params
from MOTorNOT.beams import Beam

class gratingMOT():
    def __init__(self, params, show_incident = True, show_positive = True, show_negative = True):
        ''' Creates a virtual laser beam. Params dict should contain the following fields:
                position (float): grating offset from z=0
                alpha (float): diffraction angle
                radius (float): radius of incident beam
                detuning (float): detuning of incident beam
                field (method): function returning the magnetic field at a position vector X
                power (float): power of the incident beam
                polarization (float): +/-1 for circular polarization
                R1 (float): diffraction efficiency
        '''
        position = params['position']
        alpha = params['alpha']
        detuning = params['detuning']
        radius = params['radius']
        self.field = params['field']
        power = params['power']
        polarization = params['polarization']
        R1 = params['R1']

        self.beams = []
        if show_positive:
            for n in [1,2,3]:
                self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -polarization, position))
        if show_negative:
            for n in [-1, -2, -3]:
                self.beams.append(diffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -polarization, position))
        if show_incident:
            self.beams.append(Beam({'wavevector': atom['k']*np.array([0,0,-1]),
                                    'power': power,
                                    'radius': radius,
                                    'detuning': detuning,
                                    'handedness': polarization}))


    def acceleration(self, X, V):
        return self.force(X,V)/atom['m']

    def overlap(self, X):
        overlap = True
        for beam in self.beams:
            overlap = np.logical_and(overlap, beam.exists_at(X))
        return int(overlap)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.exists_at(X) * beam.intensity
        return It

    def force(self, X, V):
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/atom['Isat']
        b = self.field(X)
        for beam in self.beams:
            force += constants['hbar']* np.outer(beam.scattering_rate(X,V, b, betaT), beam.wavevector)
        return force

    def plot(self):
        from MOTorNOT.plotting import subplots
        subplots(self.acceleration, numpoints=plot_params['numpoints'], label='a', units = r'm/s^2')

class diffractedBeam():
    def __init__(self, n, alpha, power, radius, detuning, handedness, position, origin = np.array([0,0,0])):
        self.n = n
        self.phi = np.pi/3*(4*np.abs(n)-5)
        self.wavevector = atom['k']*np.array([-np.sign(n)*np.cos(self.phi)*np.sin(alpha), np.sign(n)*np.sin(self.phi)*np.sin(alpha), np.cos(alpha)])
        self.power = power
        self.radius = radius
        self.detuning = detuning
        self.handedness = handedness
        self.alpha = alpha     # diffraction angle
        self.intensity = power/np.pi/radius**2
        self.beta = self.intensity / atom['Isat']
        self.z0 = -position

    def exists_at(self, X):
        x = X.T[0] - self.wavevector[0]/atom['k'] * (X.T[2]-self.z0)/np.cos(self.alpha)
        y = X.T[1] - self.wavevector[1]/atom['k'] * (X.T[2]-self.z0)/np.cos(self.alpha)
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

    def scattering_rate(self, X, V, b, betaT):
        prefactor = atom['gamma']/2 * self.exists_at(X)*self.beta
        summand = 0
        eta = self.eta(b)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/atom['gamma']**2*(self.detuning-np.dot(self.wavevector, V.T)-mF*atom['mu']*np.linalg.norm(b,axis=1)/constants['hbar'])**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate
