import numpy as np
from scipy.constants import hbar, physical_constants
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT import load_parameters
atom = load_parameters()['atom']
Isat = atom['Isat'] * 10   # convert from mW/cm^2 to W/cm^2

class Beam():
    def __init__(self, direction, power, radius, detuning, handedness, origin = np.array([0,0,0])):
        ''' Creates a virtual laser beam.
            Args:
                direction (array-like): a unit vector representing the beam's direction
                power (float)
                radius (float): radius of the beam. Beams are currently treated as uniform intensity within the radius.
                detuning (float)
                handedness (float): +/- 1 for circular polarization.
        '''
        self.direction = direction
        wavenumber = 2*np.pi/(atom['wavelength']*1e-9)
        self.wavevector = direction*wavenumber
        self.power = power
        self.radius = radius
        self.detuning = detuning
        self.handedness = handedness

        self.I = self.power/np.pi/self.radius**2
        self.beta = self.I / Isat
        self.origin = origin



    def exists_at(self, X):
        ''' A boolean check for whether or not the beam exists at position X. Only works for beams along the x, y, or z axes; arbitrary directions will be supported later. Also assumes that the beam passes through the origin. '''
        X0 = X-self.origin
        return np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1) < self.radius

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
        return self.exists_at(X) * self.I

    def scattering_rate(self, X, V, b, betaT):
        linewidth = 2*np.pi*atom['gamma']
        prefactor = linewidth/2 * self.intensity(X)/Isat
        summand = 0
        eta = self.eta(b)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/linewidth**2*(self.detuning-np.dot(self.wavevector, V.T)-mF*atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate

class GaussianBeam(Beam):
    def __init__(self, params):
        super().__init__(params)

    def intensity(self, X):
        X0 = X-self.origin
        r = np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1)
        w = self.radius
        return self.I*np.exp(-2*r**2/w**2)

class Beams():
    def __init__(self, beams, field):
        self.beams = beams
        self.field = field

    def acceleration(self, X, V):
        mass = atom['mass'] * amu
        return self.force(X,V)/mass

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def recoil_serial(self, X, V, duration):
        ''' Simulate random momentum kicks from all beams summed over a duration. Since the duration may be slower
            than the scattering rate, we exploit scale invariance to reduce the timestep by a factor and the step
            size by the square root of the same factor.'''
        diffusion_constant = (hbar*k)**2 * self.scattering_rate(X,V)
        num_steps = 1000
        target_step_time = duration / num_steps
        step_size = np.sqrt(diffusion_constant*target_step_time)
        for i in range(num_steps):
            ''' Generate random momentum kick '''
            random_vector = np.random.normal(0,1, X.shape)
            random_vector /= np.linalg.norm(random_vector, axis=1)[:,np.newaxis]
            V += step_size[:,np.newaxis] * random_vector
        return V

    def recoil(self, X, V, duration, full_output = False):
        ''' Simulate random momentum kicks from all beams summed over a duration. Since the duration may be slower
            than the scattering rate, we exploit scale invariance to reduce the timestep by a factor and the step
            size by the square root of the same factor.'''
        diffusion_constant = (hbar*k)**2 * self.scattering_rate(X,V)
        num_steps = 1000
        target_step_time = duration / num_steps
        step_size = np.sqrt(diffusion_constant*target_step_time)
        ''' Generate random momentum kick '''
        random_vector = np.random.normal(0,1, [X.shape[0], X.shape[1], num_steps])
        random_vector /= np.linalg.norm(random_vector, axis=1)[:,np.newaxis]
        kick = (step_size[:,np.newaxis, np.newaxis] * random_vector)

        if full_output:
            return kick
        else:
            return kick.sum(axis=2)

    def scattering_rate(self, X, V):
        rate = 0
        for beam in self.beams:
            rate += beam.scattering_rate(X, V, self.field(X), self.total_intensity(X))
        return rate

    def force(self, X, V):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/Isat
        b = self.field(X)
        for beam in self.beams:
            force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), beam.wavevector)
        return force

    def plot(self):
        from MOTorNOT.plotting import plot_2D
        plot_2D(self.force, numpoints=30, quiver=True)

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

def prepare_slowing_beam(beams = [], axis = 0, origin = np.array([0,0,0])):
    vec = k*np.array([0,0,0])
    vec[axis] = 1
    beams.append(Beam(wavevector = vec, power = power['slow'], radius = radius['slow'], detuning = detuning['slow'], handedness = 1, origin = origin))
    return beams

def prepare_two_beam(beams = [], theta = 45, phi = np.pi/2):
    theta *= np.pi/180
    for vec in [k*np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)]),k*np.array([-np.cos(theta)*np.sin(phi),-np.sin(theta)*np.sin(phi),np.cos(phi)])]:
        beams.append(Beam(wavevector = vec, power = power['trap'], radius = radius['trap'], detuning = detuning['trap'], handedness = -1))
    return beams
