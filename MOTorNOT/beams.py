import numpy as np
from MOTorNOT.parameters import constants, atom, plot_params

class Beam():
    def __init__(self, wavevector, power, radius, detuning, handedness, origin = np.array([0,0,0])):
        ''' Creates a virtual laser beam.
            Args:
                wavevector (array-like): the 3D wavevector of the beam
                power (float)
                radius (float): radius of the beam. Beams are currently treated as uniform intensity within the radius.
                detuning (float)
                handedness (float): +/- 1 for circular polarization.
        '''
        self.wavevector = wavevector
        self.power = power
        self.radius = radius
        self.detuning = detuning
        self.handedness = handedness

        self.direction = self.wavevector / np.linalg.norm(self.wavevector)
        self.I = self.power/np.pi/self.radius**2
        self.beta = self.I / atom['Isat']
        self.origin = origin

        ''' Form a pair of vectors orthogonal to the wavevector '''
        r = np.array([1,1,1])
        self.orth1 = r-np.outer(np.dot(self.direction, r), self.direction)[0]
        self.orth1 = np.array([1,0,0])
        self.orth2 = np.cross(self.direction, self.orth1)

        self.orth1 = self.orth1 / np.linalg.norm(self.orth1)
        self.orth2 = self.orth2 /  np.linalg.norm(self.orth2)

    def exists_at(self, X):
        ''' A boolean check for whether or not the beam exists at position X. Only works for beams along the x, y, or z axes; arbitrary directions will be supported later. Also assumes that the beam passes through the origin. '''
        X0 = X-self.origin
        return np.logical_and(np.linalg.norm(-X0+np.outer(np.dot(X0,self.direction),(self.direction)),axis=1) < self.radius, self.angular_inequality(X0))


    def angular_inequality(self, X):
        phi = np.mod(np.arctan2(np.dot(X, self.orth2), np.dot(X, self.orth1)),2*np.pi)
        return True
#        return np.logical_and(0 < phi, phi < 2*np.pi)

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
        prefactor = atom['gamma']/2 * self.intensity(X)/atom['Isat']
        summand = 0
        eta = self.eta(b)
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/atom['gamma']**2*(self.detuning-np.dot(self.wavevector, V.T)-mF*atom['mu']*np.linalg.norm(b,axis=1)/constants['hbar'])**2)
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
        return self.force(X,V)/atom['m']

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def recoil_serial(self, X, V, duration):
        ''' Simulate random momentum kicks from all beams summed over a duration. Since the duration may be slower
            than the scattering rate, we exploit scale invariance to reduce the timestep by a factor and the step
            size by the square root of the same factor.'''
        diffusion_constant = (constants['hbar']*k)**2 * self.scattering_rate(X,V)
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
        diffusion_constant = (constants['hbar']*k)**2 * self.scattering_rate(X,V)
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
        force = np.atleast_2d(np.zeros(X.shape))
        betaT = self.total_intensity(X)/atom['Isat']
        b = self.field(X)
        for beam in self.beams:
            force += constants['hbar']* np.outer(beam.scattering_rate(X,V, b, betaT), beam.wavevector)
        return force

    def plot(self):
        from MOTorNOT.plotting import subplots
        subplots(self.acceleration, numpoints=plot_params['numpoints'], label='a', units = r'm/s^2')

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
