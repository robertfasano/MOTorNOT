import numpy as np
import attr
from scipy.constants import hbar, physical_constants
from scipy.optimize import root, root_scalar, bisect
from scipy.interpolate import interp1d
import plotly.graph_objs as go
mu_B = physical_constants['Bohr magneton'][0]
amu = physical_constants['atomic mass constant'][0]
from MOTorNOT.beams import *
from MOTorNOT.integration import solve
import matplotlib.pyplot as plt

@attr.s
class MOT:
    ''' This class calculates the scattering rate and optical forces arising
        from a set of beams and a magnetic field.
    '''
    beams = attr.ib(converter=list)
    field = attr.ib()
    atom = attr.ib(default={})

    def acceleration(self, X, V):
        return self.force(X,V)/(self.atom['mass'] * amu)

    def total_intensity(self, X):
        It = 0
        for beam in self.beams:
            It += beam.intensity(X)
        return It

    def beam_scattering_rate(self, beam, X, V):
        ''' The scattering rate of the beam at a given position and velocity.
            Args:
                X (ndarray): position array with shape (N, 3)
                V (ndarray): velocity array with shape (N, 3)
                b (ndarray): magnetic field evaluated at the position
                betaT (ndarray): total saturation fraction evaluated at X
        '''
        linewidth = 2*np.pi*self.atom['gamma']
        wavenumber = 2*np.pi/(self.atom['wavelength']*1e-9)
        wavevector = beam.direction * wavenumber
        Isat = self.atom['Isat'] * 10
        prefactor = linewidth/2 * beam.intensity(X)/Isat
        summand = 0
        b = self.field(X)
        eta = self.eta(b, beam.direction, beam.handedness)
        betaT = self.total_intensity(X)/Isat
        for mF in [-1, 0, 1]:
            amplitude = eta.T[mF+1]
            denominator = (1+betaT+4/linewidth**2*(beam.detuning-np.dot(wavevector, V.T)-mF*self.atom['gF']*mu_B*np.linalg.norm(b,axis=1)/hbar)**2)
            summand += amplitude / denominator
        rate = (prefactor.T*summand).T
        return rate



    def scattering_rate(self, X, V, i=None):
        rate = 0
        if i is not None:
            return self.beam_scattering_rate(self.beams[i], X, V)
        for beam in self.beams:
            rate += self.beam_scattering_rate(beam, X, V)
        return rate

    def force(self, X, V):
        X = np.atleast_2d(X)
        V = np.atleast_2d(V)
        force = np.atleast_2d(np.zeros(X.shape))
        Isat = self.atom['Isat'] * 10
        betaT = self.total_intensity(X)/Isat
        b = self.field(X)
        wavenumber = 2*np.pi/(self.atom['wavelength']*1e-9)
        for beam in self.beams:
            # force += hbar* np.outer(beam.scattering_rate(X,V, b, betaT), wavenumber * beam.direction)
            force += hbar* np.outer(self.beam_scattering_rate(beam, X, V), wavenumber * beam.direction)

        return force

    def plot(self, plane='xy', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50, quiver_scale=30, component='all'):
        from MOTorNOT.plotting import plot_2D, plane_indices
        fig = plot_2D(self.acceleration, plane=plane, limits=limits, numpoints=numpoints, quiver=True, quiver_scale=quiver_scale, component=component)

        field_center = self.field_center()
        trap_center = self.trap_center()
        i, j = plane_indices(plane)
        fig.add_trace(go.Scatter(x=[field_center[i]], y=[field_center[j]], marker={'symbol': 'x', 'color': 'white', 'size': 12}))
        fig.add_trace(go.Scatter(x=[trap_center[i]], y=[trap_center[j]], marker={'symbol': 'circle-open', 'color': 'white', 'size': 12}))

        fig.show()


    def phase_plot(self, axis='x', limits=[(-10e-3, 10e-3), (-10e-3, 10e-3)], numpoints=50):
        from MOTorNOT.plotting import plot_phase_space_force, plane_indices
        import plotly.graph_objs as go

        surf = plot_phase_space_force(self.acceleration, axis=axis, limits=limits, numpoints=numpoints)
        fig = go.Figure([surf])

        fig.show()

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
        with np.errstate(divide='ignore', invalid='ignore'):
            Bhat = np.divide(bT, bnorm)
            Bhat[:, bnorm==0] = 0

        xi = khat.dot(Bhat)
        return np.array([(1+s*xi)**2/4, (1-xi**2)/2, (1-s*xi)**2/4]).T

    def trap_center(self):
        ''' Numerically locate the potential minimum '''
        def acceleration(x):
            return self.acceleration(np.atleast_2d(x), V=np.atleast_2d([0, 0, 0]))[0]
        x0 = [0, 0, self.beams[0].z0 + self.radius / np.tan(self.alpha) / 2]
        return root(acceleration, x0=x0, tol=1e-4).x

    def field_center(self):
        ''' Numerically locate the field strength minimum '''
        def field_strength(x):
            return self.field(np.atleast_2d(x))[0]
        return root(field_strength, x0=[0, 0, 0], tol=1e-4).x

class SixBeam(MOT):
    def __init__(self, power, radius, detuning, handedness, field, theta=0, phi=0):
        from MOTorNOT import rotate
        beams = []
        directions = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]
        for d in directions:
            d = np.dot(rotate(0, theta), d)
            d = np.dot(rotate(2, phi), d)

            beam = UniformBeam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = handedness)
            beams.append(beam)

        directions = [[0, 0, 1], [0, 0, -1]]
        for d in directions:
            d = np.dot(rotate(0, theta), d)
            d = np.dot(rotate(2, phi), d)
            beam = UniformBeam(direction = np.array(d),
                        power = power,
                        radius = radius,
                        detuning = detuning,
                        handedness = -handedness)
            beams.append(beam)
        super().__init__(beams, field=field)

class GratingMOT(MOT):
    def __init__(self, atom, position, alpha, detuning, radius, power, handedness, R1, field, sectors=3, grating_radius=None, beam_type = 'uniform', R0=0, infinite=False):
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
            self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius, infinite=infinite))
        # for n in np.linspace(-1, -sectors, sectors):
            # self.beams.append(DiffractedBeam(n, alpha, R1*power/np.cos(alpha), radius, detuning, -handedness, position, beam_type=beam_type, sectors=sectors, grating_radius=grating_radius))

        if grating_radius is None:
            grating_radius = 2*radius
        beam_params = {'direction': np.array([0,0,-1]),
                       'power': power,
                        'radius': radius,
                        'detuning': detuning,
                        'handedness': handedness,
                        'cutoff': grating_radius}

        if beam_type == 'uniform':
            incident_beam = UniformBeam(
                direction=np.array([0, 0, -1]),
                power=power,
                radius=radius,
                detuning=detuning,
                handedness=handedness,
                cutoff=grating_radius,
                infinite=infinite
            )
            zeroth_order = UniformBeam(
                direction=np.array([0, 0, 1]),
                power=power*R0,
                radius=radius,
                detuning=detuning,
                handedness=-handedness,
                cutoff=grating_radius,
                infinite=infinite
            )
            self.beams.append(incident_beam)
            # self.beams.append(zeroth_order)

        elif beam_type == 'gaussian':
            incident_beam = GaussianBeam(
                direction=np.array([0, 0, -1]),
                power=power,
                radius=radius,
                detuning=detuning,
                handedness=handedness,
                cutoff=grating_radius
            )
            zeroth_order = GaussianBeam(
                direction=np.array([0, 0, 1]),
                power=power*R0,
                radius=radius,
                detuning=detuning,
                handedness=-handedness,
                cutoff=grating_radius
            )
            self.beams.append(incident_beam)
            # self.beams.append(zeroth_order)

        self.radius = radius
        self.alpha = alpha
        super().__init__(self.beams, self.field, atom=atom)

    def capture_velocity(self):
        ## check if a stable trap position even exists
        if not self.beams[0].z0 < self.trap_position()[0, 2] < -self.beams[0].z0:
            return 0

        ## backwards integration to determine capture velocity
        X0 = np.atleast_2d([0, 0, self.turning_point() - 1e-6])
        V0 = np.atleast_2d([0, 0, 0])
        y, X, V, t, events = solve(self.acceleration, X0, V0, -0.1)

        y_interp = interp1d(y[2, :], t)
        v_interp = interp1d(t, y[5, :])
        try:
            t_grating = y_interp(self.beams[0].z0)
            v_capture = v_interp(t_grating)
            return v_capture

        except ValueError:
            return 0
    
    def trap_position(self):
        ''' Compute the axial trap shift of the MOT '''
        Fz = lambda z: self.force(np.atleast_2d([0, 0, z]), np.atleast_2d([0, 0, 0]))[0, 2]
        # z0 = root_scalar(Fz, bracket=(-3e-3, 1e-3)).root
        z0 = root_scalar(Fz, x0=-1e-5, x1=1e-5).root

        return np.atleast_2d([0, 0, z0])

    def turning_point(self):
        ''' Returns the z coordinate of the anti-trapping region at z>0. '''
        z_trap = self.trap_position()[0, 2]
        z_edge = -self.beams[0].z0
        try:
            return bisect(lambda x: np.sign(self.force(np.atleast_2d([0, 0, x]), np.atleast_2d([0, 0, 0]))[0, 2]), z_trap+1e-6, z_edge-1e-6)
        except ValueError:
            return -self.beams[0].z0

    def damping_constant(self, R=None, direction='all'):
        if direction in ['x', 'y', 'z']:
            if R is None:
                R = self.trap_position()
            V = np.atleast_2d([0, 0, 0])
            i = {'x': 0, 'y': 1, 'z': 2}[direction]
            dV = [0, 0, 0]
            dV[i] = 1e-6
            dV = np.atleast_2d(dV)

            return (self.force(R, V-dV) - self.force(R, V+dV))[0, i] / (2*dV[0, i])
        
        elif direction == 'all':
            return [self.damping_constant(R, 'x'), self.damping_constant(R, 'y'), self.damping_constant(R,'z')]

    def spring_constant(self, R=None, direction='all'):
        if direction in ['x', 'y', 'z']:
            if R is None:
                R = self.trap_position()
            i = {'x': 0, 'y': 1, 'z': 2}[direction]
            dR = [0, 0, 0]
            dR[i] = 1e-4
            dR = np.atleast_2d(dR)
            V = np.atleast_2d([0, 0, 0])

            return (self.force(R-dR, V) - self.force(R+dR, V))[0, i] / (2*dR[0, i])
        
        elif direction == 'all':
            return [self.spring_constant(R=R, direction='x'), self.spring_constant(R=R, direction='y'), self.spring_constant(R=R, direction='z')]




class PyramidMOT(MOT):
    def __init__(self, atom, leg_length, separation, detuning, radius, power, handedness, field):
        ''' Creates a virtual laser beam. Params dict should contain the following fields:
                radius (float): radius of incident beam
                detuning (float): detuning of incident beam
                field (method): function returning the magnetic field at a position vector X
                power (float): power of the incident beam
                handedness (float): +/-1 for circular polarization
        '''
        self.field = field
        self.beams = []

        kvecs = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 1, 0])]
        for k in kvecs:
            self.beams.append(PyramidBeam(leg_length, separation, k, power, radius, detuning, -handedness))

        beam_params = {'direction': np.array([0,0,-1]),
                       'power': power,
                        'radius': radius,
                        'detuning': detuning,
                        'handedness': handedness}

        incident_beam = UniformBeam(
            direction=np.array([0, 0, -1]),
            power=power,
            radius=radius,
            detuning=detuning,
            handedness=handedness
        )
        reflected_beam = UniformBeam(
            direction=np.array([0, 0, 1]),
            power=power,
            radius=radius,
            detuning=detuning,
            handedness=handedness,
            cutoff=separation
        )

        self.beams.append(incident_beam)
        self.beams.append(reflected_beam)

        super().__init__(self.beams, self.field, atom=atom)

# class PyramidBeam(Beam):
#     def __init__(self, leg_length, separation, direction, power, radius, detuning, handedness):
#         super().__init__(direction, power, radius, detuning, handesness)
#         self.d = leg_length
#         self.D = separation

#     def intensity(self, X):
#         x = X.T[0]
#         y = X.T[1]
#         z = X.T[2]

#         beam_exists = True

#         ## trace back to mirror surface and check if the point originates from inside the incident beam
#         beam_exists &= (z+(self.d+self.D)/2)**2+y**2<self.radius**2

#         ## check that intersection point is actually on mirror surface
#         beam_exists &= -d/2 < z < d/2

#         return self.I * beam_exists