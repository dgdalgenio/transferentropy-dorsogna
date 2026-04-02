import numpy as np

class CorrRandomWalkParticles:
    def __init__(self, n_particles, kappa, seed = None):

        self.n_particles = n_particles
        self.seed = seed
        self.kappa = kappa

        if seed is not None:
            np.random.seed(seed)
    
    def initiate(self, tmax, dt=1):
        self.tmax = tmax
        self.dt = dt
        self.time_step = int(round(self.tmax / self.dt)) + 1
        self.times = np.linspace(0.0, self.tmax, self.time_step)
        
        self.xpos = np.zeros((self.time_step, self.n_particles))
        self.ypos = np.zeros((self.time_step, self.n_particles))
        self.vx = np.zeros((self.time_step, self.n_particles))
        self.vy = np.zeros((self.time_step, self.n_particles))

        self.simulate()

    def simulate(self):

        self.xpos[0] = np.random.uniform(0, 1, size=self.n_particles)
        self.ypos[0] = np.random.uniform(0, 1, size=self.n_particles)
        theta0 = np.random.uniform(-np.pi, np.pi, size=self.n_particles)
        self.vx[0] = np.cos(theta0)
        self.vy[0] = np.sin(theta0)

        for t in range(1, self.time_step):
            
            self.turn_angle = np.random.vonmises(mu=0, kappa=self.kappa, size = self.n_particles)
            cos_turn_angle = np.cos(self.turn_angle)
            sin_turn_angle = np.sin(self.turn_angle)
            
            # Move using old velocity
            self.xpos[t, :] = self.xpos[t - 1, :] + self.vx[t - 1, :]
            self.ypos[t, :] = self.ypos[t - 1, :] + self.vy[t - 1, :]

            # Turn after moving 
            self.vx[t, :] = self.vx[t-1,:] * cos_turn_angle - self.vy[t-1,:] * sin_turn_angle
            self.vy[t, :] = self.vx[t-1,:] * sin_turn_angle + self.vy[t-1,:] * cos_turn_angle
