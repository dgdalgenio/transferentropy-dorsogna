import numpy as np

class RandomWalkParticles:
    def __init__(self, n_particles, sigma=1.0, seed=None):
        self.n_particles = n_particles
        self.sigma = sigma # sigma is the noise strength — it controls how big the random steps are at each time step

        if seed is not None:
            np.random.seed(seed)

    def initiate(self, tmax, dt=1):
        self.tmax = tmax
        self.dt = dt
        self.time_step = int(round(self.tmax / self.dt)) + 1
        self.times = np.linspace(0.0, self.tmax, self.time_step)

        # State arrays: shape = (time, particles)
        self.xpos = np.zeros((self.time_step, self.n_particles))
        self.ypos = np.zeros((self.time_step, self.n_particles))
        self.vx = np.zeros((self.time_step, self.n_particles))
        self.vy = np.zeros((self.time_step, self.n_particles))

        self.simulate()

    def simulate(self):
        # Initial conditions
        self.xpos[0] = np.random.uniform(0, 1, size=self.n_particles)
        self.ypos[0] = np.random.uniform(0, 1, size=self.n_particles)
        theta0 = np.random.uniform(-np.pi, np.pi, size=self.n_particles)
        self.vx[0] = np.cos(theta0)
        self.vy[0] = np.sin(theta0)

        for t in range(1, self.time_step):
            # Random velocities (Gaussian noise)
            self.vx[t, :] = np.random.normal(
                loc=0.0, scale=self.sigma * np.sqrt(self.dt), size=self.n_particles
            )
            self.vy[t, :] = np.random.normal(
                loc=0.0, scale=self.sigma * np.sqrt(self.dt), size=self.n_particles # N(0, σ√dt)
            )

            # Position update
            self.xpos[t, :] = self.xpos[t - 1, :] + self.vx[t, :]
            self.ypos[t, :] = self.ypos[t - 1, :] + self.vy[t, :]
