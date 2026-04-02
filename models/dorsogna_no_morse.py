import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.integrate import ode

class NoMorseParticle:
    type_num = 0
    label_list = []
    alpha_list = []
    beta_list = []
    num_list = []

    def __init__(self, label, alpha, beta, numbers):
        NoMorseParticle.type_num += 1
        NoMorseParticle.label_list.append(label)
        NoMorseParticle.num_list.append(numbers)
        NoMorseParticle.alpha_list.append(alpha)
        NoMorseParticle.beta_list.append(beta)

    def reset(self):
        NoMorseParticle.type_num = 0
        NoMorseParticle.label_list = []
        NoMorseParticle.num_list = []
        NoMorseParticle.alpha_list = []
        NoMorseParticle.beta_list = []

class DorsognaNoMorseGenerator(NoMorseParticle):

    def __init__(self, label, alpha, beta, numbers):
        super().__init__(label, alpha, beta, numbers)

        self.label_list = NoMorseParticle.label_list
        self.type_num = NoMorseParticle.type_num
        self.num_list = NoMorseParticle.num_list
        self.alpha_list = NoMorseParticle.alpha_list
        self.beta_list = NoMorseParticle.beta_list


    def initiate(self, tmax, dt=1):
        self.tmax = tmax
        self.dt = dt
        self.time_step = int(round(self.tmax / self.dt)) + 1 # changed: int(self.tmax / self.dt)
        self.times = np.linspace(0.0, self.tmax, self.time_step) # added
        
        self.type_label = np.repeat([*range(self.type_num)], self.num_list)
        self.num_sum = sum(self.num_list)
        self.xpos = [None] * self.time_step
        self.ypos = [None] * self.time_step
        self.vx = [None] * self.time_step
        self.vy = [None] * self.time_step

        self.simulate()

    def simulate(self):

        self.xpos[0] = np.random.uniform(0, 1, size=self.num_sum)
        self.ypos[0] = np.random.uniform(0, 1, size=self.num_sum)
        theta0 = np.random.uniform(-np.pi, np.pi, size=self.num_sum)
        self.vx[0] = np.cos(theta0)
        self.vy[0] = np.sin(theta0)

        self.dorsogna_no_morse()

    def dorsogna_no_morse(self):

        def dorsogna_no_morse_model(t, init, alpha, beta):

            l = len(init) // 4
 
            vx = init[2 * l:3 * l]
            vy = init[3 * l:]

            # Compute model components
 
            v_sq = vx ** 2 + vy ** 2

            dvxdt = (alpha - beta * v_sq) * vx 
            dvydt = (alpha - beta * v_sq) * vy 

            return np.hstack((vx, vy, dvxdt, dvydt))


        alpha = np.repeat(self.alpha_list, self.num_list)
        beta = np.repeat(self.beta_list, self.num_list)

        init = np.hstack((self.xpos[0], self.ypos[0], self.vx[0], self.vy[0]))
        
        sol = ode(dorsogna_no_morse_model).set_integrator('dopri5', atol=10 ** (-3))
        sol.set_initial_value(init, 0).set_f_params(alpha, beta)

        for k in range(1, self.time_step):
            res = sol.integrate(self.times[k])
            if not sol.successful():
                raise RuntimeError(f"Integrator failed at t={self.times[k]}")
            self.xpos[k] = res[0:self.num_sum]
            self.ypos[k] = res[self.num_sum:2 * self.num_sum]
            self.vx[k]   = res[2 * self.num_sum:3 * self.num_sum]
            self.vy[k]   = res[3 * self.num_sum:]
            print(f"Dorsogna w/o Morse simulation: {k} / {self.time_step-1}", end="\r")