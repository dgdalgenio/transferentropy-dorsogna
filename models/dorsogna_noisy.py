import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.integrate import ode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

from utils import accu_type_score


class DorsognaNoisyParticle:
    type_num = 0
    label_list = []
    alpha_list = []
    beta_list = []
    diff_coef_list = []
    num_list = []

    def __init__(self, label, alpha, beta, diff_coef, numbers):
        DorsognaNoisyParticle.type_num += 1
        DorsognaNoisyParticle.label_list.append(label)
        DorsognaNoisyParticle.num_list.append(numbers)
        DorsognaNoisyParticle.alpha_list.append(alpha)
        DorsognaNoisyParticle.beta_list.append(beta)
        DorsognaNoisyParticle.diff_coef_list.append(diff_coef)

    def reset(self):
        DorsognaNoisyParticle.type_num = 0
        DorsognaNoisyParticle.label_list = []
        DorsognaNoisyParticle.num_list = []
        DorsognaNoisyParticle.alpha_list = []
        DorsognaNoisyParticle.beta_list = []
        DorsognaNoisyParticle.diff_coef_list = []


class DorsognaNoisyGenerator(DorsognaNoisyParticle):

    def __init__(self, label, alpha, beta, diff_coef, numbers):
        super().__init__(label, alpha, beta, diff_coef, numbers)

        self.label_list = DorsognaNoisyParticle.label_list
        self.type_num = DorsognaNoisyParticle.type_num
        self.num_list = DorsognaNoisyParticle.num_list
        self.alpha_list = DorsognaNoisyParticle.alpha_list
        self.beta_list = DorsognaNoisyParticle.beta_list
        self.diff_coef_list = DorsognaNoisyParticle.diff_coef_list


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
        
        alpha = np.repeat(self.alpha_list, self.num_list)
        beta = np.repeat(self.beta_list, self.num_list)
        diff_coef = np.repeat(self.diff_coef_list, self.num_list)
        
        for t in range(1, self.time_step):
            vx = self.vx[t-1]  
            vy = self.vy[t-1] 
            v_sq = vx ** 2 + vy ** 2
            
            noise_x = np.random.normal(0, 1, size=self.num_sum)
            noise_y = np.random.normal(0, 1, size=self.num_sum)
            
            dvx = (alpha - beta * v_sq) * vx
            dvy = (alpha - beta * v_sq) * vy
            
            self.vx[t] = vx + dvx * self.dt + np.sqrt(2 * diff_coef * self.dt) * noise_x  
            self.vy[t] = vy + dvy * self.dt + np.sqrt(2 * diff_coef * self.dt) * noise_y  
            
            self.xpos[t] = self.xpos[t-1] + self.vx[t-1] * self.dt 
            self.ypos[t] = self.ypos[t-1] + self.vy[t-1] * self.dt  