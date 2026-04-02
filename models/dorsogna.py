import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.integrate import ode
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import SpectralClustering

# from utils import accu_type_score


class Particle:
    type_num = 0
    label_list = []
    alpha_list = []
    beta_list = []
    cA_list = []
    cR_list = []
    lA_list = []
    lR_list = []
    num_list = []

    def __init__(self, label, alpha, beta, cA, cR, lA, lR, numbers):
        Particle.type_num += 1
        Particle.label_list.append(label)
        Particle.num_list.append(numbers)
        Particle.alpha_list.append(alpha)
        Particle.beta_list.append(beta)
        Particle.cA_list.append(cA)
        Particle.cR_list.append(cR)
        Particle.lA_list.append(lA)
        Particle.lR_list.append(lR)

    def reset(self):
        Particle.type_num = 0
        Particle.label_list = []
        Particle.num_list = []
        Particle.alpha_list = []
        Particle.beta_list = []
        Particle.cA_list = []
        Particle.cR_list = []
        Particle.lA_list = []
        Particle.lR_list = []


class DorsognaGenerator(Particle):

    def __init__(self, label, alpha, beta, cA, cR, lA, lR, numbers):
        super().__init__(label, alpha, beta, cA, cR, lA, lR, numbers)

        self.label_list = Particle.label_list
        self.type_num = Particle.type_num
        self.num_list = Particle.num_list
        self.alpha_list = Particle.alpha_list
        self.beta_list = Particle.beta_list
        self.cA_list = Particle.cA_list
        self.cR_list = Particle.cR_list
        self.lA_list = Particle.lA_list
        self.lR_list = Particle.lR_list

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
        # self.vx[0] = np.random.uniform(-1, 1, size=self.num_sum)
        # self.vy[0] = np.random.uniform(-1, 1, size=self.num_sum)
        theta0 = np.random.uniform(-np.pi, np.pi, size=self.num_sum)
        self.vx[0] = np.cos(theta0)
        self.vy[0] = np.sin(theta0)

        self.dorsogna()

    def dorsogna(self):

        def dorsogna_model(t, init, alpha, beta, cA, lA, cR, lR):
            # cA = self.cA_list[0]
            # lA = self.lA_list[0]
            # cR = self.cR_list[0]
            # lR = self.lR_list[0]
            # alpha = self.alpha_list[0]
            # beta = self.beta_list[0]

            meps = np.finfo(np.float64).eps

            l = len(init) // 4
            xpos0 = init[0:l].reshape(1, l)
            ypos0 = init[l:2 * l].reshape(1, l)
            vx = init[2 * l:3 * l]
            vy = init[3 * l:]

            # Compute model components
            xdiff = xpos0 - xpos0.T
            ydiff = ypos0 - ypos0.T
            dist = np.sqrt(xdiff ** 2 + ydiff ** 2)
            v_sq = vx ** 2 + vy ** 2

            u_prime = - cA / lA * np.exp(-dist / lA) + cR / lR * np.exp(-dist / lR)

            dvxdt = (alpha - beta * v_sq) * vx - np.sum(u_prime * xdiff / (dist + meps), axis=1)
            dvydt = (alpha - beta * v_sq) * vy - np.sum(u_prime * ydiff / (dist + meps), axis=1)

            return np.hstack((vx, vy, dvxdt, dvydt))

        cA = np.repeat(self.cA_list, self.num_list)
        lA = np.repeat(self.lA_list, self.num_list)
        cR = np.repeat(self.cR_list, self.num_list)
        lR = np.repeat(self.lR_list, self.num_list)
        alpha = np.repeat(self.alpha_list, self.num_list)
        beta = np.repeat(self.beta_list, self.num_list)

        init = np.hstack((self.xpos[0], self.ypos[0], self.vx[0], self.vy[0]))
        
        sol = ode(dorsogna_model).set_integrator('dopri5', atol=10 ** (-3))
        sol.set_initial_value(init, 0).set_f_params(alpha, beta, cA, lA, cR, lR)

#         while sol.successful() and sol.t < self.tmax - 1:
        for k in range(1, self.time_step):
            res = sol.integrate(self.times[k])
            if not sol.successful():
                raise RuntimeError(f"Integrator failed at t={self.times[k]}")
            self.xpos[k] = res[0:self.num_sum]
            self.ypos[k] = res[self.num_sum:2 * self.num_sum]
            self.vx[k]   = res[2 * self.num_sum:3 * self.num_sum]
            self.vy[k]   = res[3 * self.num_sum:]
            print(f"Dorsogna simulation: {k} / {self.time_step-1}", end="\r")
#             res = sol.integrate(sol.t + self.dt) 
#             idx = int(sol.t / self.dt) # changed. initially `idx = int(sol.t)`
#             self.xpos[idx] = res[0:self.num_sum]
#             self.ypos[idx] = res[self.num_sum:2 * self.num_sum]
#             self.vx[idx] = res[2 * self.num_sum:3 * self.num_sum]
#             self.vy[idx] = res[3 * self.num_sum:]
#             print(f"Dorsogna simulation: {idx + 1} / {int(self.tmax)}", end="\r")

    # def pca(self, time_series, pca_n=2):

    #     self.time_series = [int(i) for i in time_series]
    #     self.pca_comp = [None] * len(self.time_series)

    #     for i, t in enumerate(self.time_series):
    #         data = [np.arctan2(self.vy[int(j)], self.vx[int(j)]) for j in range(t)]
    #         # data = np.concatenate((np.sin(np.array(data).T), np.cos(np.array(data).T)), axis=1)
    #         # data = StandardScaler().fit_transform(np.array(data).T)
    #         data = np.array(data).T
    #         self.pca_comp[i] = PCA(n_components=pca_n, random_state=31415).fit_transform(data)
    #         print(f"pca: {t} / {self.time_series[-1]}", end="\r")

    # def cluster(self, cluster_type="spectral", pca_n=2):

    #     self.cluster_type = cluster_type
    #     self.cluster_list = [None] * len(self.time_series)
    #     self.accuracy_arry = [None] * len(self.time_series)

    #     for i, t in enumerate(self.time_series):
    #         components = self.pca_comp[i][:, 0:pca_n]
    #         if cluster_type == "spectral":
    #             cluster_label = SpectralClustering(self.type_num, affinity='nearest_neighbors',random_state=31415).fit(components).labels_
    #             [self.cluster_list[i], self.accuracy_arry[i]] = accu_type_score(self.type_label, cluster_label)
    #         else:
    #             print("Invalid cluster type!")
    #         print(f"cluster: {t} / {self.time_series[-1]}", end="\r")

    # def remove_item(self, item="xpos"):
    #     item_list = ["xpos", "ypos", "vx", "vy"]
    #     if item not in item_list:
    #         raise ValueError(f"Invalid item name. Only capable for item in {item_list}")
    #     try:
    #         if hasattr(self, str(f"{item}")):
    #             if item == "xpos":
    #                 del [self.xpos]
    #             if item == "ypos":
    #                 del [self.ypos]
    #             if item == "vx":
    #                 del [self.vx]
    #             if item == "vy":
    #                 del [self.vy]
    #     except NameError:
    #         pass