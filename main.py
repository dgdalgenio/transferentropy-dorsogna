from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pandas as pd
import utils
from tqdm import tqdm
import os
import calculateTE

## Import models ##
import models.dorsogna as dorsogna
import models.dorsogna_no_morse as dorsogna_no_morse
import models.dorsogna_noisy as dorsogna_noisy
import models.random_walk as random_walk
import models.corr_random_walk as corr_random_walk

import warnings
warnings.filterwarnings("ignore")

"""
Transfer Entropy (TE) simulation classes.
 
Hierarchy:
    BaseTE          — shared TE computation, masking, logging, graphing
    ├── DorsognaTE  — D'Orsogna particle swarm model
    ├── RandomWalkTE  — random walk particle swarm model
    ├── CorrRandomWalkTE  — correlated random walk particle swarm model
    ├── DorsognaNoMorseTE  — D'orsogna (without morse function) particle swarm model
    └── DorsognaNoisyTE  — D'orsogna (without morse function and with noise) particle swarm model
"""

class BaseTE(ABC):
    """
    Abstract base class for Transfer Entropy simulations.
 
    Abstract interface:
        develop_model(...)     — build self.pos, self.vel, self.total_timesteps,
                                 self.particle_count, self.overall_TE_df, and
                                 any model-specific attributes used by the name
                                 helpers below.
        _model_label()         — short string that identifies this run
                                 (used in filenames and column names).
        _proper_model_label()  — string that identifies the simulation type
                                 (used in figures).
        _overall_csv_path()    — path to the per-run overall-TE CSV.
 
    Everything else (TE computation, masking, logging, graphing) lives here.
    """
 
    def __init__(self, outdir: str = ""):
        self.outdir = outdir
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        # set by develop_model in subclasses
        self.pos: np.ndarray | None = None          # (T, N, 2)
        self.vel: np.ndarray | None = None          # (T, N, 2)
        self.total_timesteps: int | None = None
        self.particle_count: int | None = None
        self.overall_TE_df: pd.DataFrame | None = None
        self.dt: float = 1
 
        # set by compute_modelTE
        self.TE_ver: str | None = None
        self.TE_embedding: int | None = None
        self.nearest_neighbors: int | None = None
        self.neighbor_radius: float | None = None
        self.permute_seed: int | None = None
        self.TE_log_df: pd.DataFrame | None = None
        self.mask_df: pd.DataFrame | None = None
        self.overall_avg_TE: pd.Series | None = None
 
    ## Abstract interface ## ──────────────────────────────────────────────────

    @abstractmethod
    def develop_model(self, *args, **kwargs):
        """Initialise the simulation and populate self.pos / self.vel."""
 
    @abstractmethod
    def _model_label(self) -> str:
        """Return a compact label used in file names and DataFrame columns."""
 
    @abstractmethod
    def _overall_csv_path(self) -> str:
        """Return the path to this run's overall-TE CSV log."""
 
    ## Helpers for naming convention ## ──────────────────────────────────────────────────
      
    def _condition_str(self) -> str:
        """Return the masking-condition suffix (empty string if none)."""
        if self.neighbor_radius:
            return f"fr{self.neighbor_radius}"
        if self.nearest_neighbors:
            return f"nn{self.nearest_neighbors-1}"
        return ""
 
    def _permute_suffix(self, permutation: bool) -> str:
        if permutation and self.permute_seed:
            return f"_permuted{self.permute_seed}"
        if permutation:
            return "_permuted"
        return ""
 
    def _te_base_name(self, condition: str = "", permutation: bool = False) -> str:
        """Return a consistent name stem for files and column names."""
        return (
            f"TE_{self._model_label()}"
            f"_{self.TE_ver}{condition}"
            f"_k{self.TE_embedding}"
            f"{self._permute_suffix(permutation)}"
        )
 
    ## Other helper functions ## ──────────────────────────────────────────────────
 
    @staticmethod
    def _central_difference(pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute velocity from positions using central differencing.
 
            v[0]    = (pos[1]  - pos[0])   / dt          (forward)
            v[t]    = (pos[t+1]- pos[t-1]) / (2*dt)      (central)
            v[-1]   = (pos[-1] - pos[-2])  / dt          (backward)
        """
        vel = np.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
        vel[0]    = (pos[1]  - pos[0])   / dt
        vel[-1]   = (pos[-1] - pos[-2])  / dt
        return vel

    def calculateTE_ver(self, TE_ver: str, i: int, j: int) -> np.ndarray:
        """Dispatch to the correct TE estimator."""
        dispatch = {
            "linvel": lambda: calculateTE.TE_KSG_linvel(
                vel=self.vel, i=i, j=j, k=self.TE_embedding),
            "angvel": lambda: calculateTE.TE_KSG_angvel(
                vel=self.vel, i=i, j=j, k=self.TE_embedding),
        }
        if TE_ver not in dispatch:
            raise ValueError(
                f"Unknown TE_ver '{TE_ver}'. Valid options: {list(dispatch)}"
            )
        return dispatch[TE_ver]()
 
    def saveTEgraph(self, condition: str = "") -> bool:
        """Save TE graph"""
        fig_path = f"{self.outdir}/TE_{self._model_label()}_{self.TE_ver}{condition}_k{self.TE_embedding}.png"

        utils.set_plot_style()

        plt.figure(figsize=(8, 4))

        plt.plot(self.overall_avg_TE, color='black')
        plt.xlabel("Time Step")
        plt.ylabel("Average Local Transfer Entropy")
        plt.title(f"TE Over Time for {self._proper_model_label()} ({self.TE_ver}, {condition}, $k=l=${self.TE_embedding})")
        plt.xlim(0, self.total_timesteps*self.dt)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.show()
        print(f"Figure saved: {fig_path}")
        return True
 
    def log_overall_avg_TE(
            self, permutation: bool = False, condition: str = ""
        ) -> pd.DataFrame:
        col_name = self._te_base_name(condition=condition, permutation=permutation)
        self.overall_TE_df[col_name] = self.overall_avg_TE
        csv_path = self._overall_csv_path()
        self.overall_TE_df.to_csv(csv_path)
        print(f"Overall TE log saved: {csv_path}")
        return self.overall_TE_df
 
    ## Permutation helpers ## ──────────────────────────────────────────────────
 
    def _permute_velocity(self) -> np.ndarray:
        """Return a time-shuffled copy of self.vel (per-particle permutation)."""
        permuted = np.empty_like(self.vel)
        for p in range(self.vel.shape[1]):
            perm = np.random.permutation(self.vel.shape[0])
            permuted[:, p, :] = self.vel[perm, p, :]
        return permuted
 
    ## Masking for aggregation schemes ## ──────────────────────────────────────────────────
 
    def _build_fixed_radius_mask(
            self, mask_df: pd.DataFrame
        ) -> pd.DataFrame:
        for ts in tqdm(range(len(self.pos) - 1), desc="Fixed-radius mask"):
            tree = cKDTree(self.pos[ts])
            for i in range(self.particle_count):
                neighbors = tree.query_ball_point(self.pos[ts][i], r=self.neighbor_radius)
                for j in neighbors:
                    if j != i:
                        mask_df.at[ts, f"{i}-{j}"] = 1
        return mask_df
 
    def _build_nearest_neighbor_mask(
            self, mask_df: pd.DataFrame
        ) -> pd.DataFrame:
        for ts in tqdm(range(len(self.pos) - 1), desc="Nearest-neighbor mask"):
            tree = cKDTree(self.pos[ts])
            for i in range(self.particle_count):
                _, indices = tree.query(self.pos[ts][i], k=self.nearest_neighbors)
                for j in indices:
                    if j != i:
                        mask_df.at[ts, f"{i}-{j}"] = 1
        return mask_df
 
    ## Main TE computation ──────────────────────────────────────────────────
 
    def compute_modelTE(
            self,
            TE_ver: str,
            TE_embedding: int = 1,
            nearest_neighbors: int | None = None,
            neighbor_radius: float | None = None,
            permute_seed: int | None = None,
            save_graph: bool = True
        ) -> bool:
        """
        Compute pairwise Transfer Entropy for all particles.
 
        Parameters
        ----------
        TE_ver           : one of 'linvel', 'angvel'
        TE_embedding     : history embedding length k=l
        nearest_neighbors: if set, mask TE to k-nearest neighbours only
        neighbor_radius  : if set, mask TE to particles within this radius
        permute_seed     : if set, shuffle velocity time-series before computing
                           (null-model / significance test)
        save_graph       : if True, display graph as output and save graph as file
        """
        if neighbor_radius and nearest_neighbors:
            raise ValueError("Specify at most one of neighbor_radius / nearest_neighbors.")
 
        self.TE_embedding     = TE_embedding
        self.TE_ver           = TE_ver
        self.permute_seed     = permute_seed
        self.nearest_neighbors = nearest_neighbors
        self.neighbor_radius   = neighbor_radius
        condition              = self._condition_str()
 
        if permute_seed:
            np.random.seed(permute_seed)
            os.makedirs(f"{self.outdir}_permute", exist_ok=True)
 
        # Build index arrays
        timesteps_arr = list(range(self.total_timesteps))
        pairwise_arr  = [
            f"{i}-{j}"
            for i in range(self.particle_count)
            for j in range(self.particle_count)
            if i != j
        ]
        TE_log_df = pd.DataFrame(index=timesteps_arr, columns=pairwise_arr)
        mask_df   = pd.DataFrame(index=timesteps_arr, columns=pairwise_arr)
 
        # If for permutation test, permute velocity
        original_vel = None
        if permute_seed:
            original_vel = self.vel.copy()
            self.vel = self._permute_velocity()
 
        # Compute pairwise TE 
        desc = f"Computing TE for each particle {'(permuted, seed {permute_seed})' if permute_seed else ''}"
        for i in tqdm(range(self.particle_count), desc=desc):
            for j in range(self.particle_count):
                if j != i:
                    te_vals = self.calculateTE_ver(TE_ver, i, j)
                    TE_log_df[f"{i}-{j}"] = np.ravel(te_vals)[1:]
 
        self.TE_log_df = TE_log_df
 
        # If permuted, restore original velocity
        if original_vel is not None:
            self.vel = original_vel
 
        # If fixed radius or nearest neighbors, implement masking
        if neighbor_radius:
            mask_df = self._build_fixed_radius_mask(mask_df)
        elif nearest_neighbors:
            mask_df = self._build_nearest_neighbor_mask(mask_df)
 
        if neighbor_radius or nearest_neighbors:
            self.TE_log_df = self.TE_log_df * mask_df.values
            self.mask_df   = mask_df
            mask_csv = f"{self.outdir}/mask_{self._model_label()}_{self.TE_ver}{condition}.csv" 
            self.mask_df.to_csv(mask_csv)
            print(f"Mask saved: {mask_csv}")
 
        # Save raw TE log
        out_subdir = f"{self.outdir}_permute" if permute_seed else self.outdir
        te_csv_filename = f"{out_subdir}/TElog_{self._model_label()}_{self.TE_ver}{condition}_k{self.TE_embedding}{self._permute_suffix(bool(permute_seed))}.csv"
        self.TE_log_df.to_csv(te_csv_filename)
        print(f"TE log saved as {te_csv_filename}")
 
        # Log overall (swarm) TE
        self.overall_avg_TE = self.TE_log_df.mean(axis=1)
        self.log_overall_avg_TE(permutation=bool(permute_seed), condition=condition)
 
        if permute_seed and save_graph:
            print(f'Disabling display and save of graph...')
            save_graph = False
        
        if save_graph:
            return self.saveTEgraph(condition=condition)
        else:
            return True

# ──────────────────────────────────────────────────────────────────────────────
# D'Orsogna subclass
# ──────────────────────────────────────────────────────────────────────────────
class DorsognaTE(BaseTE):
    """
    D'Orsogna model Transfer Entropy simulation.
 
    Usage
    -----
    sim = DorsognaTE(...)
    sim.develop_model(...)
    sim.compute_modelTE(...)
    """
 
    def __init__(self, outdir: str = ""):
        super().__init__(outdir)
        # model-specific attributes (populated by develop_model)
        self.C: float | None             = None
        self.l: float | None             = None
        self.phenotype_name: str | None  = None
 
    # ── Naming helpers ────────────────────────────────────────────────────────
 
    def _model_label(self) -> str:
        return f"{self.phenotype_name}_{self.C}_{self.l}"
    
    def _proper_model_label(self) -> str:
        return f"{utils.proper_phenotype_names[self.phenotype_name]} $C=${self.C} $l=${self.l}"
 
    def _overall_csv_path(self) -> str:
        return f"{self.outdir}/TElogoverall_{self._model_label()}.csv"
 
    # ── Model initialisation ──────────────────────────────────────────────────
 
    def develop_model(
        self,
        C: float,
        l: float,
        phenotype_name: str,
        particle_count: int,
        t_max: float,
        seed: int = 42,
        fps: int = 2,
        show_velocity: bool = True,
        vel_scale: int = 10,
        dt: float = 1,
        animate: bool = True
    ):
        self.C              = C
        self.l              = l
        self.phenotype_name = phenotype_name
        self.particle_count = particle_count
        self.t_max          = t_max
        self.seed           = seed
        self.dt             = dt
        self.total_timesteps = int(round(t_max / dt))
 
        np.random.seed(seed)
 
        # Fixed D'Orsogna constants
        alpha, beta, cA, lA = 1.5, 0.5, 1.0, 1.0
 
        # Reset any lingering particle state
        dorsogna.Particle(label="tmp", alpha=0, beta=0, cA=0, cR=0, lA=1, lR=1, numbers=0).reset()
 
        # Build and run simulation
        dor_sim = dorsogna.DorsognaGenerator(
            label=phenotype_name,
            alpha=alpha, beta=beta,
            cA=cA, lA=lA,
            cR=C, lR=l,
            numbers=particle_count,
        )
        dor_sim.initiate(tmax=t_max, dt=dt)
        self.dor_sim = dor_sim
 
        # Save animation
        if animate:
            os.makedirs(self.outdir, exist_ok=True)
            self.frames = utils.animate_positions(
                dor_sim,
                filename=f"{self.outdir}/sim_{self._model_label()}.gif",
                title=f"{phenotype_name} ({C}, {l})",
                fps=fps,
                show_velocity=show_velocity,
                vel_scale=vel_scale,
                vel_subsample=1,
                vel_alpha=0.5,
            )
 
        # Positions: (T, N, 2)
        pos = np.stack(
            [np.array(dor_sim.xpos), np.array(dor_sim.ypos)], axis=2
        )
        self.pos = pos
        self.vel = self._central_difference(pos, dt)
 
        # Load or create overall-TE log
        csv_path = self._overall_csv_path()
        self.overall_TE_df = (
            pd.read_csv(csv_path, index_col=0)
            if os.path.exists(csv_path)
            else pd.DataFrame()
        )

# ──────────────────────────────────────────────────────────────────────────────
# Random Walk subclass
# ──────────────────────────────────────────────────────────────────────────────
class RandomWalkTE(BaseTE):
    """
    Gaussian random walk model Transfer Entropy simulation.
 
    Usage
    -----
    sim = RandomWalk(...)
    sim.develop_model(...)
    sim.compute_modelTE(...)
    """
 
    def __init__(self, outdir: str = ""):
        super().__init__(outdir)
        # model-specific attributes (populated by develop_model)
        self.sigma: float | None = None

    # ── Naming helpers ────────────────────────────────────────────────────────
 
    def _model_label(self) -> str:
        return f"randomwalk_s{self.sigma}_seed{self.seed}"
    
    def _proper_model_label(self) -> str:
        return rf"random walk $\sigma=${self.sigma} seed:{self.seed}"
 
    def _overall_csv_path(self) -> str:
        return f"{self.outdir}/TElogoverall_{self._model_label()}.csv"
 
    # ── Model initialisation ──────────────────────────────────────────────────
 
    def develop_model(
        self,
        particle_count: int,
        sigma: float,
        t_max: float,
        seed: int = 42,
        fps: int = 2,
        show_velocity: bool = True,
        vel_scale: int = 10,
        dt: float = 1,
        animate: bool = False,
        trail_length: int = None
    ):
        self.particle_count  = particle_count
        self.sigma           = sigma
        self.t_max           = t_max
        self.seed            = seed
        self.dt              = dt
        self.total_timesteps = int(round(t_max / dt))
 
        np.random.seed(seed)
 
        # Build and run simulation
        ranwalk_sim = random_walk.RandomWalkParticles(
            n_particles=particle_count,
            sigma=sigma,
            seed=seed,
        )
        ranwalk_sim.initiate(tmax=t_max, dt=dt)
        self.ranwalk_sim = ranwalk_sim
 
        if animate:
            os.makedirs(self.outdir, exist_ok=True)
            self.frames = utils.animate_positions(
                ranwalk_sim,
                filename=f"{self.outdir}/sim_{self._model_label()}.gif",
                title=f"Random Walk (sigma: {sigma}, seed: {seed})",
                fps=fps,
                show_velocity=show_velocity,
                vel_scale=vel_scale,
                vel_subsample=1,
                vel_alpha=0.5,
                trail_length=trail_length
            )
 
        # Positions: (T, N, 2)
        pos = np.stack(
            [np.array(ranwalk_sim.xpos), np.array(ranwalk_sim.ypos)], axis=2
        )
        self.pos = pos
        self.vel = self._central_difference(pos, dt)
 
        # Load or create overall-TE log
        csv_path = self._overall_csv_path()
        self.overall_TE_df = (
            pd.read_csv(csv_path, index_col=0)
            if os.path.exists(csv_path)
            else pd.DataFrame()
        )

# ──────────────────────────────────────────────────────────────────────────────
# Correlated Random Walk subclass
# ──────────────────────────────────────────────────────────────────────────────
class CorrRandomWalkTE(BaseTE):
    """
    Correlated Random Walk Transfer Entropy simulation.

    Heading direction is drawn from a von Mises distribution with
    concentration parameter kappa:
        kappa = 0 : uniform / uncorrelated,
        kappa → ∞ : straight-line motion
 
    Usage
    -----
    sim = CorrRandomWalk(...)
    sim.develop_model(...)
    sim.compute_modelTE(...)
    """
 
    def __init__(self, outdir: str = ""):
        super().__init__(outdir)
        # model-specific attributes (populated by develop_model)
        self.kappa: float | None = None

    # ── Naming helpers ────────────────────────────────────────────────────────
 
    def _model_label(self) -> str:
        return f"corrrandomwalk_kappa{self.kappa}_seed{self.seed}"
    
    def _proper_model_label(self) -> str:
        return rf"correlated random walk $\kappa=${self.kappa} seed:{self.seed}"
 
    def _overall_csv_path(self) -> str:
        return f"{self.outdir}/TElogoverall_{self._model_label()}.csv"
 
    # ── Model initialisation ──────────────────────────────────────────────────
 
    def develop_model(
        self,
        particle_count: int,
        kappa: float,
        t_max: float,
        seed: int = 42,
        fps: int = 2,
        show_velocity: bool = True,
        vel_scale: int = 10,
        dt: float = 1,
        animate: bool = False,
        trail_length: int = None
    ):
        self.particle_count  = particle_count
        self.kappa           = kappa
        self.t_max           = t_max
        self.seed            = seed
        self.dt              = dt
        self.total_timesteps = int(round(t_max / dt))
 
        np.random.seed(seed)
 
        # Build and run simulation
        corr_ranwalk_sim = corr_random_walk.CorrRandomWalkParticles(
            n_particles=particle_count,
            kappa=kappa,
            seed=seed,
        )
        corr_ranwalk_sim.initiate(tmax=t_max, dt=dt)
        self.corr_ranwalk_sim = corr_ranwalk_sim

        if animate:
            os.makedirs(self.outdir, exist_ok=True)
            self.frames = utils.animate_positions(
                corr_ranwalk_sim,
                filename=f"{self.outdir}/sim_{self._model_label()}.gif",
                title=f"Corr Random Walk (kappa: {kappa}, seed: {seed})",
                fps=fps, 
                show_velocity=show_velocity,
                vel_scale=vel_scale, 
                vel_subsample=1, 
                vel_alpha=0.5,
                trail_length=trail_length
            )

         # Positions: (T, N, 2)
        pos = np.stack(
            [np.array(corr_ranwalk_sim.xpos), np.array(corr_ranwalk_sim.ypos)], axis=2
        )
        self.pos = pos
        self.vel = self._central_difference(pos, dt)
 
        # Load or create overall-TE log
        csv_path = self._overall_csv_path()
        self.overall_TE_df = (
            pd.read_csv(csv_path, index_col=0)
            if os.path.exists(csv_path)
            else pd.DataFrame()
        )

# ──────────────────────────────────────────────────────────────────────────────
# D'Orsogna No Morse subclass
# ──────────────────────────────────────────────────────────────────────────────
class DorsognaNoMorseTE(BaseTE):
    """
    D'Orsogna variant without a Morse potential (no attraction/repulsion term).
 
    Only the self-propulsion / friction terms (alpha, beta) are active,
    so particles move under pure velocity dynamics.
 
    Usage
    -----
    sim = DorsognaNoMorseTE(...)
    sim.develop_model(...)
    sim.compute_modelTE(...)
    """
 
    def __init__(self, outdir: str = ""):
        super().__init__(outdir)

    # ── Naming helpers ────────────────────────────────────────────────────────
 
    def _model_label(self) -> str:
        return f"dorsognanomorse_seed{self.seed}"
    
    def _proper_model_label(self) -> str:
        return rf"dorsogna (no morse)"
 
    def _overall_csv_path(self) -> str:
        return f"{self.outdir}/TElogoverall_{self._model_label()}.csv"
 
    # ── Model initialisation ──────────────────────────────────────────────────
 
    def develop_model(
        self,
        phenotype_name: str,
        particle_count: int,
        t_max: float,
        seed: int = 42,
        fps: int = 2,
        show_velocity: bool = True,
        vel_scale: int = 10,
        dt: float = 1,
        animate: bool = False,
        trail_length: int = None
    ):
        self.phenotype_name  = phenotype_name
        self.particle_count  = particle_count
        self.t_max           = t_max
        self.seed            = seed
        self.dt              = dt
        self.total_timesteps = int(round(t_max / dt))
 
        np.random.seed(seed)
 
        # Fixed D'Orsogna constants (no Morse terms)
        alpha, beta = 1.5, 0.5
 
        # Reset any lingering particle state
        dorsogna_no_morse.NoMorseParticle(label="tmp", alpha=0, beta=0, numbers=0).reset()
 
        # Build and run simulation
        dor_sim = dorsogna_no_morse.DorsognaNoMorseGenerator(
            label=phenotype_name,
            alpha=alpha,
            beta=beta,
            numbers=particle_count,
        )
        dor_sim.initiate(tmax=t_max, dt=dt)
        self.dor_sim = dor_sim
    
        if animate:
            os.makedirs(self.outdir, exist_ok=True)
            self.frames = utils.animate_positions(
                dor_sim,
                filename=f"{self.outdir}/sim_{self._model_label()}.gif",
                title="D'Orsogna Variant - No Morse",
                fps=fps, 
                show_velocity=show_velocity,
                vel_scale=vel_scale, 
                vel_subsample=1, 
                vel_alpha=0.5,
                trail_length=trail_length
            )
 
        # Positions: (T, N, 2)
        pos = np.stack(
            [np.array(dor_sim.xpos), np.array(dor_sim.ypos)], axis=2
        )
        self.pos = pos
        self.vel = self._central_difference(pos, dt)
 
        # Load or create overall-TE log
        csv_path = self._overall_csv_path()
        self.overall_TE_df = (
            pd.read_csv(csv_path, index_col=0)
            if os.path.exists(csv_path)
            else pd.DataFrame()
        )


# ──────────────────────────────────────────────────────────────────────────────
# D'Orsogna Noisy subclass
# ──────────────────────────────────────────────────────────────────────────────
class DorsognaNoisyTE(BaseTE):
    """
    D'Orsogna variant with additive diffusion noise (no Morse potential).
 
    Usage
    -----
    sim = DorsognaNoisyTE(...)
    sim.develop_model(...)
    sim.compute_modelTE(...)
    """
 
    def __init__(self, outdir: str = ""):
        super().__init__(outdir)

    # ── Naming helpers ────────────────────────────────────────────────────────
 
    def _model_label(self) -> str:
        return f"dorsognanoisy_diff{self.diff_coef}_seed{self.seed}"
    
    def _proper_model_label(self) -> str:
        return rf"dorsogna (noisy)"
 
    def _overall_csv_path(self) -> str:
        return f"{self.outdir}/TElogoverall_{self._model_label()}.csv"
 
    # ── Model initialisation ──────────────────────────────────────────────────
 
    def develop_model(
        self,
        phenotype_name: str,
        particle_count: int,
        t_max: float,
        diff_coef: float = 1.0,
        seed: int = 42,
        fps: int = 2,
        show_velocity: bool = True,
        vel_scale: int = 10,
        dt: float = 1,
        animate: bool = False,
        trail_length: int = None
    ):
        self.phenotype_name  = phenotype_name
        self.particle_count  = particle_count
        self.t_max           = t_max
        self.diff_coef       = diff_coef
        self.seed            = seed
        self.dt              = dt
        self.total_timesteps = int(round(t_max / dt))
 
        np.random.seed(seed)
 
        # Fixed D'Orsogna constants
        alpha, beta = 1.5, 0.5
 
        # Reset any lingering particle state
        dorsogna_noisy.DorsognaNoisyParticle(
            label="tmp", alpha=0, beta=0, diff_coef=diff_coef, numbers=0
        ).reset()
 
        # Build and run simulation
        dor_sim = dorsogna_noisy.DorsognaNoisyGenerator(
            label=phenotype_name,
            alpha=alpha,
            beta=beta,
            diff_coef=diff_coef,
            numbers=particle_count,
        )
        dor_sim.initiate(tmax=t_max, dt=dt)
        self.dor_sim = dor_sim
 
        if animate:
            os.makedirs(self.outdir, exist_ok=True)
            self.frames = utils.animate_positions(
                dor_sim,
                filename=f"{self.outdir}/sim_{self._model_label()}.gif",
                title=f"D'Orsogna Noisy (diff={diff_coef})",
                fps=fps, 
                show_velocity=show_velocity,
                vel_scale=vel_scale, 
                vel_subsample=1, 
                vel_alpha=0.5,
                trail_length=trail_length
            )
 
        # Positions: (T, N, 2)
        pos = np.stack(
            [np.array(dor_sim.xpos), np.array(dor_sim.ypos)], axis=2
        )
        self.pos = pos
        self.vel = self._central_difference(pos, dt)
 
        # Load or create overall-TE log
        csv_path = self._overall_csv_path()
        self.overall_TE_df = (
            pd.read_csv(csv_path, index_col=0)
            if os.path.exists(csv_path)
            else pd.DataFrame()
        )