"""
Functions combinate, permute, makedir, normalize, accu_type_score imported from https://github.com/tanpei0513/vicsek_trajectory
"""
import os
import numpy as np
# import pandas as pd
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def combinate(cluster_label, num):
    return [list(p) for p in combinations(set(np.int_(cluster_label)), num)]




def permute(labels):
    # permute([1,2,3]) --> [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    return [list(p) for p in permutations(set(np.int_(labels)))]




def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)




def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))




def accu_type_score(set1, set2):
    all_comb = permute(set2)
    l_len = len(set1)
    best_accu = 0


    for comb in all_comb:  # throughout every possible combinations except comb[0]
        type_switch = np.zeros(shape=[l_len, ])
        for idx, val in enumerate(comb):  # change label from comb[0] to comb[i]
            type_switch[np.where(set2 == all_comb[0][idx])] = val
        accu = accuracy_score(set1, type_switch)
        if accu > best_accu:
            best_accu = accu
            best_type = type_switch
            if accu > 0.8:
                break
            else:
                pass
        else:
            pass


    return [best_type, best_accu]


"""
Functions below are created solely for the project
"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


def save_simulation(generator, path="simulation.npz"):
    """
    Export simulation to a path
    """
    np.savez_compressed(
        path,
        xpos=np.stack(generator.xpos),  # shape: (T, N)
        ypos=np.stack(generator.ypos),
        vx=np.stack(generator.vx),
        vy=np.stack(generator.vy),
        type_label=generator.type_label,
        tmax=generator.tmax,
        dt=generator.dt,
        type_num=generator.type_num,
        num_list=np.array(generator.num_list),
    )
    print(f"Saved simulation to {path}")


def load_simulation(path="single_1_sim.npz"):
    data = np.load(path, allow_pickle=True)
    """
    Load saved simulation
    """
    class Snapshot:
        pass
    snap = Snapshot()
    snap.xpos = [data["xpos"][i] for i in range(data["xpos"].shape[0])]
    snap.ypos = [data["ypos"][i] for i in range(data["ypos"].shape[0])]
    snap.vx = [data["vx"][i] for i in range(data["vx"].shape[0])]
    snap.vy = [data["vy"][i] for i in range(data["vy"].shape[0])]
    snap.type_label = data["type_label"]
    snap.tmax = float(data["tmax"])
    snap.dt = float(data["dt"])
    snap.type_num = int(data["type_num"])
    snap.num_list = data["num_list"].tolist()
    return snap


def animate_positions(generator, filename="simulation.mp4", title=None, fps=5,
                      xlim=None, ylim=None, show_velocity=False,
                      vel_scale=0.05, vel_subsample=1, vel_alpha=0.7,
                      ccw_color="red",cw_color="blue", neutral_color="gray"):
    """
    Animate and save particle positions, optionally with velocity arrows.


    Parameters
    ----------
    generator : object
        Should have xpos, ypos, vx, vy lists (indexed by time) and optionally type_label.
    filename : str
        Output path; .mp4 uses ffmpeg, .gif uses Pillow.
    title: str or None
        Base title shown in display; time is appended as "(t=...)".
    fps : int
        Frames per second of the output video.
        Slower fps is recommended for interpretability.
    xlim, ylim : tuple or None
        Plot limits for x and y. If None, auto-computed from data with a border.
    show_velocity : bool
        If True, overlay velocity vectors as arrows.
    vel_scale : float
        Scaling factor for velocity arrows (adjust for visibility).
    vel_subsample : int
        Show the velocity arrow of every Xth particle for every frame.
    vel_alpha : float in [0,1]
        Transparency of the velocity arrows (0 = invisible, 1 = opaque).
    """
    xpos = generator.xpos
    ypos = generator.ypos
    vx_list = getattr(generator, "vx", None)
    vy_list = getattr(generator, "vy", None)
    type_label = getattr(generator, "type_label", None)
   
    T = len(xpos)
    if T == 0:
        raise ValueError("No simulation data found (xpos is empty).")
    num_particles = len(xpos[0])


    # Compute global bounds if not provided
    all_x = np.concatenate(xpos)
    all_y = np.concatenate(ypos)
    if xlim is None:
        xmin, xmax = all_x.min(), all_x.max()
        xspan = xmax - xmin
        pad_x = xspan * 0.05 if xspan > 0 else 0.05
        xlim = (xmin - pad_x, xmax + pad_x)
    if ylim is None:
        ymin, ymax = all_y.min(), all_y.max()
        yspan = ymax - ymin
        pad_y = yspan * 0.05 if yspan > 0 else 0.05
        ylim = (ymin - pad_y, ymax + pad_y)


    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


    def rotation_color(x, y, vx, vy, ccw_color, cw_color, neutral_color):
        # center of mass
        cx = np.mean(x)
        cy = np.mean(y)
        # relative position to center
        rx = x - cx
        ry = y - cy
        # z-component of r x v: posiive = counterclockwise around center
        cross = rx * vy - ry * vx
        colors = np.full_like(cross, fill_value=neutral_color, dtype=object)
        colors[cross>0] = ccw_color
        colors[cross<0] = cw_color
        return colors
   
    # Initial frame data    
    x0 = xpos[0]
    y0 = ypos[0]
    vx0 = vx_list[0] if vx_list is not None else np.zeros(num_particles)
    vy0 = vy_list[0] if vy_list is not None else np.zeros(num_particles)
    if type_label is not None:
        scat = ax.scatter(x0, y0, s=20, c=type_label, cmap="tab10",
                          vmin=0, vmax=max(type_label)+1)
    else:
        scat = ax.scatter(x0, y0, s=20, c="black")
    colors0 = rotation_color(x0, y0, vx0,vy0, ccw_color, cw_color, neutral_color)
   
    quiv = None
    if show_velocity:
        if vx_list is None or vy_list is None:
            raise ValueError("Velocity data (vx, vy) missing but show_velocity=True.")
        idx = np.arange(0, num_particles, vel_subsample)
        vx0 = vx_list[0][idx]
        vy0 = vy_list[0][idx]


        quiv = ax.quiver(x0[idx], y0[idx], vx0, vy0,
                         angles='xy', scale_units='xy',
                         scale=1/vel_scale, width=0.003,
                         alpha=vel_alpha)


    def format_title(frame):
        return f"{title} (t={frame})" if title else f"t = {frame}"


    title_text = ax.text(0.5, 1.02, format_title(0), ha="center", va="bottom", transform=ax.transAxes)

    # Store line objects for each particle's trail
    lines = []
    trail_length = 50  # How many previous positions to show (None for full trail)
    for i in range(num_particles):
        line, = ax.plot([], [], '-', alpha=0.3, linewidth=1)
        lines.append(line)

    
    
    def update(frame):
        x = xpos[frame]
        y = ypos[frame]
        vx = vx_list[frame] if vx_list is not None else np.zeros(num_particles)
        vy = vy_list[frame] if vy_list is not None else np.zeros(num_particles)
       
        scat.set_offsets(np.stack((x, y), axis=1))
        colors = rotation_color(x, y, vx, vy, ccw_color, cw_color, neutral_color)
        # colors = plt.cm.tab10(np.arange(num_particles) % 10) 
        scat.set_color(colors)
        title_text.set_text(format_title(frame))

                # Update trails
        start_frame = max(0, frame - trail_length) if trail_length else 0
        for i in range(num_particles):
            trail_x = [xpos[t][i] for t in range(start_frame, frame + 1)]
            trail_y = [ypos[t][i] for t in range(start_frame, frame + 1)]
            lines[i].set_data(trail_x, trail_y)
            # Optional: color the trail to match particle color
            lines[i].set_color(colors[i])
            
        artists = [scat, title_text]
        if show_velocity:
            vx = vx_list[frame]
            vy = vy_list[frame]
            idx = np.arange(0, num_particles, vel_subsample)
            quiv.set_offsets(np.stack((x[idx], y[idx]), axis=1))
            quiv.set_UVC(vx[idx], vy[idx])
            quiv.set_alpha(vel_alpha)
            artists.append(quiv)


        return tuple(artists)


    anim = FuncAnimation(fig, update, frames=range(T),
                         interval=1000 / fps, blit=True)


    if filename.lower().endswith(".mp4"):
        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
    elif filename.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
    else:
        raise ValueError("Unsupported extension; use .mp4 or .gif")


    plt.close(fig)
    print(f"Saved simulation to {filename}")

    return fig

def set_plot_style() -> None:
    """Apply a consistent matplotlib style for all figures."""
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"],
        "font.size":        11,
        "axes.labelsize":   12,
        "axes.titlesize":   12,
        "axes.linewidth":   0.8,
        "xtick.top":        False,
        "ytick.right":      False,
        "grid.linewidth":   0.5,
        "grid.alpha":       0.4,
        "grid.linestyle":   "--",
        "figure.dpi":       300,
    })

proper_phenotype_names = {
    'singlemill': 'single mill',
    'doublemill': 'double mill',
    'doublering': 'double ring',
    'collswarm': 'collective swarm',
    'escapesymm': 'escape symmetric',
    'escapeunsymm': 'escape unsymmetric',
    'escapecoll': 'escape collective'
}