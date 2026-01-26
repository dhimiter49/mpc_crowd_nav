from matplotlib import animation
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

# =====================================================
# Obstacles
# =====================================================
class Obstacle:
    """
    Obstacles are simplified as oblique cylinders:
      (x - x0 - t*vx)^2 + (y - y0 - t*vy)^2 = r^2
    """
    def __init__(self, x0, y0, t0, t1, vx, vy, radius):
        self.x0 = x0
        self.y0 = y0
        self.t0 = t0
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.t1 = t1 if t1 is not None else float('inf')

    def speed(self):
        return (self.vx, self.vy, 1)

    def initial_position(self):
        return (self.x0, self.y0, self.t0)

    def position_at_time(self, t):
       dt = t - self.t0
       x = self.x0 + dt * self.vx
       y = self.y0 + dt * self.vy
       return x, y, t

# =====================================================
# JIT-compiled helpers (standalone functions)
# =====================================================
@njit
def _feasible_nearest_jit(x_new, y_new, t_new, tree, n_nodes, v_max):
    """
    JIT-compiled core logic of feasible_nearest.
    tree shape: (4, max_iter)
    We'll only look at columns up to n_nodes.
    """
    if n_nodes == 0:
        return -1  # means None

    # Extract arrays from tree
    x_tree = tree[0, :n_nodes]
    y_tree = tree[1, :n_nodes]
    t_tree = tree[2, :n_nodes]

    dt = t_new - t_tree
    mask = (dt > 1e-9)
    if not np.any(mask):
        return -1

    dx = x_new - x_tree
    dy = y_new - y_tree
    dist_xy_sq = dx*dx + dy*dy

    v2 = v_max * v_max
    # Feasible if dist_xy_sq <= (v_max * dt)^2
    feasible_mask = mask & (dist_xy_sq <= (v2 * dt * dt))
    if not np.any(feasible_mask):
        return -1

    # Among feasible, pick the node with the smallest 3D distance
    dz = t_new - t_tree
    dist_3d_sq = dist_xy_sq + dz*dz

    best_idx = -1
    best_val = 1e20
    for i in range(n_nodes):
        if feasible_mask[i]:
            if dist_3d_sq[i] < best_val:
                best_val = dist_3d_sq[i]
                best_idx = i

    return best_idx

@njit(cache=True, fastmath=True)
def liang_barsky_numba(x1, y1, x2, y2, hx, hy):
    """
    Teste l'intersection segment-AABB 2-D (Liang–Barsky).

    Paramètres
    ----------
    x1, y1, x2, y2 : float
        Extrémités du segment.
    hx, hy : float
        Demi-longueur et demi-largeur du rectangle centré en (0,0).

    Retour
    ------
    bool
        True  si le segment touche / traverse le rectangle élargi,
        False sinon.
    """
    dx = x2 - x1
    dy = y2 - y1

    p = (-dx, dx, -dy, dy)
    q = ( x1 + hx, hx - x1,  y1 + hy, hy - y1)

    u0 = 0.0
    u1 = 1.0

    for i in range(4):
        pi = p[i]
        qi = q[i]

        if pi == 0.0:          # segment // au bord
            if qi < 0.0:       # …et totalement à l'extérieur
                return False
        else:
            t = qi / pi
            if pi < 0.0:       # entrée du segment
                if t > u1:
                    return False
                if t > u0:
                    u0 = t
            else:              # sortie du segment
                if t < u0:
                    return False
                if t < u1:
                    u1 = t
    return True                # u0 ≤ u1  →  overlap non vide

@njit
def _is_colliding_jit(x1, y1, t1, x2, y2, t2,
                      x0_obs, y0_obs, t0_obs, t1_obs,
                      vx_obs, vy_obs, r_obs,
                      xc_tab, yc_tab, cos_tab, sin_tab, hx_tab, hy_tab):
    """
    JIT-compiled core logic of is_colliding.
    All obstacle arrays have length n_obstacles.
    """
    ### CROWD ###
    # 1) Final point collision
    # positions of obstacles at t2
    n_obs = x0_obs.size
    for i in range(n_obs):
        # 0) skip si en dehors de [t0, t1]
        if (t2 < t0_obs[i] and t1 < t0_obs[i]) or (t1 > t1_obs[i] and t2 > t1_obs[i]):
            continue
        dx2 = x2 - (x0_obs[i] + (t2 - t0_obs[i]) * vx_obs[i])
        dy2 = y2 - (y0_obs[i] + (t2 - t0_obs[i]) * vy_obs[i])
        dist2 = dx2*dx2 + dy2*dy2
        if dist2 < 4*r_obs[i]*r_obs[i]:
            return True

    # 2) Segment collision
    dt_seg = t2 - t1
    if abs(dt_seg) >= 1e-9:     # <-- garde anti division par zéro
        for i in range(n_obs):
            if (t2 < t0_obs[i] and t1 < t0_obs[i]) or (t1 > t1_obs[i] and t2 > t1_obs[i]):
                continue

            Ax = (vx_obs[i] - (x2 - x1)/(t2 - t1))
            Ay = (vy_obs[i] - (y2 - y1)/(t2 - t1))
            Bx = (x0_obs[i] - x1 + (x2 - x1)/(t2 - t1) * (t1 - t0_obs[i]))
            By = (y0_obs[i] - y1 + (y2 - y1)/(t2 - t1) * (t1 - t0_obs[i]))

            t_num = Ax*Bx + Ay*By
            t_den = Ax*Ax + Ay*Ay
            if t_den < 1e-12:       # quasi-parallèle
                continue
            # t_star = AxBx + AyBy / (Ax^2 + Ay^2)
            t_star = -t_num / t_den
            # clamp t_star to [t1, t2]
            if t_star < t1:
                t_star = t1
            elif t_star > t2:
                t_star = t2


            dxs = ((x0_obs[i] + (t_star - t0_obs[i]) * vx_obs[i]) - (x1 + (t_star - t1)/(t2 - t1) * (x2 - x1))) ** 2
            dys = ((y0_obs[i] + (t_star - t0_obs[i]) * vy_obs[i]) - (y1 + (t_star - t1)/(t2 - t1) * (y2 - y1))) ** 2
            distance = dxs + dys
            if distance < 4*r_obs[i]*r_obs[i]:
                return True

    ### TABLES ###
    n_tab = xc_tab.size
    # 1) Point collision
    for i in range(n_tab):
        # passage repère local
        dx =  cos_tab[i] * (x2 - xc_tab[i]) + sin_tab[i] * (y2 - yc_tab[i])
        dy = -sin_tab[i] * (x2 - xc_tab[i]) + cos_tab[i] * (y2 - yc_tab[i])
        if (-hx_tab[i] <= dx <= hx_tab[i]) and (-hy_tab[i] <= dy <= hy_tab[i]):
            return True

    # 2) Segment collision
    for i in range(n_tab):
        # coordonnées locales des deux extrémités
        x1l =  cos_tab[i]*(x1 - xc_tab[i]) + sin_tab[i]*(y1 - yc_tab[i])
        y1l = -sin_tab[i]*(x1 - xc_tab[i]) + cos_tab[i]*(y1 - yc_tab[i])
        x2l =  cos_tab[i]*(x2 - xc_tab[i]) + sin_tab[i]*(y2 - yc_tab[i])
        y2l = -sin_tab[i]*(x2 - xc_tab[i]) + cos_tab[i]*(y2 - yc_tab[i])

        if liang_barsky_numba(x1l, y1l, x2l, y2l, hx_tab[i], hy_tab[i]):
            return True

    return False

# =====================================================
# RRT Spatio-temporel (optimized version)
# =====================================================
class RRTSpatioTemporal:
    def __init__(
        self,
        start,          # tuple (x0, y0, t0)
        goal,           # tuple (xg, yg), forced t=0
        obstacles:list[Obstacle],
        rectangles:list[tuple[float, float, float, float, float]],  # (xc, yc, theta, L, W)
        robot_radius,
        t_range,        # (t_min, t_max)
        goal_bias,
        v_max=1.0,
        dt=0.1,
        max_iter=2000,
        goal_tolerance_xy=0.5,
        min_spatial_step=0.0,
        verbose=False
    ):
        self.start = (start[0], start[1], start[2])
        self.goal  = (goal[0],  goal[1],  0.0)

        self.t_range   = t_range
        self.v_max     = v_max
        self.dt = dt
        self.goal_tolerance_xy = goal_tolerance_xy
        self.max_iter  = max_iter
        self.goal_bias = goal_bias
        self.radius = obstacles[0].radius
        self.min_spatial_step = min_spatial_step

        # Store obstacles in separate arrays (no repeated .T or slicing).
        # Each array has length = n_obstacles
        self.x0_obs = np.array([obs.x0 for obs in obstacles], dtype=float)
        self.y0_obs = np.array([obs.y0 for obs in obstacles], dtype=float)
        self.t0_obs = np.array([obs.t0 for obs in obstacles], dtype=float)
        self.t1_obs = np.array([obs.t1 for obs in obstacles], dtype=float)
        self.vx_obs = np.array([obs.vx for obs in obstacles], dtype=float)
        self.vy_obs = np.array([obs.vy for obs in obstacles], dtype=float)
        self.r_obs  = np.array([obs.radius for obs in obstacles], dtype=float)
        # --------- tables and walls ---------
        self.xc_tab = np.array([c for (c,_,_,_,_) in rectangles], dtype=float)
        self.yc_tab = np.array([d for (_,d,_,_,_) in rectangles], dtype=float)
        th          = np.array([th for (_,_,th,_,_) in rectangles])
        self.cos_tab = np.cos(th)
        self.sin_tab = np.sin(th)
        L2 = 0.5*np.array([L for (_,_,_,L,_) in rectangles])
        W2 = 0.5*np.array([W for (_,_,_,_,W) in rectangles])
        self.hx_tab = L2 + robot_radius
        self.hy_tab = W2 + robot_radius

        self.verbose = verbose

    # -------------------------------------------------------------------
    # Distances
    # -------------------------------------------------------------------
    @staticmethod
    def distance_3d(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    @staticmethod
    def distance_xy(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx*dx + dy*dy)

    # -------------------------------------------------------------------
    # Feasible nearest (vectorized + streamlined)
    # -------------------------------------------------------------------
    def feasible_nearest(self, new_node):
        """
        Returns index of best parent in self.tree for new_node,
        subject to t[new_node] >= t[node] and speed <= v_max.
        """
        x_new, y_new, t_new = new_node
        idx = _feasible_nearest_jit(
            x_new, y_new, t_new, self.tree, self.n_nodes, self.v_max
        )
        # We return None if idx==-1
        return None if (idx < 0) else idx

    # -------------------------------------------------------------------
    # Steer
    # -------------------------------------------------------------------
    def steer(self, from_node, to_node):
        """
        Move from from_node toward to_node by self.dt in 3D (x,y,t).
        """
        d = self.distance_xy(from_node, to_node)
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]
        dt = to_node[2] - from_node[2]
        if dt < self.dt:
            return to_node
        ratio_dt = self.dt / dt
        return ( from_node[0] + ratio_dt*dx,
                 from_node[1] + ratio_dt*dy,
                 from_node[2] + self.dt)

    # -------------------------------------------------------------------
    # Collision check
    # -------------------------------------------------------------------
    def is_colliding(self, from_node, to_node):
        x1, y1, t1 = from_node
        x2, y2, t2 = to_node
        return _is_colliding_jit(
            x1, y1, t1, x2, y2, t2,
            self.x0_obs, self.y0_obs, self.t0_obs, self.t1_obs,
            self.vx_obs, self.vy_obs, self.r_obs,
            self.xc_tab, self.yc_tab, self.cos_tab, self.sin_tab,
            self.hx_tab, self.hy_tab
        )

    # -------------------------------------------------------------------
    # Build RRT
    # -------------------------------------------------------------------
    def build(self):
        """
        Reset variables if needed, then build the RRT.
        """
        # Tree storage: shape = (4, max_iter) => (x, y, t, parent_idx)
        self.tree = np.zeros((4, self.max_iter), dtype=float)
        self.tree[3, :] = -1.0  # parent = -1 => not defined
        self.tree[0:3, 0] = self.start  # node 0 is start
        self.n_nodes = 1

        # Store pre-sampled random nodes for speed purposes
        self.random_nodes = np.zeros((self.max_iter, 3), dtype=float)
        u_t = np.random.uniform(0, 1, self.max_iter)
        self.random_nodes[:, 2] = self.start[2] + (self.t_range[1] - self.start[2]) * (u_t ** (1/3))
        # R and Theta
        dt = self.random_nodes[:, 2] - self.start[2]
        r_max = self.v_max * dt
        r = np.sqrt(np.random.uniform(0, 1, self.max_iter)) * r_max
        theta = np.random.uniform(0, 2*np.pi, self.max_iter)
        self.random_nodes[:, 0] = self.start[0] + r*np.cos(theta)  # x
        self.random_nodes[:, 1] = self.start[1] + r*np.sin(theta)  # y

        for i in range(self.max_iter-1):
            # 1) Sample
            if random.random() < self.goal_bias:
                d0 = self.distance_xy(self.start, self.goal)
                t_goal = self.start[2] + d0 / (self.v_max * 0.8) + 1e-6
                rnd_node = (self.goal[0], self.goal[1], t_goal)
            else:
                rnd_node = self.random_nodes[i]
            # 2) Nearest feasible
            parent_idx = self.feasible_nearest(rnd_node)
            if parent_idx is None:
                continue
            parent_node = (
                self.tree[0, parent_idx],
                self.tree[1, parent_idx],
                self.tree[2, parent_idx]
            )
            # 3) Steer
            new_node = self.steer(parent_node, rnd_node)
            # Reject if spatial step too small
            dxy = self.distance_xy(parent_node, new_node)
            if dxy < self.min_spatial_step:
                continue
            # 4) Collision check
            if self.is_colliding(parent_node, new_node):
                continue
            # Insert
            self.tree[0:3, self.n_nodes] = new_node
            self.tree[3, self.n_nodes]   = parent_idx
            self.n_nodes += 1

            # 5) Check goal proximity
            dxy = self.distance_xy(new_node, self.goal)
            if dxy < self.goal_tolerance_xy:
                dt = dxy / self.v_max
                goal_node = (self.goal[0], self.goal[1], new_node[2] + dt)
                # Last segment check (from the actually inserted new_node)
                if self.is_colliding(new_node, goal_node):
                    continue
                # Add goal node with parent = index of new_node we just inserted
                new_node_idx = self.n_nodes - 1
                self.tree[0:3, self.n_nodes] = goal_node
                self.tree[3, self.n_nodes]   = new_node_idx
                self.n_nodes += 1
                if self.verbose:
                    print(f"[INFO] Goal approché à l'itération {i}.")
                return True
        if self.verbose:
            print("[WARN] No valid path found.")
        return False

    # -------------------------------------------------------------------
    # 7) Final path extraction
    # -------------------------------------------------------------------
    def get_path(self):
        """
        Find the node whose (x,y) is closest to (goal.x, goal.y),
        then climb back up parents to root.
        """
        N = self.n_nodes
        if N == 0:
            return []

        goal_xy = np.array([self.goal[0], self.goal[1]])
        tree_xy = self.tree[0:2, :N].T  # (N,2)

        dist_goal = np.sum((tree_xy - goal_xy)**2, axis=1)
        best_idx  = np.argmin(dist_goal)

        path = []
        curr_idx = best_idx
        while curr_idx != -1:
            x = self.tree[0, curr_idx]
            y = self.tree[1, curr_idx]
            t = self.tree[2, curr_idx]
            par = int(self.tree[3, curr_idx])
            path.append( (x, y, t) )
            curr_idx = par

        path.reverse()
        return path

# =====================================================
# Visualization
# =====================================================
def plot_cone(ax, start_node, t_max, v_max, resolution=30):
    """
    Visualize the sampling cone (from start_node up to t_max).
    """
    x0, y0, t0 = start_node
    t_vals = np.linspace(t0, t_max, resolution)
    theta_vals = np.linspace(0, 2*np.pi, resolution)

    T, TH = np.meshgrid(t_vals, theta_vals)
    R = v_max * (T - t0)
    R[R < 0] = 0.0

    X = x0 + R*np.cos(TH)
    Y = y0 + R*np.sin(TH)
    Z = T
    ax.plot_wireframe(X, Y, Z, color='gray', linewidth=0.5, alpha=0.4)

def plot_cylinder(ax, center, direction, height, radius, resolution=10):
    """
    Plots an oblique cylinder in 3D.
    center = (x0, y0, t0)
    direction = (vx, vy, 1)
    """
    x0, y0, _ = center
    vx, vy, _ = direction

    theta_vals = np.linspace(0, 2*np.pi, resolution)
    z_vals = np.linspace(0, height, resolution)

    TH, ZZ = np.meshgrid(theta_vals, z_vals)
    # Parametric coords
    X = x0 + vx*ZZ + radius*np.cos(TH)
    Y = y0 + vy*ZZ + radius*np.sin(TH)
    Z = ZZ

    ax.plot_surface(X, Y, Z, color='red', alpha=1)

def visualize_3d(rrt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N = rrt.n_nodes
    xs = rrt.tree[0, :N]
    ys = rrt.tree[1, :N]
    ts = rrt.tree[2, :N]
    ax.scatter(xs, ys, ts, c='blue', s=10, marker='.', label='Nodes')

    path = rrt.get_path()
    if len(path) > 1:
        px, py, pt = zip(*path)
        ax.plot(px, py, pt, color='magenta', linewidth=2, label='Path')

        # Plot path nodes as spheres
        for (x, y, t) in path:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            sphere_x = x + rrt.radius * np.cos(u) * np.sin(v)
            sphere_y = y + rrt.radius * np.sin(u) * np.sin(v)
            sphere_z = t + rrt.radius * np.cos(v)
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='cyan', alpha=0.6)

    # Start
    ax.scatter(rrt.start[0], rrt.start[1], rrt.start[2],
               c='green', marker='o', s=80, label='Start')

    # Goal (vertical line in T)
    ax.plot([rrt.goal[0], rrt.goal[0]],
            [rrt.goal[1], rrt.goal[1]],
            [rrt.t_range[0], rrt.t_range[1]],
            c='green', linewidth=2, label='Goal')

    # Cone
    plot_cone(ax, rrt.start, rrt.t_range[1], rrt.v_max)

    # Obstacles
    # We reconstruct them from the arrays just for plotting
    for (x0, y0, t0, vx, vy, r) in zip(rrt.x0_obs, rrt.y0_obs, rrt.t0_obs,
                                      rrt.vx_obs, rrt.vy_obs, rrt.r_obs):
        plot_cylinder(ax, (x0, y0, t0), (vx, vy, 1), rrt.t_range[1], r)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("T")
    ax.legend()
    ax.set_title("RRT spatio-temporel (optimized)")
    plt.show()

def animate_path_2d(rrt, obstacles, fps=25, save_path=None):
    """
    Animation 2D (x-y) du chemin spatio-temporel de `rrt`
    et des obstacles dynamiques.

    Parameters
    ----------
    rrt : RRTSpatioTemporal déjà construit et valide
    obstacles : liste d'instances Obstacle
    fps : int, images par seconde
    save_path : str ou None, ex. 'trajet.mp4' ou 'trajet.gif'
    """
    # ------------------------------------------------------------------
    # 0. Récupération du chemin et chronologie
    path = rrt.get_path()
    if len(path) < 2:
        raise ValueError("Chemin vide ; rien à animer.")

    pts = np.array(path)           # shape (N,3)  -> (x,y,t)
    t_min, t_max = pts[0,2], pts[-1,2]
    duration     = t_max - t_min
    n_frames     = int(np.ceil(duration * fps)) + 1
    times        = np.linspace(t_min, t_max, n_frames)

    # ------------------------------------------------------------------
    # 1. Pré-calcul de la position des obstacles aux temps des frames
    obs_xy = []
    for obs in obstacles:
        x = obs.x0 + times * obs.vx
        y = obs.y0 + times * obs.vy
        obs_xy.append((x, y, obs.radius))
    # ------------------------------------------------------------------
    # 2. Pré-calcul position robot (interp linéaire par segment)
    rob_x = np.interp(times, pts[:,2], pts[:,0])
    rob_y = np.interp(times, pts[:,2], pts[:,1])

    # ------------------------------------------------------------------
    # 3. Figure
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Animation 2D du chemin")

    # Limites de la scène
    margin = 1.5 * max(max(rrt.v_max * (rrt.t_range[1]-rrt.t_range[0]), 1),
                       max(obs.radius for obs in obstacles))
    xmin = min(np.min(rob_x), np.min([ox[0] for ox in obs_xy])) - margin
    xmax = max(np.max(rob_x), np.max([ox[0] for ox in obs_xy])) + margin
    ymin = min(np.min(rob_y), np.min([oy[1] for oy in obs_xy])) - margin
    ymax = max(np.max(rob_y), np.max([oy[1] for oy in obs_xy])) + margin
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True)

    # ------------------------------------------------------------------
    # 4. Artistes initiaux
    robot_dot, = ax.plot([], [], "bo", markersize=6, label="Robot")
    goal_dot  ,= ax.plot(rrt.goal[0], rrt.goal[1], "gx", markersize=8, label="But")
    start_dot ,= ax.plot(rrt.start[0], rrt.start[1], "go", markersize=8, label="Départ")
    path_line ,= ax.plot([], [], "b--", linewidth=1, alpha=0.4)
    obs_patches = []
    for _ in obstacles:
        patch = plt.Circle((0,0), 1, color="r", alpha=0.4)
        ax.add_patch(patch)
        obs_patches.append(patch)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # 5. Fonctions d’init et de mise à jour
    def init():
        robot_dot.set_data([], [])
        path_line.set_data([], [])
        for patch in obs_patches:
            patch.center = (-999, -999)  # hors cadre
        return [robot_dot, path_line, *obs_patches]

    def update(frame):
        # 1) robot (x, y doivent être des séquences)
        robot_dot.set_data([rob_x[frame]], [rob_y[frame]])
        path_line.set_data(rob_x[:frame+1], rob_y[:frame+1])

        # 2) obstacles
        for k, patch in enumerate(obs_patches):
            patch.center = (obs_xy[k][0][frame], obs_xy[k][1][frame])
            patch.radius = obs_xy[k][2]
        return [robot_dot, path_line, *obs_patches]


    # ------------------------------------------------------------------
    # 6. Animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   init_func=init, blit=True,
                                   interval=1000/fps)

    # 7. Enregistrement éventuel
    if save_path is not None:
        ext = save_path.split(".")[-1].lower()
        if ext == "gif":
            anim.save(save_path, writer="pillow", fps=fps)
        else:  # mp4 (ou autre supporté par FFMpeg)
            anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"[INFO] Animation enregistrée dans “{save_path}”")

    plt.show()

# =====================================================
# Example main
# =====================================================
def main():
    start = (0, 0, 0)
    goal  = (7, 7)
    t_range = (0, 30)
    n_obstacles = 10

    import time
    timestart = time.time()

    # Repeat multiple times for average timing
    N_TRIALS = 100
    for _ in range(N_TRIALS):
        # Random obstacle generation
        obstacles = []
        for _ in range(n_obstacles):
            x0 = random.uniform(-10, 10)
            y0 = random.uniform(-10, 10)
            vx = random.uniform(-1, 1)
            vy = random.uniform(-1, 1)
            obstacles.append(Obstacle(x0, y0, 0, vx, vy, 0.4))

        # Build RRT
        rrt = RRTSpatioTemporal(
            start=start,
            goal=goal,
            obstacles=obstacles,
            t_range=t_range,
            v_max=1.0,
            max_iter=4000,
        )
        is_valid = rrt.build()
        if not is_valid:
            print("[WARN] No valid path found.")
            path = []
        else:
            path = rrt.get_path()
            # animate_path_2d(rrt, obstacles, fps=60, save_path=None)
            visualize_3d(rrt)

    elapsed = (time.time() - timestart) / N_TRIALS
    print(f"[INFO] Average time over {N_TRIALS} trials: {elapsed:.6f} s")


# Profiler entry point
if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.run("main()")
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(20)
