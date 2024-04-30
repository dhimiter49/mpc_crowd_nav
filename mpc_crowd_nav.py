import sys
from tqdm import tqdm
import fancy_gym
import gymnasium as gym
import numpy as np
import scipy
from qpsolvers import solve_qp


PHYSICAL_SPACE = 0.4
AGENT_MAX_VEL = 3.0
AGENT_MAX_ACC = 1.5
DT = 0.1
MAX_STEPS = 4 / DT  # The episode is 4 seconds
N = 21
M = 20
TIME_TO_STOP_FROM_MAX = AGENT_MAX_VEL / AGENT_MAX_ACC
DIST_TO_STOP_FROM_MAX = TIME_TO_STOP_FROM_MAX ** 2 * AGENT_MAX_ACC * 0.5


# debug
COUNTER_VIOLATIONS = 0
IN_VIOLATION = False
LAST_PREDICTED_STATES = np.zeros((N, 2))
LAST_PREDICTED_VELOCITY = np.zeros((N, 2))


def rot_mat(rad):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])


def gen_polygon(radius, sides=8):
    polygon = [[radius, 0]]
    for i in range(1, sides + 1):
        polygon.append(rot_mat(2 * np.pi / sides) @ polygon[i - 1])
    polygon_lines = []
    for i in range(sides):
        m = (polygon[i][1] - polygon[i + 1][1]) / (polygon[i][0] - polygon[i + 1][0])
        b = polygon[i][1] - m * polygon[i][0]
        polygon_lines.append([m, b])
    return polygon_lines


POLYGON_ACC_LINES = gen_polygon(AGENT_MAX_ACC, sides=8)
POLYGON_VEL_LINES = gen_polygon(AGENT_MAX_VEL, sides=8)


"""
Matrices representing dynamics

Dynamics: X = x_0 + M_xv * v_x0 * DT + M_xa A_x * DT ** 2
and:      Y = y_0 + M_xv * v_y0 * DT + M_ya A_y * DT ** 2

Linearizing x-y for the qpsolver. M_xv is [1,2,...,N] in order to generalize for position
in 2D (M_pv) we stack x-y one after the other and rewrite M_xv as [1,...N,1,...N].

In case of M_xa we generalize for A = (A_x | A_y)^T so M_xa (M_pa) becomes
M_xa = [[ M_xa |  0   ]
        [  0   | M_xa ]]
"""
M_xv = np.hstack([np.arange(1, N + 1)] * 2) * DT
M_bv = np.hstack([np.hstack([np.arange(i, N + i) for i in range(1, M + 1)])] * 2) * DT
M_xa = scipy.linalg.toeplitz(
    np.array([(2 * i - 1) / 2 * DT ** 2 for i in range(1, N + 1)]),
    np.zeros(N)
)
M_ba = np.zeros((M * N, M * N))
for i in range(M):
    M_ba[i * N:i * N + N, i * N:i * N + N] = M_xa
    for j in range(i):
        M_ba[i * N:(i + 1) * N, j * N] = np.repeat(DT ** 2 * (2 * (i + 1 - j) - 1) / 2, N)
M_ba = np.stack(
    [np.hstack([M_ba, M_ba * 0]), np.hstack([M_ba * 0, M_ba])]
).reshape(2 * M * N, 2 * M * N)
M_xa = np.stack(
    [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
).reshape(2 * N, 2 * N)
M_va = scipy.linalg.toeplitz(np.ones(N), np.zeros(N)) * DT
M_bva = np.zeros((M * N, M * N))
for i in range(M):
    M_bva[i * N:i * N + N, i * N:i * N + N] = M_va
    for j in range(i):
        M_bva[i * N:(i + 1) * N, j * N] = np.repeat(DT * 1, N)
M_bva = np.stack(
    [np.hstack([M_bva, M_bva * 0]), np.hstack([M_bva * 0, M_bva])]
).reshape(2 * M * N, 2 * M * N)
M_va = np.stack(
    [np.hstack([M_va, M_va * 0]), np.hstack([M_va * 0, M_va])]
).reshape(2 * N, 2 * N)
alpha, beta, epsilon = 0.1, 100, 1e-4

F_p = np.zeros(N, dtype=int)
F_p[0] = 1
F_p = np.hstack([np.hstack([F_p] * M)] * 2)
F_b = np.zeros(N, dtype=int)
F_b[-1] = 1
F_b = np.hstack([np.hstack([F_b] * M)] * 2)

M_bv_f = M_bv[np.nonzero(F_p)]
M_ba_f = M_ba[np.nonzero(F_p)]
M_bva_f = M_bva[np.nonzero(F_p)]
M_bva_b = M_bva[np.nonzero(F_b)]


def crowd_const_mat(n_crowd):
    return np.zeros((n_crowd, 4 * n_crowd))


def linear_planner(goal_vec, horizon=N):
    """
    A naive plan that draws a straight path from the current agent position to the goal.
    The trajecotry is discretized by the maximum distance that the agent can traverse in
    in one time step that is the (maximum_velocity x time_step). This reference positions
    are not feasible if the agent does not have a maximal current velocity. All referebnce
    steps after the goal are 'cliped' to the goal position. This means that the resulting
    plan when the agent is near the goal looks like [p_1, p_2,..., p_j, g, g, g,...g].

    Args:
        goal_vec (numpy.ndarray): vector in 2 plane going from current position to goal
    Return:
        (numpy.ndarray): array with size 2 * N where the first N elements are the x-coords
            and the second N elements are the y-coords
    """
    steps = np.zeros((horizon, 2))
    vels = np.stack(
        [(AGENT_MAX_VEL + 1) / np.linalg.norm(goal_vec) * goal_vec] * horizon
    ).reshape(horizon, 2)
    if AGENT_MAX_VEL * DT > np.linalg.norm(goal_vec):
        oneD_steps = np.array([np.linalg.norm(goal_vec)])
    else:
        oneD_steps = np.arange(
            AGENT_MAX_VEL * DT, np.linalg.norm(goal_vec), 2 * AGENT_MAX_VEL * DT
        )
    twoD_steps = np.array([
        i / np.linalg.norm(goal_vec) * goal_vec for i in oneD_steps
    ])
    n_steps = min(horizon, len(oneD_steps))
    steps[:n_steps, :] = twoD_steps[:n_steps]
    steps[n_steps:, :] += goal_vec
    vels[n_steps - 1:, :] = np.zeros(2)
    return np.hstack([steps[:, 0], steps[:, 1]]), np.hstack([vels[:, 0], vels[:, 1]])


def qp(goal_vec, agent_vel):
    """
    Naive solution to the navigation problem where the optimization is with regards to
    minimizing the distance between all steps of the horizon and the goal.

    Args:
        goal_vec (numpy.ndarray): vector in 2 plane going from current position to goal
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 0.5 * M_va.T @ M_va + M_xa.T @ M_xa
    opt_V = (-np.repeat(goal_vec, N) + M_xv * np.repeat(agent_vel, N)).T @ M_xa +\
        0.5 * np.repeat(agent_vel, N) @ M_va

    # passive safety
    # acc_b = np.ones(2 * N) * AGENT_MAX_ACC
    term_const_M = M_va[[N - 1, 2 * N - 1], :]
    term_const_b = -agent_vel

    const_M = []
    const_b = []
    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    for i, line in enumerate(POLYGON_ACC_LINES):
        sgn = 1 if i < len(POLYGON_ACC_LINES) / 2 else -1
        M_a = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_a = np.ones(N) * line[1]
        const_M.append(sgn * M_a)
        const_b.append(sgn * b_a)
    # velocity constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    for i, line in enumerate(POLYGON_VEL_LINES):
        sgn = 1 if i < len(POLYGON_VEL_LINES) / 2 else -1
        M_v = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_v = np.ones(N) * line[1] - M_v @ np.repeat(agent_vel, N)
        const_M.append(sgn * M_v @ M_va)
        const_b.append(sgn * b_v)

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel"
    )
    return np.array([acc[:N], acc[N:]]).T


def qp_planning(reference_plan, agent_vel):
    """
    Optimize navigation by using a reference plan for the upcoming horizon.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 0.25 * M_va.T @ M_va + M_xa.T @ M_xa
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)).T @ M_xa +\
        0.25 * np.repeat(agent_vel, N) @ M_va

    # passive safety
    # acc_b = np.ones(2 * N) * AGENT_MAX_ACC
    term_const_M = M_va[[N - 1, 2 * N - 1], :]
    term_const_b = -agent_vel

    const_M = []
    const_b = []
    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    for i, line in enumerate(POLYGON_ACC_LINES):
        sgn = 1 if i < len(POLYGON_ACC_LINES) / 2 else -1
        M_a = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_a = np.ones(N) * line[1]
        const_M.append(sgn * M_a)
        const_b.append(sgn * b_a)
    # velocity constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    for i, line in enumerate(POLYGON_VEL_LINES):
        sgn = 1 if i < len(POLYGON_VEL_LINES) / 2 else -1
        M_v = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_v = np.ones(N) * line[1] - M_v @ np.repeat(agent_vel, N)
        const_M.append(sgn * M_v @ M_va)
        const_b.append(sgn * b_v)

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel"
    )
    return np.array([acc[:N], acc[N:]]).T


def qp_planning_vel(reference_plan, reference_vels, agent_vel):
    """
    Optimize navigation by using a reference plan for the trajectory and the velocity in
    the upcoming horizon.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        reference_vels (numpy.ndarray): vector of reference velocities with same size as
            the given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 0.26 * np.eye(2 * N) + 0.2 * M_va.T @ M_va + M_xa.T @ M_xa
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)).T @ M_xa +\
        0.2 * (-reference_vels).T @ M_va

    # passive safety
    # acc_b = np.ones(2 * N) * AGENT_MAX_ACC
    term_const_M = M_va[[N - 1, 2 * N - 1], :]
    term_const_b = -agent_vel

    const_M = []
    const_b = []
    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    for i, line in enumerate(POLYGON_ACC_LINES):
        sgn = 1 if i < len(POLYGON_ACC_LINES) / 2 else -1
        M_a = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_a = np.ones(N) * line[1]
        const_M.append(sgn * M_a)
        const_b.append(sgn * b_a)
    # velocity constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    for i, line in enumerate(POLYGON_VEL_LINES):
        sgn = 1 if i < len(POLYGON_VEL_LINES) / 2 else -1
        M_v = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_v = np.ones(N) * line[1] - M_v @ np.repeat(agent_vel, N)
        const_M.append(sgn * M_v @ M_va)
        const_b.append(sgn * b_v)

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel"
    )
    return np.array([acc[:N], acc[N:]]).T


def calculate_sep_plane(crowd_pos):
    dist = np.linalg.norm(crowd_pos)
    return np.concatenate((crowd_pos / dist, [dist - 2 * PHYSICAL_SPACE]))


def opt_sep_plane(crowd_pos, old_sep_plane, next_crowd_pos=None):
    """
    Calculate the separating planes between agent and a single member of the crowd.

    Args:
        crowd_poss (numpy.ndarray): position of the person
        next_crowd_poss (numpy.ndarray): position of the person in the next iteration
            for the next step
        old_sep_planes (numpy.ndarray): the separation normals and constant from the last
            iteration
    Return:
        (numpy.ndarray): separating plane between agent and member of the crowd
    """
    M_cc = np.array([crowd_pos[0], crowd_pos[1], -1, 0])
    M_ca = np.array([0, 0, -1, -1])
    M_cs = np.array([1, 1, 0, 0])
    M_cls = np.array([old_sep_plane[0], old_sep_plane[1], 0, 0])

    opt_M = np.diag([beta, beta, beta, alpha])
    opt_V = np.concatenate((old_sep_plane, [-1]))
    sep_plane = solve_qp(
        opt_M, opt_V,
        G=np.vstack([M_cc, -M_ca, -M_cs, M_cs, -M_cls, M_cls]),
        h=np.hstack([
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            np.ones(1),
            epsilon - np.ones(1),
            np.ones(1),
        ]),
        solver="clarabel",
    )
    return sep_plane[:-1]


def opt_sep_planes(crowd_poss, old_sep_planes, next_crowd_poss=None):
    """
    Calculate the separating planes between agent and crowd.

    Args:
        crowd_poss (numpy.ndarray): position of the crowd relative to the agent
        next_crowd_poss (numpy.ndarray): position of the crowd relative to the agent for
            for the next step
        old_sep_planes (numpy.ndarray): the separation normals and constant from the last
            iteration
    Return:
        (numpy.ndarray): array of all separating planes betweeen agent and each member of
            the crowd
    """
    n_crowd = len(crowd_poss)
    M_cc = crowd_const_mat(n_crowd)  # crowd constraint matrix
    for i, pos in enumerate(crowd_poss):
        M_cc[i][i * 4:i * 4 + 2] = pos
        M_cc[i][i * 4 + 2] = -1

    # robot position is always origin
    M_ca = crowd_const_mat(n_crowd)  # agent constraint matrix
    for i in range(n_crowd):
        M_ca[i][i * 4 + 2:i * 4 + 4] = -1
    M_cs = crowd_const_mat(n_crowd)  # sep normal constraint
    for i in range(n_crowd):
        M_cs[i][i * 4:i * 4 + 2] = 1
    M_cls = crowd_const_mat(n_crowd)  # old sep normal constraint
    for i in range(n_crowd):
        M_cls[i][i * 4:i * 4 + 2] = old_sep_planes[i * 3:i * 3 + 2]

    opt_M = np.eye((4 * n_crowd)) * beta
    for i in range(n_crowd):
        opt_M[i * 4 + 3, i * 4 + 3] = alpha
    opt_V = -np.ones(4 * n_crowd)
    for i in range(n_crowd):
        opt_V[i * 4:i * 4 + 3] *= 2 * old_sep_planes[i * 3:i * 3 + 3]
    sep_planes = solve_qp(
        opt_M, opt_V,
        G=np.vstack([M_cc, -M_ca, -M_cs, M_cs, -M_cls, M_cls]),
        h=np.hstack([
            np.zeros(n_crowd),
            np.zeros(n_crowd),
            np.ones(n_crowd),
            np.ones(n_crowd),
            epsilon - np.ones(n_crowd),
            np.ones(n_crowd),
        ]),
        solver="clarabel",
    )
    rm_d = []
    for i in range(4 * n_crowd):
        if i % 4 != 3:
            rm_d.append(sep_planes[i])
    sep_planes = np.array(rm_d)
    return sep_planes


def qp_planning_col_avoid(reference_plan, agent_vel, crowd_poss, old_plan, agent_pos):
    """
    Optimize navigation by using a reference plan for the upcoming horizon and use
    collision avoidance constraints on crowd where each member is assumed to be a circle.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
        crowd_poss (numpy.ndarray): 2D position of each member of the crowd
        old_plan (numpy.ndarrray): plan from last iteration
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 0.25 * M_va.T @ M_va + M_xa.T @ M_xa
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)).T @ M_xa +\
        0.25 * np.repeat(agent_vel, N) @ M_va
    const_M = []  # constraint matrices
    const_b = []  # constraint bounds
    for member in range(len(crowd_poss[1])):
        poss = crowd_poss[:, member, :]
        vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
        M_ca = np.hstack([np.eye(N) * vec[:, 0], np.eye(N) * vec[:, 1]])
        v_cb = M_ca @ (-poss.flatten("F") + M_xv * np.repeat(agent_vel, N)) -\
            np.array([4 * PHYSICAL_SPACE] * N)
        M_cac = -M_ca @ M_xa
        const_M.append(M_cac)
        const_b.append(v_cb)

    # passive safety
    # acc_b = np.ones(2 * N) * AGENT_MAX_ACC
    term_const_M = M_va[[N - 1, 2 * N - 1], :]
    term_const_b = -agent_vel

    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    for i, line in enumerate(POLYGON_ACC_LINES):
        sgn = 1 if i < len(POLYGON_ACC_LINES) / 2 else -1
        M_a = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_a = np.ones(N) * line[1]
        const_M.append(sgn * M_a)
        const_b.append(sgn * b_a)
    # velocity constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    for i, line in enumerate(POLYGON_VEL_LINES):
        sgn = 1 if i < len(POLYGON_VEL_LINES) / 2 else -1
        M_v = np.hstack([np.eye(N) * -line[0], np.eye(N)])
        b_v = np.ones(N) * line[1] - M_v @ np.repeat(agent_vel, N)
        const_M.append(sgn * M_v @ M_va)
        const_b.append(sgn * b_v)

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel",
        tol_gap_abs=5e-5,
        tol_gap_rel=5e-5,
        tol_feas=1e-4,
        tol_infeas_abs=5e-5,
        tol_infeas_rel=5e-5,
        tol_ktratio=1e-4
    )

    global LAST_PREDICTED_STATES, LAST_PREDICTED_VELOCITY
    if acc is None:
        print("Executing last computed trajectory for braking!")
        global IN_VIOLATION, COUNTER_VIOLATIONS
        if not IN_VIOLATION:
            COUNTER_VIOLATIONS += 1
            IN_VIOLATION = True
        dist_to_crowd = np.linalg.norm(crowd_poss[0], axis=-1)
        print("Margin from sep plane: ", min(dist_to_crowd - 4 * PHYSICAL_SPACE))
        print("Current agent pos: ", agent_pos,
              " Diff in position: ", LAST_PREDICTED_STATES[0] - agent_pos)
        print("Velocity: ", agent_vel,
              " Diff in velocity: ", LAST_PREDICTED_VELOCITY[0] - agent_vel)
        # input()
        acc = np.zeros(2 * N)
        acc[0:N - 1] = old_plan[:, 0]
        acc[N:2 * N - 1] = old_plan[:, 1]
    else:
        position = np.repeat(agent_pos, N) + M_xv * np.repeat(agent_vel, N) + M_xa @ acc
        velocity = np.repeat(agent_vel, N) + M_va @ acc
        LAST_PREDICTED_STATES = np.array([position[:N], position[N:]]).T
        LAST_PREDICTED_VELOCITY = np.array([velocity[:N], velocity[N:]]).T
        IN_VIOLATION = False

    return np.array([acc[:N], acc[N:]]).T


def qp_planning_casc_safety(reference_plan, agent_vel, crowd_poss, old_plan, agent_pos):
    """
    Optimize navigation by using a reference plan for the upcoming horizon and use
    collision avoidance constraints on crowd where each member is assumed to be a circle.
    Safety is computed in a cascading manner where for each timestep in the horizon a
    braking trajecotry is computed. This alows us to decouple the trajectory for safety
    and that for planning.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
        crowd_poss (numpy.ndarray): 2D position of each member of the crowd
        old_plan (numpy.ndarrray): plan from last iteration
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 0.25 * M_bva_f.T @ M_bva_f + M_ba_f.T @ M_ba_f
    opt_V = (-reference_plan + M_bv_f * np.repeat(agent_vel, M)).T @ M_ba_f +\
        0.25 * np.repeat(agent_vel, M) @ M_bva_f
    const_M = []  # constraint matrices
    const_b = []  # constraint bounds
    for member in range(len(crowd_poss[1])):
        poss = crowd_poss[:, member, :]
        vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
        M_ca = np.hstack([np.eye(M * N) * vec[:, 0], np.eye(M * N) * vec[:, 1]])
        v_cb = M_ca @ (-poss.flatten("F") + M_bv * np.repeat(agent_vel, M * N)) -\
            np.array([4 * PHYSICAL_SPACE] * M * N)
        M_cac = -M_ca @ M_ba
        const_M.append(M_cac)
        const_b.append(v_cb)

    # passive safety
    # acc_b = np.ones(2 * M * N) * AGENT_MAX_ACC
    term_const_M = M_bva_b
    term_const_b = -np.repeat(agent_vel, M)

    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    for i, line in enumerate(POLYGON_ACC_LINES):
        sgn = 1 if i < len(POLYGON_ACC_LINES) / 2 else -1
        M_a = np.hstack([np.eye(M * N) * -line[0], np.eye(M * N)])
        b_a = np.ones(M * N) * line[1]
        const_M.append(sgn * M_a)
        const_b.append(sgn * b_a)
    # velocity constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    for i, line in enumerate(POLYGON_VEL_LINES):
        sgn = 1 if i < len(POLYGON_VEL_LINES) / 2 else -1
        M_v = np.hstack([np.eye(M * N) * -line[0], np.eye(M * N)])
        b_v = np.ones(M * N) * line[1] - M_v @ np.repeat(agent_vel, M * N)
        const_M.append(sgn * M_v @ M_bva)
        const_b.append(sgn * b_v)

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel",
        tol_gap_abs=5e-5,
        tol_gap_rel=5e-5,
        tol_feas=1e-4,
        tol_infeas_abs=5e-5,
        tol_infeas_rel=5e-5,
        tol_ktratio=1e-4
    )

    # global LAST_PREDICTED_STATES, LAST_PREDICTED_VELOCITY
    if acc is None:
        print("Executing last computed trajectory for braking!")
        global IN_VIOLATION, COUNTER_VIOLATIONS
        if not IN_VIOLATION:
            COUNTER_VIOLATIONS += 1
            IN_VIOLATION = True
        dist_to_crowd = np.linalg.norm(crowd_poss[0], axis=-1)
        print("Margin from sep plane: ", min(dist_to_crowd - 4 * PHYSICAL_SPACE))
        print("Current agent pos: ", agent_pos,
              " Diff in position: ", LAST_PREDICTED_STATES[0] - agent_pos)
        print("Velocity: ", agent_vel,
              " Diff in velocity: ", LAST_PREDICTED_VELOCITY[0] - agent_vel)
        # input()
        acc = np.zeros(2 * N)
        acc[0:N - 1] = old_plan[:, 0]
        acc[N:2 * N - 1] = old_plan[:, 1]
    else:
    #     position = np.repeat(agent_pos, N) + M_xv * np.repeat(agent_vel, N) + M_xa @ acc
    #     velocity = np.repeat(agent_vel, N) + M_va @ acc
    #     LAST_PREDICTED_STATES = np.array([position[:N], position[N:]]).T
    #     LAST_PREDICTED_VELOCITY = np.array([velocity[:N], velocity[N:]]).T
    #     IN_VIOLATION = False
        acc = np.hstack([acc[:N], acc[M * N:M * N + N]])

    return np.array([acc[:N], acc[N:]]).T


def calculate_crowd_positions(crowd_poss, crowd_vels, horizon=N):
    """
    Calculate the crowd positions for the next horizon given the constant velocity for
    each member. The formula P_i = p_0 + i * v * dt, where for point i in horizon the
    position will be p_0 + i * v * dt.

    Args:
        crowd_poss (numpy.ndarray): an array of size (n_crowd, 2) with the current
            positions of each member
        crowd_vels (numpy.ndarray): an array of size (n_crowd, 2) with the current
            velocities of each member
    Return:
        (numpy.ndarray): with the predicted positions of the crowd throughout the horizon
    """
    return np.stack([crowd_poss] * horizon) + np.einsum(
        'ijk,i->ijk', np.stack([crowd_vels] * horizon, 0) * DT, np.arange(0, horizon)
    )


def cascade_crowd_positions(crowd_poss):
    """
    Take crowd positions and cascade them meaning from [1,..., M + N] converrt to
    [1, 2,.., N, 2,..., N + 1, 3,..., M, M + 2,..., M + N].

    Args:
        crowd_poss (numpy.ndarray): an array of size (n_crowd, 2) with the current
            positions of each member
    Return:
        (numpy.ndarray): with the predicted positions of the crowd throughout the horizon
    """
    casc_crowd_poss = np.zeros((M * N,) + crowd_poss.shape[1:])
    for i in range(M):
        casc_crowd_poss[i * N:(i + 1) * N, :, :] = crowd_poss[i:i + N, :, :]
    return casc_crowd_poss


def sep_planes_from_plan(plan, num_crowd):
    """
    From given plan given as a vector calculate the normal representation. Assuming that
    b is 0 since the agent is always at coordinate 0. if vector is [x, y] then normal is
    [-y, x] then x(-y) + yx = 0
    """
    normal = np.array([-plan[N], plan[0], 0])
    return np.array([normal / np.linalg.norm(normal)] * num_crowd)


separating_planes = None
intersect = lambda ls_a, ls_b: bool(set(ls_a).intersection(ls_b))
if intersect(["-c", "-csc"], sys.argv):
    env = gym.make("fancy/CrowdNavigationStatic-v0")
elif intersect(["-mc", "-csmc"], sys.argv):
    env = gym.make("fancy/CrowdNavigation-v0")
else:
    env = gym.make("fancy/Navigation-v0", width=40, height=20)
returns, return_, vels, action = [], 0, [], [0, 0]
step_counter = 0
ep_counter = 0
obs = env.reset()
plan = np.ones((N, 2))
print("Observation shape: ", env.observation_space.shape)
print("Action shape: ", env.action_space.shape)


for t in [0.5 * i for i in range(1)]:
    coeff = t
    returns, return_ = [], 0
    for i in tqdm(range(4000)):
        step_counter += 1
        if intersect(["-mc", "-c", "-csmc", "-csc"], sys.argv):
            n_crowd = 4 * 2
            if intersect(["-mc", "-csmc"], sys.argv):
                if isinstance(obs, tuple):
                    goal_vec, crowd_poss, agent_vel, crowd_vels, wall_dist = (
                        obs[0][: 2],
                        obs[0][2:2 + n_crowd],
                        obs[0][n_crowd + 2: n_crowd + 4],
                        obs[0][n_crowd + 4:2 * n_crowd + 4],
                        obs[0][2 * n_crowd + 4:]
                    )
                else:
                    pos_idx = len(obs) // 2
                    goal_vec, crowd_poss, agent_vel, crowd_vels, wall_dist = (
                        obs[: 2],
                        obs[2:2 + n_crowd],
                        obs[2 + n_crowd:n_crowd + 4],
                        obs[n_crowd + 4:2 * n_crowd + 4],
                        obs[2 * n_crowd + 4:]
                    )
                crowd_vels.resize(len(crowd_vels) // 2, 2)
            if intersect(["-c", "-csc"], sys.argv):
                if isinstance(obs, tuple):
                    goal_vec, crowd_poss, agent_vel, wall_dist = (
                        obs[0][: 2],
                        obs[0][2:n_crowd + 2],
                        obs[0][n_crowd + 2:n_crowd + 4],
                        obs[0][n_crowd + 4:]
                    )
                else:
                    pos_idx = len(obs) // 2
                    goal_vec, crowd_poss, agent_vel, wall_dist = (
                        obs[:2],
                        obs[2:n_crowd + 2],
                        obs[n_crowd + 2:n_crowd + 4],
                        obs[n_crowd + 4:]
                    )
            crowd_poss.resize(len(crowd_poss) // 2, 2)
        else:
            if isinstance(obs, tuple):
                goal_vec, agent_vel = obs[0][:2], obs[0][2:4]
            else:
                goal_vec, agent_vel = obs[:2], obs[2:4]

        if intersect(["-lpv", "-lp", "-tc", "-c", "-mc", "-csmc", "-csc"], sys.argv):
            planned_steps, planned_vels = linear_planner(goal_vec)
            steps = np.zeros((N, 2))
            steps[:, 0] = planned_steps[:N]
            steps[:, 1] = planned_steps[N:]
            env.set_trajectory(steps - steps[0], planned_vels)
            if "-tc" in sys.argv:
                plan = qp_terminal(
                    planned_steps, agent_vel, goal_vec, step_counter
                )
            elif "-lpv" in sys.argv:
                planned_vels[:N] -= agent_vel[0]
                planned_vels[N:] -= agent_vel[1]
                plan = qp_planning_vel(planned_steps, planned_vels, agent_vel)
            elif "-c" in sys.argv:
                env.set_separating_planes()
                horizon_crowd_poss = calculate_crowd_positions(crowd_poss, crowd_poss * 0)
                plan = qp_planning_col_avoid(
                    planned_steps,
                    agent_vel,
                    horizon_crowd_poss,
                    plan[1:],
                    env.current_pos
                )
            elif "-mc" in sys.argv:
                env.set_separating_planes()
                horizon_crowd_poss = calculate_crowd_positions(crowd_poss, crowd_vels)
                plan = qp_planning_col_avoid(
                    planned_steps,
                    agent_vel,
                    horizon_crowd_poss,
                    plan[1:],
                    env.current_pos
                )
            elif "-csc" in sys.argv:
                # env.set_separating_planes()
                planned_steps, planned_vels = linear_planner(goal_vec, M)
                steps = np.zeros((M, 2))
                steps[:, 0] = planned_steps[:M]
                steps[:, 1] = planned_steps[M:]
                env.set_trajectory(steps - steps[0], planned_vels)
                horizon_crowd_poss = calculate_crowd_positions(
                    crowd_poss, crowd_poss * 0, M + N
                )
                horizon_crowd_poss = cascade_crowd_positions(horizon_crowd_poss)
                plan = qp_planning_casc_safety(
                    planned_steps,
                    agent_vel,
                    horizon_crowd_poss,
                    plan[1:],
                    env.current_pos
                )
            elif "-csmc" in sys.argv:
                # env.set_separating_planes()
                planned_steps, planned_vels = linear_planner(goal_vec, M)
                steps = np.zeros((M, 2))
                steps[:, 0] = planned_steps[:M]
                steps[:, 1] = planned_steps[M:]
                env.set_trajectory(steps - steps[0], planned_vels)
                horizon_crowd_poss = calculate_crowd_positions(
                    crowd_poss, crowd_vels, M + N
                )
                horizon_crowd_poss = cascade_crowd_positions(horizon_crowd_poss)
                plan = qp_planning_casc_safety(
                    planned_steps,
                    agent_vel,
                    horizon_crowd_poss,
                    plan[1:],
                    env.current_pos
                )
            else:
                plan = qp_planning(planned_steps, agent_vel)
        else:
            plan = qp(goal_vec, agent_vel)

        vels.append(np.linalg.norm(env.current_vel))
        obs, reward, terminated, truncated, info = env.step(plan[0])
        return_ += reward
        None if "-nr" in sys.argv else env.render()
        if terminated or truncated:
            # print(return_)
            # print(np.max(vels))
            separating_planes = None
            vels = []
            returns.append(return_)
            return_ = 0
            step_counter = 0
            ep_counter += 1
            obs = env.reset()
    print("Coeff: ", coeff, " Mean: ", np.mean(returns))
    print("Episodes: ", ep_counter, " Braking executions: ", COUNTER_VIOLATIONS)
