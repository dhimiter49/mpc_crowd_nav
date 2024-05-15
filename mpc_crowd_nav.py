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
TIME_TO_STOP_FROM_MAX = AGENT_MAX_VEL / AGENT_MAX_ACC
DIST_TO_STOP_FROM_MAX = TIME_TO_STOP_FROM_MAX ** 2 * AGENT_MAX_ACC * 0.5
MAX_STOPPING_DIST = 2 * DIST_TO_STOP_FROM_MAX
DT = 0.1
"""
The horizon needs to be at least the minimal length of a braking trajectory plus one step.
In order to achieve maximal velocity the braking trajecotry would be MAX_VEL / MAX_ACC.
Assuming an horizon of one then the robot would not move at all since at the next step it
has to be able to break.
In this case MAX_VEL / MAX_ACC = 2 and DT is 0.1 which means that the horizon has to be
at leas 2 / 0.1 + 1 = 21 in order to achieve maximal velocity.
"""
N = 21
M = 20


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
    return np.array(polygon_lines)


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

The letter b in the matrix definitions below (like M_ba, M_bva) refers to the matrives
relevant for cascading MPC where a braking trajectory is computed for each step in the
future horizon.

Whereas matrices starting with MV (like MV_xv, MV_a) are used to refere to matrices that
reflect the dynamics when using velocity as control.
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
M_xa = np.stack(
    [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
).reshape(2 * N, 2 * N)
M_ba = np.stack(
    [np.hstack([M_ba, M_ba * 0]), np.hstack([M_ba * 0, M_ba])]
).reshape(2 * M * N, 2 * M * N)

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

MV_xv = scipy.linalg.toeplitz(np.ones(N), np.zeros(N)) * DT
np.fill_diagonal(MV_xv, 1 / 2 * DT)
MV_xv = MV_xv[:, :-1]
MV_bv = np.zeros((M * N, M * (N - 1)))
for i in range(M):
    MV_bv[i * N:i * N + N, i * (N - 1):i * (N - 1) + N - 1] = MV_xv
    for j in range(i):
        MV_bv[i * N:(i + 1) * N, j * N] = np.ones(N) * DT
MV_xv = np.stack(
    [np.hstack([MV_xv, MV_xv * 0]), np.hstack([MV_xv * 0, MV_xv])]
).reshape(2 * N, 2 * (N - 1))
MV_bv = np.stack(
    [np.hstack([MV_bv, MV_bv * 0]), np.hstack([MV_bv * 0, MV_bv])]
).reshape(2 * M * N, 2 * M * (N - 1))

acc_from_vel = np.zeros(N)
acc_from_vel[:2] = np.array([1, -1])
MV_a = scipy.linalg.toeplitz(acc_from_vel, np.zeros(N)) / DT
MV_a = MV_a[:, :-1]
MV_b_a = np.zeros((M * N, M * (N - 1)))
for i in range(M):
    MV_b_a[i * N:i * N + N, i * (N - 1):i * (N - 1) + N - 1] = MV_a
    if i > 0:
        MV_b_a[i * N, (i - 1) * (N - 1)] = -1 / DT
MV_a = np.stack(
    [np.hstack([MV_a, MV_a * 0]), np.hstack([MV_a * 0, MV_a])]
).reshape(2 * N, 2 * (N - 1))
MV_b_a = np.stack(
    [np.hstack([MV_b_a, MV_b_a * 0]), np.hstack([MV_b_a * 0, MV_b_a])]
).reshape(2 * M * N, 2 * M * (N - 1))

# Filters for cascading, retrieves the indexes relevant only for the objective (reference
# trajectory)
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

MV_bv_f = MV_bv[np.nonzero(F_p)]

# Precomputing acceleration and velocity constraints
horizon = N
m_v = M_va
if "-csc" in sys.argv or "-csmc" in sys.argv:
    if "-v" in sys.argv:
        horizon = (N - 1) * M
        comp = M
        m_b_a = MV_b_a
    else:
        horizon = N * M
        m_v = M_bva
elif "-v" in sys.argv:
    horizon = N - 1
    comp = 1
    m_b_a = MV_a

if "-v" not in sys.argv:
    # velocity constraint using the inner polygon of a circle with radius AGENT_MAX_VEL
    M_v_ = np.vstack([np.eye(horizon) * -line[0] for line in POLYGON_VEL_LINES])
    M_v_ = np.hstack([M_v_, np.vstack([np.eye(horizon)] * len(POLYGON_VEL_LINES))])
    sgn_vel = np.ones(len(POLYGON_VEL_LINES))
    sgn_vel[len(POLYGON_VEL_LINES) // 2:] = -1
    sgn_vel = np.repeat(sgn_vel, horizon)
    b_v_ = np.repeat(POLYGON_VEL_LINES[:, 1], horizon)

    VEL_MAT_CONST = ((M_v_ @ m_v).T * sgn_vel).T


    def vel_vec_const(agent_vel):
        return sgn_vel * (b_v_ - M_v_ @ np.repeat(agent_vel, horizon))


    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    M_a_ = np.vstack([np.eye(horizon) * -line[0] for line in POLYGON_ACC_LINES])
    M_a_ = np.hstack([M_a_, np.vstack([np.eye(horizon)] * len(POLYGON_ACC_LINES))])
    sgn_acc = np.ones(len(POLYGON_ACC_LINES))
    sgn_acc[len(POLYGON_ACC_LINES) // 2:] = -1
    sgn_acc = np.repeat(sgn_acc, horizon)
    b_a_ = np.repeat(POLYGON_ACC_LINES[:, 1], horizon)

    ACC_MAT_CONST = (M_a_.T * sgn_acc).T
    ACC_VEC_CONST = sgn_acc * b_a_
else:
    # velocity/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_VEL
    MV_v_ = np.vstack([np.eye(horizon) * -line[0] for line in POLYGON_VEL_LINES])
    MV_v_ = np.hstack([MV_v_, np.vstack([np.eye(horizon)] * len(POLYGON_VEL_LINES))])
    sgn_vel = np.ones(len(POLYGON_VEL_LINES))
    sgn_vel[len(POLYGON_VEL_LINES) // 2:] = -1
    sgn_vel = np.repeat(sgn_vel, horizon)
    b_a_ = np.repeat(POLYGON_VEL_LINES[:, 1], horizon)

    VEL_VEL_MAT_CONST = (MV_v_.T * sgn_vel).T
    VEL_VEL_VEC_CONST = sgn_vel * b_a_


    # acceleration/control constraint using the inner polygon of a circle with radius
    # AGENT_MAX_ACC
    MV_a_ = np.vstack([np.eye(horizon + comp) * -line[0] for line in POLYGON_ACC_LINES])
    MV_a_ = np.hstack([
        MV_a_, np.vstack([np.eye(horizon + comp)] * len(POLYGON_ACC_LINES))
    ])
    vel_sgn_acc = np.ones(len(POLYGON_ACC_LINES))
    vel_sgn_acc[len(POLYGON_ACC_LINES) // 2:] = -1
    vel_sgn_acc = np.repeat(vel_sgn_acc, horizon + comp)
    bv_a_ = np.repeat(POLYGON_ACC_LINES[:, 1], horizon + comp)

    VEL_ACC_MAT_CONST = ((MV_a_ @ m_b_a).T * vel_sgn_acc).T


    def vel_acc_vec_const(agent_vel):
        agent_vel_ = np.zeros(2 * (horizon + comp))
        agent_vel_[0], agent_vel_[horizon + comp] = agent_vel
        return vel_sgn_acc * (bv_a_ + MV_a_ @ agent_vel_ / DT)


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
    const_M.append(VEL_MAT_CONST)
    const_b.append(vel_vec_const(agent_vel))
    const_M.append(ACC_MAT_CONST)
    const_b.append(ACC_VEC_CONST)

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
    const_M.append(ACC_MAT_CONST)
    const_b.append(ACC_VEC_CONST)
    const_M.append(VEL_MAT_CONST)
    const_b.append(vel_vec_const(agent_vel))

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel"
    )
    return np.array([acc[:N], acc[N:]]).T


def qp_vel_planning(reference_plan, agent_vel):
    """
    Velocity control.
    Optimize navigation by using a reference plan for the upcoming horizon.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the new velocity to be
            applied in the next step
    """
    opt_M = MV_xv.T @ MV_xv + 0.25 * np.eye(2 * (N - 1))
    opt_V = (-reference_plan + 0.5 * DT * np.repeat(agent_vel, N)).T @ MV_xv

    const_M = []
    const_b = []
    const_M.append(VEL_VEL_MAT_CONST)
    const_b.append(VEL_VEL_VEC_CONST)
    const_M.append(VEL_ACC_MAT_CONST)
    const_b.append(vel_acc_vec_const(agent_vel))

    vel = solve_qp(
        opt_M, opt_V,
        # lb=-vel_b, ub=vel_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        solver="clarabel"
    )
    return np.array([np.append(vel[:N - 1], 0), np.append(vel[N - 1:], 0)]).T


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
    const_M.append(ACC_MAT_CONST)
    const_b.append(ACC_VEC_CONST)
    const_M.append(VEL_MAT_CONST)
    const_b.append(vel_vec_const(agent_vel))

    acc = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        A=term_const_M, b=term_const_b,
        solver="clarabel"
    )
    return np.array([acc[:N], acc[N:]]).T


def qp_vel_planning_vel(reference_plan, reference_vels, agent_vel):
    """
    Velocity control.
    Optimize navigation by using a reference plan for the upcoming horizon. The plan
    includes position and velocity.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        reference_vels (numpy.ndarray): vector of reference velocities with same size as
            the given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the new velocity to be
            applied in the next step
    """
    opt_M = MV_xv.T @ MV_xv + 0.25 * np.eye(2 * (N - 1))
    opt_V = (-reference_plan + 0.5 * DT * np.repeat(agent_vel, N)).T @ MV_xv -\
        0.25 * reference_vels.T

    const_M = []
    const_b = []
    const_M.append(VEL_VEL_MAT_CONST)
    const_b.append(VEL_VEL_VEC_CONST)
    const_M.append(VEL_ACC_MAT_CONST)
    const_b.append(vel_acc_vec_const(agent_vel))

    vel = solve_qp(
        opt_M, opt_V,
        # lb=-vel_b, ub=vel_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        solver="clarabel"
    )
    return np.array([np.append(vel[:N - 1], 0), np.append(vel[N - 1:], 0)]).T


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
        if np.all(np.linalg.norm(poss, axis=-1) > MAX_STOPPING_DIST):
            continue
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

    const_M.append(ACC_MAT_CONST)
    const_b.append(ACC_VEC_CONST)
    const_M.append(VEL_MAT_CONST)
    const_b.append(vel_vec_const(agent_vel))

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


def qp_vel_planning_col_avoid(reference_plan, agent_vel, crowd_poss, old_plan, agent_pos):
    """
    Velocity control.
    Optimize navigation by using a reference plan for the upcoming horizon and use
    collision avoidance constraints on crowd where each member is assumed to be a circle.

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
        crowd_poss (numpy.ndarray): 2D position of each member of the crowd
        old_plan (numpy.ndarrray): plan from last iteration
    Return:
        (numpy.ndarray): array with two elements representing the new velocity to be
            applied in the next step
    """
    opt_M = MV_xv.T @ MV_xv + 0.25 * np.eye(2 * (N - 1))
    opt_V = (-reference_plan + 0.5 * DT * np.repeat(agent_vel, N)).T @ MV_xv

    const_M = []  # constraint matrices
    const_b = []  # constraint bounds
    for member in range(len(crowd_poss[1])):
        poss = crowd_poss[:, member, :]
        if np.all(np.linalg.norm(poss, axis=-1) > MAX_STOPPING_DIST):
            continue
        vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
        M_ca = np.hstack([np.eye(N) * vec[:, 0], np.eye(N) * vec[:, 1]])
        v_cb = M_ca @ (-poss.flatten("F") + 0.5 * DT * np.repeat(agent_vel, N)) -\
            np.array([4 * PHYSICAL_SPACE] * N)
        M_cac = -M_ca @ MV_xv
        const_M.append(M_cac)
        const_b.append(v_cb)

    const_M.append(VEL_VEL_MAT_CONST)
    const_b.append(VEL_VEL_VEC_CONST)
    const_M.append(VEL_ACC_MAT_CONST)
    const_b.append(vel_acc_vec_const(agent_vel))

    vel = solve_qp(
        opt_M, opt_V,
        # lb=-acc_b, ub=acc_b,
        G=np.vstack(const_M), h=np.hstack(const_b),
        solver="clarabel",
        tol_gap_abs=5e-5,
        tol_gap_rel=5e-5,
        tol_feas=1e-4,
        tol_infeas_abs=5e-5,
        tol_infeas_rel=5e-5,
        tol_ktratio=1e-4
    )

    global LAST_PREDICTED_STATES, LAST_PREDICTED_VELOCITY
    if vel is None:
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
        vel = old_plan.flatten("F")
    else:
        position = np.repeat(agent_pos, N) + 0.5 * DT * np.repeat(agent_vel, N) +\
            MV_xv @ vel
        LAST_PREDICTED_STATES = np.array([position[:N], position[N:]]).T
        LAST_PREDICTED_VELOCITY = old_plan
        IN_VIOLATION = False

    return np.array([np.append(vel[:N - 1], 0), np.append(vel[N - 1:], 0)]).T


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
    opt_M = 0.075 * M_bva_f.T @ M_bva_f + M_ba_f.T @ M_ba_f
    opt_V = (-reference_plan + M_bv_f * np.repeat(agent_vel, M)).T @ M_ba_f +\
        0.075 * np.repeat(agent_vel, M) @ M_bva_f

    const_M = []  # constraint matrices
    const_b = []  # constraint bounds
    for member in range(len(crowd_poss[1])):
        poss = crowd_poss[:, member, :]
        if np.all(np.linalg.norm(poss, axis=-1) > MAX_STOPPING_DIST):
            continue
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

    const_M.append(ACC_MAT_CONST)
    const_b.append(ACC_VEC_CONST)
    const_M.append(VEL_MAT_CONST)
    const_b.append(vel_vec_const(agent_vel))

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
        position = np.repeat(agent_pos, M) + M_bv_f * np.repeat(agent_vel, M) + \
            M_ba_f @ acc
        velocity = np.repeat(agent_vel, M) + M_bva_f @ acc
        LAST_PREDICTED_STATES = np.array([position[:M], position[M:]]).T
        LAST_PREDICTED_VELOCITY = np.array([velocity[:M], velocity[M:]]).T
        IN_VIOLATION = False
        acc = np.hstack([acc[:N], acc[M * N:M * N + N]])

    return np.array([acc[:N], acc[N:]]).T


def qp_vel_planning_casc_safety(
        reference_plan, agent_vel, crowd_poss, old_plan, agent_pos
):
    """
    Velocity control.
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
    opt_M = MV_bv_f.T @ MV_bv_f + 0.02 * np.eye(2 * M * (N - 1))
    opt_V = (-reference_plan + 0.5 * DT * np.repeat(agent_vel, M)).T @ MV_bv_f

    const_M = []  # constraint matrices
    const_b = []  # constraint bounds
    for member in range(len(crowd_poss[1])):
        poss = crowd_poss[:, member, :]
        if np.all(np.linalg.norm(poss, axis=-1) > MAX_STOPPING_DIST):
            continue
        vec = -poss / np.stack([np.linalg.norm(poss, axis=-1)] * 2, axis=-1)
        M_ca = np.hstack([np.eye(M * N) * vec[:, 0], np.eye(M * N) * vec[:, 1]])
        v_cb = M_ca @ (-poss.flatten("F") + 0.5 * DT * np.repeat(agent_vel, M * N)) -\
            np.array([4 * PHYSICAL_SPACE] * M * N)
        M_cac = -M_ca @ MV_bv
        const_M.append(M_cac)
        const_b.append(v_cb)

    const_M.append(VEL_VEL_MAT_CONST)
    const_b.append(VEL_VEL_VEC_CONST)
    const_M.append(VEL_ACC_MAT_CONST)
    const_b.append(vel_acc_vec_const(agent_vel))

    vel = solve_qp(
        opt_M, opt_V,
        G=np.vstack(const_M), h=np.hstack(const_b),
        solver="clarabel",
        tol_gap_abs=5e-5,
        tol_gap_rel=5e-5,
        tol_feas=1e-4,
        tol_infeas_abs=5e-5,
        tol_infeas_rel=5e-5,
        tol_ktratio=1e-4
    )

    global LAST_PREDICTED_STATES, LAST_PREDICTED_VELOCITY
    if vel is None:
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
        vel = old_plan.flatten("F")
    else:
        position = np.repeat(agent_pos, M) + 0.5 * DT * np.repeat(agent_vel, M) +\
            MV_bv_f @ vel
        vel = np.hstack([vel[:N - 1], vel[M * (N - 1):M * (N - 1) + (N - 1)]])
        vel_ = np.array(
            [np.append(vel[:N - 1], 0), np.append(vel[N - 1:], 0)]
        ).T
        LAST_PREDICTED_STATES = np.array([position[:M], position[M:]]).T
        IN_VIOLATION = False
        LAST_PREDICTED_VELOCITY = vel_

    return np.array([np.append(vel[:N - 1], 0), np.append(vel[N - 1:], 0)]).T


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


def intersect(ls_a, ls_b):
    return bool(set(ls_a).intersection(ls_b))


separating_planes = None
velocity_str = "Vel" if "-v" in sys.argv else ""
if intersect(["-c", "-csc"], sys.argv):
    env = gym.make("fancy/CrowdNavigationStatic%s-v0" % velocity_str, width=40)
elif intersect(["-mc", "-csmc"], sys.argv):
    env = gym.make("fancy/CrowdNavigation%s-v0" % velocity_str)
else:
    env = gym.make("fancy/Navigation%s-v0" % velocity_str)
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

        if intersect(["-lpv", "-lp", "-c", "-mc", "-csmc", "-csc"], sys.argv):
            planned_steps, planned_vels = linear_planner(goal_vec)
            steps = np.zeros((N, 2))
            steps_vel = np.zeros((N, 2))
            steps[:, 0] = planned_steps[:N]
            steps[:, 1] = planned_steps[N:]
            steps_vel[:, 0] = planned_vels[:N]
            steps_vel[:, 1] = planned_vels[N:]
            env.set_trajectory(steps - steps[0], steps_vel)
            if "-lpv" in sys.argv:
                planned_vels[:N] -= agent_vel[0]
                planned_vels[N:] -= agent_vel[1]
                if "-v" in sys.argv:
                    planned_vels = np.append(
                        planned_vels[:N - 1], planned_vels[N:2 * N - 1]
                    )
                    plan = qp_vel_planning_vel(planned_steps, planned_vels, agent_vel)
                else:
                    plan = qp_planning_vel(planned_steps, planned_vels, agent_vel)
            elif "-c" in sys.argv or "-mc" in sys.argv:
                env.set_separating_planes()
                crowd_vels = crowd_vels if "-mc" in sys.argv else crowd_poss * 0
                horizon_crowd_poss = calculate_crowd_positions(crowd_poss, crowd_vels)
                if "-v" in sys.argv:
                    plan = qp_vel_planning_col_avoid(
                        planned_steps,
                        agent_vel,
                        horizon_crowd_poss,
                        plan[1:],
                        env.current_pos
                    )
                else:
                    plan = qp_planning_col_avoid(
                        planned_steps,
                        agent_vel,
                        horizon_crowd_poss,
                        plan[1:],
                        env.current_pos
                    )
            elif "-csc" in sys.argv or "-csmc" in sys.argv:
                # env.set_separating_planes()
                planned_steps, planned_vels = linear_planner(goal_vec, M)
                steps = np.zeros((M, 2))
                steps_vel = np.zeros((M, 2))
                steps[:, 0] = planned_steps[:M]
                steps[:, 1] = planned_steps[M:]
                steps_vel[:, 0] = planned_vels[:M]
                steps_vel[:, 1] = planned_vels[M:]
                env.set_trajectory(steps - steps[0], steps_vel)
                crowd_vels = crowd_vels if "-csmc" in sys.argv else crowd_poss * 0
                horizon_crowd_poss = calculate_crowd_positions(
                    crowd_poss, crowd_vels, M + N
                )
                horizon_crowd_poss = cascade_crowd_positions(horizon_crowd_poss)
                if "-v" in sys.argv:
                    plan = qp_vel_planning_casc_safety(
                        planned_steps,
                        agent_vel,
                        horizon_crowd_poss,
                        plan[1:],
                        env.current_pos
                    )
                else:
                    plan = qp_planning_casc_safety(
                        planned_steps,
                        agent_vel,
                        horizon_crowd_poss,
                        plan[1:],
                        env.current_pos
                    )
            else:
                if "-v" in sys.argv:
                    plan = qp_vel_planning(planned_steps, agent_vel)
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
