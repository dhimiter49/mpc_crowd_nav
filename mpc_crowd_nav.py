import sys
import fancy_gym
import gymnasium as gym
import numpy as np
import scipy
from numpy import ndarray
from qpsolvers import solve_qp


PHYSCIAL_SPACE = 0.4
AGENT_MAX_VEL = 3.0
AGENT_MAX_ACC = 1.5
DT = 0.1
MAX_STEPS = 4 / DT  # The episode is 4 seconds
N = 20
TIME_TO_STOP_FROM_MAX = AGENT_MAX_VEL / AGENT_MAX_ACC
DIST_TO_STOP_FROM_MAX = TIME_TO_STOP_FROM_MAX ** 2 * AGENT_MAX_ACC * 0.5

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
M_xa = scipy.linalg.toeplitz(
    np.array([(2 * i - 1) / 2 * DT ** 2 for i in range(1, N + 1)]),
    np.zeros(N)
)
# M_xa = scipy.linalg.toeplitz(np.arange(N, 0, -1) * DT, np.zeros(N))
M_xa = np.stack(
    [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
).reshape(2 * N ,2 * N)
M_va = scipy.linalg.toeplitz(np.ones(N), np.zeros(N)) * DT
M_va = np.stack(
    [np.hstack([M_va, M_va * 0]), np.hstack([M_va * 0, M_va])]
).reshape(2 * N ,2 * N)
crowd_const_mat = lambda n_crowd : np.zeros((n_crowd, 4 * n_crowd))
alpha, beta, epsilon = 0.1, 10, 1e-2


def linear_planner(goal_vec):
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
    steps = np.zeros((N, 2))
    vels= np.stack([AGENT_MAX_VEL / np.linalg.norm(goal_vec) * goal_vec] * N).reshape(N, 2)
    if AGENT_MAX_VEL * DT > np.linalg.norm(goal_vec):
        oneD_steps = np.array([np.linalg.norm(goal_vec)])
    else:
        oneD_steps = np.arange(
            AGENT_MAX_VEL * DT, np.linalg.norm(goal_vec), 1.1 * AGENT_MAX_VEL * DT
        )
    twoD_steps = np.array([
        i / np.linalg.norm(goal_vec) * goal_vec for i in oneD_steps
    ])
    n_steps = min(N, len(oneD_steps))
    steps[:n_steps,:] = twoD_steps[:n_steps]
    steps[n_steps:,:] += goal_vec
    vels[n_steps:,:] = np.zeros(2)
    return np.hstack([steps[:, 0], steps[:, 1]]), np.hstack([vels[:, 0], vels[:, 1]])


def qp(goal_vec, agent_vel, agent_acc):
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
    opt_M = 60000 * M_xa ** 2
    opt_V = (-np.repeat(goal_vec, N) + M_xv * np.repeat(agent_vel, N)) @ M_xa
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


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
    opt_M = 60000 * M_xa ** 2
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)) @ M_xa
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def qp_planning_vel(reference_plan, reference_vels, agent_vel):
    """
    Optimize navigation by using a reference plan for the upcoming horizon.

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
    opt_M = 10 * (M_xa ** 2) + 40 * M_va ** 2
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)) @ M_xa +\
        0.1 * (-reference_vels) @ M_va
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def qp_terminal(reference_plan, agent_vel, goal_vec, step):
    """
    Optimize navigation by using a reference plan for the upcoming horizon and speficying
    a terminal constraint on the position of the agent.

    Args:
        goal_vec (numpy.ndarray): vector in 2 plane going from current position to goal
        agent_vel (numpy.ndarray): vector representing the current agent velocity
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    # horizon = min(MAX_STEPS - step, N) if "-dh" in sys.argv else N
    opt_M = (0.5 + M_xa ** 2) * np.eye(2 * N)
    opt_V = M_xa.T @ (-reference_plan + M_xv * np.repeat(agent_vel, N))
    acc_b = np.ones(2 * N) * (AGENT_MAX_ACC +  0.5) * DT

    if np.linalg.norm(goal_vec) <= DIST_TO_STOP_FROM_MAX + AGENT_MAX_VEL * DT:
        time_after_stop = (np.linalg.norm(goal_vec) / AGENT_MAX_ACC * 2) ** 0.5
        start = int(-(- time_after_stop //  DT))  # ceiling function
        eq_con_M = np.stack(
            [M_xa[start : N], M_xa[N + start :]]
        ).reshape(-1, 2 * N)
        _M_xv = np.stack([M_xv[start : N], M_xv[N + start :]]).flatten()
        eq_con_b = np.repeat(goal_vec, N - start) -\
            _M_xv * np.repeat(agent_vel, N - start)
        acc = solve_qp(
            opt_M, opt_V, A=eq_con_M, b=eq_con_b, lb=-acc_b, ub=acc_b, solver="clarabel"
        )
    else:
        acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def calculate_sep_plane(crowd_pos, old_sep_plane, next_crowd_pos=None):
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


def calculate_sep_planes(crowd_poss, old_sep_planes, next_crowd_poss=None):
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
        M_cc[i][i * 4 : i * 4 + 2] = pos
        M_cc[i][i * 4 + 2] = -1

    # robot position is always origin
    M_ca = crowd_const_mat(n_crowd) # agent constraint matrix
    for i in range(n_crowd):
        M_ca[i][i * 4 + 2 : i * 4 + 4] = -1
    M_cs = crowd_const_mat(n_crowd)  # sep normal constraint
    for i in range(n_crowd):
        M_cs[i][i * 4 : i * 4 + 2] = 1
    M_cls = crowd_const_mat(n_crowd)  # old sep normal constraint
    for i in range(n_crowd):
        M_cls[i][i * 4 : i * 4 + 2] = old_sep_planes[i * 3 : i * 3 + 2]

    opt_M = np.eye((4 * n_crowd)) * beta
    for i in range(n_crowd):
        opt_M[i * 4 + 3, i * 4 + 3] = alpha
    opt_V = -np.ones(4 * n_crowd)
    for i in range(n_crowd):
        opt_V[i * 4 : i * 4 + 3] *= 2 * old_sep_planes[i * 3 : i * 3 + 3]
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


def qp_planning_col_avoid(reference_plan, agent_vel, sep_planes, crowd_poss):
    """
    Optimize navigation by using a reference plan for the upcoming horizon and use
    collision avoidance constraints on crowd where each member is assumed to be a circle
    with constant velocity

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
        sep_planes (numpy.ndarray): vector representing the separating planes between
            crowd and agent with the shape of len(crowd_poss) * 3, with 2 coefficients
            representing the normal and the third being the displacement from the origin
        crowd_poss (numpy.ndarray): 2D position of each member of the crowd
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 10 * M_xa ** 2
    opt_V = (-reference_plan + M_xv * np.repeat(agent_vel, N)) @ M_xa
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def calculate_crowd_positions(crowd_poss, crowd_vels):
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
    return np.stack(crowd_poss, N) + np.einsum(
        'ijk,i->ijk', np.stack([crowd_vels] * N, 0), np.arange(-1, N)
    )

def sep_planes_from_plan(plan, num_crowd):
    """
    From given plan given as a vector calculate the normal representation. Assuming that
    b is 0 since the agent is always at coordinate 0. if vector is [x, y] then normal is
    [-y, x] then x(-y) + yx = 0
    """
    normal = np.array([-plan[N], plan[0], 0])
    return np.array([normal / np.linalg.norm(normal)] * num_crowd)


separating_planes = None
if "-c" in sys.argv:
    env = gym.make("fancy/CrowdNavigationStatic-v0", width=20, height=20)
else:
    env = gym.make("fancy/CrowdNavigationStatic-v0", width=20, height=20)
returns, return_, vels, action = [], 0, [], [0, 0]
step_counter = 0
obs = env.reset()
print("Observation shape: ", env.observation_space.shape)
print("Action shape: ", env.action_space.shape)

from tqdm import tqdm

for t in [0.5 * i for i in range(1)]:
    coeff = t
    returns, return_ = [], 0
    for i in tqdm(range(4000)):
        step_counter += 1
        if isinstance(obs, tuple):
            goal_vec, crowd_poss, agent_vel = obs[0][:2], obs[0][2:-2], obs[0][-2:]
        else:
            goal_vec, crowd_poss, agent_vel = obs[:2], obs[2:-2], obs[-2:]
        crowd_poss.resize(len(crowd_poss) // 2, 2)

        if "-lpv" in sys.argv or "-lp" in sys.argv or "-tc" in sys.argv or "-c" in sys.argv:
            planned_steps, planned_vels = linear_planner(goal_vec)
            steps= np.zeros((N, 2))
            steps[:, 0] = planned_steps[:N]
            steps[:, 1] = planned_steps[N:]
            # env.set_trajectory(steps - steps[0])
            if "-tc" in sys.argv:
                action = qp_terminal(
                    planned_steps, agent_vel, goal_vec, step_counter
                )
            elif "-lpv" in sys.argv:
                planned_vels[:N] -= agent_vel[0]
                planned_vels[N:] -= agent_vel[1]
                action = qp_planning_vel(planned_steps, planned_vels, agent_vel)
            elif "-c" in sys.argv:
                if separating_planes is None:
                    separating_planes = sep_planes_from_plan(planned_steps, len(crowd_poss))
                new_separating_planes = []
                for i, member in enumerate(crowd_poss):
                    new_separating_planes.append(
                        calculate_sep_plane(member, separating_planes[i])
                    )
                # separating_planes = calculate_sep_planes(crowd_poss, separating_planes)
                separating_planes = np.array(new_separating_planes)
                env.set_separating_planes(separating_planes)
                action = qp_planning_col_avoid(
                    planned_steps, agent_vel, separating_planes, crowd_poss
                )
            else:
                action = qp_planning(planned_steps, agent_vel)
        else:
            action = qp(goal_vec, agent_vel, action)

        vels.append(np.linalg.norm(env.current_vel))
        obs, reward, terminated, truncated, info = env.step(action)
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
            obs = env.reset()
    print("Coeff: " , coeff, " Mean: ", np.mean(returns))
