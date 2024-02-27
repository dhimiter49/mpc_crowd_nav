import sys
import fancy_gym
import gymnasium as gym
import numpy as np
import scipy
from numpy import ndarray
from qpsolvers import solve_qp


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
M_xa = scipy.linalg.toeplitz(np.arange(N, 0, -1) * DT ** 2, np.zeros(N))
M_xa[0] = (3 * N - 1) / 6 * DT ** 2
M_xa[-1] = 1 / 6 * DT ** 2
M_xa = np.stack(
    [np.hstack([M_xa, M_xa * 0]), np.hstack([M_xa * 0, M_xa])]
).reshape(2 * N ,2 * N)
M_va = scipy.linalg.toeplitz(np.ones(N), np.zeros(N)) * DT
M_va[0] *= 0.5
M_va[-1] = 0.5
M_va = np.stack(
    [np.hstack([M_va, M_va * 0]), np.hstack([M_va * 0, M_va])]
).reshape(2 * N ,2 * N)


def planning_steps(goal_vec):
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
    if AGENT_MAX_VEL * DT > np.linalg.norm(goal_vec):
        oneD_steps = np.array([np.linalg.norm(goal_vec)])
    else:
        oneD_steps = np.arange(
            AGENT_MAX_VEL * DT, np.linalg.norm(goal_vec), AGENT_MAX_VEL * DT
        )
    twoD_steps = np.array([
        i / np.linalg.norm(goal_vec) * goal_vec for i in oneD_steps
    ])
    n_steps = min(N, len(oneD_steps))
    steps[:n_steps,:] = twoD_steps[:n_steps]
    steps[n_steps:,:] += goal_vec
    return np.hstack([steps[:, 0], steps[:, 1]])


def qp_solution(goal_vec, agent_vel):
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
    opt_M = (0.5 + M_xa ** 2 * DT ** 2) * np.eye(2 * N)
    opt_V = M_xa.T @ (-np.repeat(goal_vec, N) + M_xv * np.repeat(agent_vel, N))
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC * DT

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def qp_solution_planning(reference_plan, agent_vel):
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
    opt_M = (0.5 + M_xa ** 2 * DT ** 2) * np.eye(2 * N)
    opt_V = M_xa.T @ (-reference_plan + M_xv * np.repeat(agent_vel, N))
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC * DT

    acc = solve_qp(opt_M, opt_V, lb=-acc_b, ub=acc_b, solver="clarabel")
    return np.array([acc[0], acc[N]])


def qp_solution_terminal(reference_plan, agent_vel, goal_vec, step):
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
    opt_M = (0.5 + M_xa ** 2 * DT ** 2) * np.eye(2 * N)
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


def qp_solution_collision_avoidance(reference_plan, agent_vel, crowd_poss, crowd_vels):
    """
    Optimize navigation by using a reference plan for the upcoming horizon and use
    collision avoidance constraints on crowd where each member is assumed to be a circle
    with constant velocity

    Args:
        reference_plan (numpy.ndarray): vector of reference points with same size as the
            given horizon
        agent_vel (numpy.ndarray): vector representing the current agent velocity
        crowd_poss (numpy.ndarray): 2D position of each member of the crowd
        crowd_vels (numpy.ndarray): 2D velocity of each member of the crowd
    Return:
        (numpy.ndarray): array with two elements representing the change in velocity (acc-
            eleration) to be applied in the next step
    """
    opt_M = 5000 * (M_xa ** 2 * DT ** 4) * np.eye(2 * N)
    opt_V = M_xa.T @ (-reference_plan + M_xv * np.repeat(agent_vel, N))
    acc_b = np.ones(2 * N) * AGENT_MAX_ACC * DT

    # collision constraints
    con_M = M_xa.T @ (-crowd_poss + M_xv * np.repeat(agent_vel, N))
    h = PERSONAL_SPACE * 2

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


env = gym.make("fancy/Navigation-v0", width=20, height=20)
returns, return_, vels, action = [], 0, [], [0, 0]
step_counter = 0
obs = env.reset()
print("Observation shape: ", env.observation_space.shape)
print("Action shape: ", env.action_space.shape)

for i in range(400):
    step_counter += 1
    if isinstance(obs, tuple):
        goal_vec, agent_vel = obs[0][:2], obs[0][-2:]
    else:
        goal_vec, agent_vel = obs[:2], obs[-2:]

    if "-lp" in sys.argv or "-tc" in sys.argv:
        planned_steps = planning_steps(goal_vec)
        steps= np.zeros((N, 2))
        steps[:, 0] = planned_steps[:N]
        steps[:, 1] = planned_steps[N:]
        env.set_trajectory(steps - steps[0])
        if "-tc" in sys.argv:
            action = qp_solution_terminal(
                planned_steps, agent_vel, goal_vec, step_counter
            )
        else:
            action = qp_solution_planning(planned_steps, agent_vel)
    else:
        action = qp_solution(goal_vec, agent_vel)

    obs, reward, terminated, truncated, info = env.step(action)
    return_ += reward
    None if "-nr" in sys.argv else env.render()
    if terminated or truncated:
        print(return_)
        returns.append(return_)
        return_ = 0
        step_counter = 0
        obs = env.reset()
print("Mean: ", np.mean(returns))
