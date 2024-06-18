import sys
import numpy as np
from tqdm import tqdm
import fancy_gym
import gymnasium as gym

from mpc.mpc_factory import get_mpc
from plan import Plan
from obs_handler import ObsHandler


MPC_DICT = {
    "-d": "simple",
    "-lp": "linear_plan",
    "-v": "velocity_control",
    "-cs": "cascading",
    "-vcs": "velocity_control_cascading"
}

ENV_DICT = {
    "-d": "Navigation",
    "-c": "CrowdNavigationStatic",
    "-mc": "CrowdNavigation",
}

PLAN_DICT = {
    "-lp": "Position",
    "-lpv": "PositionVelocity",
    "-vp": "Velocity",
}


velocity_str = "Vel" if "-v" in sys.argv else ""
if "-c" in sys.argv or "-csc" in sys.argv:
    env_type = ENV_DICT["-c"]
    mpc_type = MPC_DICT["-lp"]
    env = gym.make("fancy/CrowdNavigationStatic%s-v0" % velocity_str)
elif "-mc" in sys.argv or "-csmc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    mpc_type = MPC_DICT["-lp"]
    env = gym.make("fancy/CrowdNavigation%s-v0" % velocity_str)
else:
    env_type = ENV_DICT["-d"]
    mpc_type = MPC_DICT["-d"]
    env = gym.make("fancy/Navigation%s-v0" % velocity_str)
obs_handler = ObsHandler(env_type, env.get_wrapper_attr("n_crowd"))
render = "-nr" not in sys.argv

N = 21
M = 20
DT = env.unwrapped.dt

mpc_kwargs = {}
plan_type = ""
if "-lpv" in sys.argv:
    plan_type = PLAN_DICT["-lpv"]
    mpc_kwargs["plan_type"] = plan_type
elif "-vp" in sys.argv:
    plan_type = PLAN_DICT["-vp"]
    mpc_kwargs["plan_type"] = plan_type
elif "-lp" in sys.argv:
    plan_type = PLAN_DICT["-lp"]
    mpc_kwargs["plan_type"] = plan_type

plan_steps = N
if "-v" in sys.argv and "-cs" in sys.argv:
    mpc_type = MPC_DICT["-vcs"]
    plan_steps = M
    mpc_kwargs["plan_length"] = M
elif "-v" in sys.argv:
    mpc_type = MPC_DICT["-v"]
elif "-cs" in sys.argv:
    mpc_type = MPC_DICT["-cs"]
    plan_steps = M
    mpc_kwargs["plan_length"] = M
elif "-lp" in sys.argv or "-lpv" in sys.argv or "-vp" in sys.argv:
    mpc_type = MPC_DICT["-lp"]
else:
    mpc_type = MPC_DICT["-d"]
planner = Plan(plan_steps + 10, DT, env.get_wrapper_attr("AGENT_MAX_VEL"))


mpc = get_mpc(
    mpc_type,
    horizon=N,
    dt=DT,
    physical_space=env.get_wrapper_attr("PHYSICAL_SPACE"),
    agent_max_vel=env.get_wrapper_attr("AGENT_MAX_VEL"),
    agent_max_acc=env.get_wrapper_attr("MAX_ACC"),
    n_crowd=env.get_wrapper_attr("n_crowd"),
    **mpc_kwargs
)

obs = env.reset()
plan = np.zeros((N, 2))
returns, ep_return, vels, action = [], 0, [], [0, 0]
ep_step = 0
for i in tqdm(range(100000)):
    obs = obs_handler(obs)
    if ep_step % 10 == 0:
        replan_plan = planner.plan(obs)
        og_pos = env.get_wrapper_attr("current_pos")
        pos_plan, vel_plan = replan_plan
        pos_plan = np.array([pos_plan[:plan_steps + 10], pos_plan[plan_steps + 10:]]).T
        vel_plan = np.array([vel_plan[:plan_steps + 10], vel_plan[plan_steps + 10:]]).T
        pos_plan = pos_plan + og_pos
        replan_plan = np.concatenate([pos_plan[:, 0], pos_plan[:, 1]]), \
            np.concatenate([vel_plan[:, 0], vel_plan[:, 1]])

        pos_plan = pos_plan[:plan_steps]
        vel_plan = vel_plan[:plan_steps]
        pos_plan = np.concatenate([pos_plan[:, 0], pos_plan[:, 1]])
        vel_plan = np.concatenate([vel_plan[:, 0], vel_plan[:, 1]])
        plan = pos_plan, vel_plan
    else:
        entire_pos_plan, entire_vel_plan = replan_plan
        entire_pos_plan = np.array(
            [entire_pos_plan[:plan_steps + 10], entire_pos_plan[plan_steps + 10:]]
        ).T
        entire_vel_plan = np.array(
            [entire_vel_plan[:plan_steps + 10], entire_vel_plan[plan_steps + 10:]]
        ).T
        curr_pos = env.get_wrapper_attr("current_pos")
        min_dist_point = np.argmin(
            np.linalg.norm(curr_pos - entire_pos_plan, axis=-1)
        )
        closest_point = entire_pos_plan[min_dist_point]
        pos_plan = entire_pos_plan[min_dist_point:min_dist_point + plan_steps]
        vel_plan = entire_vel_plan[min_dist_point:min_dist_point + plan_steps]
        # vectors to destination
        closest_point_vec = closest_point - entire_pos_plan[-1]
        current_pos_vec = curr_pos - entire_pos_plan[-1]
        if np.linalg.norm(closest_point_vec) > 0:
            closest_point_vec = closest_point_vec / np.linalg.norm(closest_point_vec)
        if np.linalg.norm(current_pos_vec) > 0:
            current_pos_vec = current_pos_vec / np.linalg.norm(current_pos_vec)
        angle = -np.arctan2(*current_pos_vec) + np.arctan2(*closest_point_vec)
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]
        ])
        pos_plan_ = np.empty_like(pos_plan)
        for i, (pos, vel) in enumerate(zip(pos_plan, vel_plan)):
            pos_vec = pos - entire_pos_plan[-1]
            rot_pos_vec = rot_mat @ pos_vec
            pos_plan_[i] = pos + rot_pos_vec - pos_vec
        pos_plan = pos_plan_
        pos_plan -= curr_pos
        pos_plan = np.concatenate([pos_plan[:, 0], pos_plan[:, 1]])
        vel_plan = np.concatenate([vel_plan[:, 0], vel_plan[:, 1]])
        plan = pos_plan, vel_plan


    None if mpc_type == "simple" else env.get_wrapper_attr("set_trajectory")(
        *planner.prepare_plot(plan, plan_steps)
    )
    env.get_wrapper_attr("set_separating_planes")() if "Crowd" in env_type else None
    control_plan = mpc.get_action(plan, obs)
    obs, reward, terminated, truncated, info = env.step(control_plan[0])
    ep_return += reward
    env.render() if render else None
    ep_step += 1
    if terminated or truncated:
        ep_step = 0
        obs = env.reset()
        returns.append(ep_return)
        ep_return = 0
print("Mean: ", np.mean(returns))
