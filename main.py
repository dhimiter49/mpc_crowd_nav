import sys
import numpy as np
from tqdm import tqdm
import fancy_gym
import gymnasium as gym

from mpc.factory import get_mpc
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
    "-mcc": "CrowdNavigationConst",
}

PLAN_DICT = {
    "-lp": "Position",
    "-lpv": "PositionVelocity",
    "-vp": "Velocity",
}


gen_data = "-gd" in sys.argv

velocity_str = "Vel" if "-v" in sys.argv else ""
env_str = ""
if "-c" in sys.argv:
    env_type = ENV_DICT["-c"]
    env_str = "CrowdNavigationStatic%s-v0" % velocity_str
elif "-mc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigation%s-v0" % velocity_str
elif "-mcc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigationConst%s-v0" % velocity_str
else:
    env_type = ENV_DICT["-d"]
    env_str = "Navigation%s-v0" % velocity_str
env = gym.make("fancy/" + env_str)
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
planner = Plan(plan_steps, DT, env.get_wrapper_attr("AGENT_MAX_VEL"))


mpc = get_mpc(
    mpc_type,
    horizon=N,
    dt=DT,
    physical_space=env.get_wrapper_attr("PHYSICAL_SPACE"),
    const_dist_crowd=env.get_wrapper_attr("PHYSICAL_SPACE") * 2 + 0.1,
    agent_max_vel=env.get_wrapper_attr("AGENT_MAX_VEL"),
    agent_max_acc=env.get_wrapper_attr("MAX_ACC"),
    n_crowd=env.get_wrapper_attr("n_crowd"),
    **mpc_kwargs
)

steps = 100000
obs = env.reset()
plan = np.zeros((N, 2))
returns, ep_return, vels, action = [], 0, [], np.array([0, 0])
ep_count = 0
dataset = np.empty((
    steps,
    np.sum(env.observation_space.shape) * 2 + np.sum(env.action_space.shape) + 1 + 1 + 1
))
for i in tqdm(range(steps)):
    old_obs = obs[0].copy() if isinstance(obs, tuple) else obs.copy()
    obs = obs_handler(obs)
    plan = planner.plan(obs)
    None if mpc_type == "simple" else env.get_wrapper_attr("set_trajectory")(
        *planner.prepare_plot(plan, plan_steps)
    )
    env.get_wrapper_attr("set_separating_planes")() if "Crowd" in env_type else None
    control_plan = mpc.get_action(plan, obs)
    obs, reward, terminated, truncated, info = env.step(control_plan[0])
    if gen_data:
        dataset[i] = np.hstack([
            old_obs.flatten(),
            obs.flatten(),
            control_plan[0].flatten(),
            np.array(reward),
            np.array(terminated),
            np.array(truncated)
        ])
    ep_return += reward
    env.render() if render else None
    if terminated or truncated:
        obs = env.reset()
        returns.append(ep_return)
        ep_return = 0
        ep_count += 1
if gen_data:
    np.save("dataset_" + env_str + ".npy", dataset)
print("Mean: ", np.mean(returns))
print("Number of episodes", ep_count)
