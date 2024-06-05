import sys
import numpy as np
from tqdm import tqdm
import fancy_gym
import gymnasium as gym

from mpc.mpc_factory import get_mpc
from plan import Plan
from obs_handler import ObsHandler


MPC_DICT = {
    "-s": "simple",
    "-lr": "linear_plan",
    "-v": "velocity_control",
    "-cs": "cascading"
}

ENV_DICT = {
    "-c": "CrowdNavigationStatic",
    "-mc": "CrowdNavigation",
    "-s": "Navigation"
}


velocity_str = "Vel" if "-v" in sys.argv else ""
if "-c" in sys.argv or "-csc" in sys.argv:
    env_type = ENV_DICT["-c"]
    mpc_type = MPC_DICT["-c"]
    env = gym.make("fancy/CrowdNavigationStatic%s-v0" % velocity_str)
elif "-mc" in sys.argv or "-csmc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    mpc_type = MPC_DICT["-c"]
    env = gym.make("fancy/CrowdNavigation%s-v0" % velocity_str)
else:
    env_type = ENV_DICT["-s"]
    mpc_type = MPC_DICT["-s"]
    env = gym.make("fancy/Navigation%s-v0" % velocity_str)
obs_handler = ObsHandler(env_type, env.n_crowd)
render = "-nr" not in sys.argv

N = 21
M = 20
DT = env.dt


if "-lp" in sys.argv:
    mpc_type = MPC_DICT["-lp"]
elif "-v" in sys.argv:
    mpc_type = MPC_DICT["-v"]
else:
    mpc_type = MPC_DICT["-s"]


mpc = get_mpc(
    mpc_type,
    horizon=N,
    dt=DT,
    physical_space=env.PHYSICAL_SPACE,
    agent_max_vel=env.AGENT_MAX_VEL,
    agent_max_acc=env.MAX_ACC,
)

obs = env.reset()
plan = np.zeros((N, 2))
for i in tqdm(range(40000)):
    obs = obs_handler(obs)
    plan = mpc.get_action(None, obs)
    obs, reward, terminated, truncated, info = env.step(plan[0])
    env.render() if render else None
    if terminated or truncated:
        obs = env.reset()
