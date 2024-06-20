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
    "-mcc": "CrowdNavigationConst",
}

PLAN_DICT = {
    "-lp": "Position",
    "-lpv": "PositionVelocity",
    "-vp": "Velocity",
}


velocity_str = "Vel" if "-v" in sys.argv else ""
if "-c" in sys.argv:
    env_type = ENV_DICT["-c"]
    env = gym.make("fancy/CrowdNavigationStatic%s-v0" % velocity_str)
elif "-mc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env = gym.make("fancy/CrowdNavigation%s-v0" % velocity_str)
elif "-mcc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env = gym.make("fancy/CrowdNavigationConst%s-v0" % velocity_str)
else:
    env_type = ENV_DICT["-d"]
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
planner = Plan(plan_steps, DT, env.get_wrapper_attr("AGENT_MAX_VEL"))


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
for i in tqdm(range(40000)):
    obs = obs_handler(obs)
    plan = planner.plan(obs)
    None if mpc_type == "simple" else env.get_wrapper_attr("set_trajectory")(
        *planner.prepare_plot(plan, plan_steps)
    )
    env.get_wrapper_attr("set_separating_planes")() if "Crowd" in env_type else None
    control_plan = mpc.get_action(plan, obs)
    obs, reward, terminated, truncated, info = env.step(control_plan[0])
    ep_return += reward
    env.render() if render else None
    if terminated or truncated:
        obs = env.reset()
        returns.append(ep_return)
        ep_return = 0
print("Mean: ", np.mean(returns))
