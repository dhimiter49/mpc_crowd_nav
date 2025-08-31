import sys
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import fancy_gym
import gymnasium as gym
import subprocess
import multiprocessing as mp


from mpc.factory import get_mpc
from plan import Plan
from obs_handler import ObsHandler
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


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
    "-mcs": "CrowdNavigation",
    "-mco": "CrowdNavigation",
    "-mci": "CrowdNavigationInter",
}

PLAN_DICT = {
    "-lp": "Position",
    "-lpv": "PositionVelocity",
    "-vp": "Velocity",
}


result = subprocess.run(["git", "diff"], capture_output=True, text=True)
gen_data = "-gd" in sys.argv

velocity_str = "Vel" if "-v" in sys.argv else ""
env_str = ""
crowd_shift_idx = 0
exp_name = "mpc" if "-n" not in sys.argv else sys.argv[sys.argv.index("-n") + 1]
if "-c" in sys.argv:
    env_type = ENV_DICT["-c"]
    env_str = "CrowdNavigationStatic%s-v0" % velocity_str
elif "-mc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigation%s-v0" % velocity_str
elif "-mcc" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigationConst%s-v0" % velocity_str
elif "-mcs" in sys.argv:
    env_type = ENV_DICT["-mcs"]
    env_str = "CrowdNavigationSFM%s-v0" % velocity_str
elif "-mco" in sys.argv:
    env_type = ENV_DICT["-mco"]
    env_str = "CrowdNavigationORCA%s-v0" % velocity_str
elif "-mci" in sys.argv:
    env_type = ENV_DICT["-mci"]
    env_str = "CrowdNavigationInter%s-v0" % velocity_str
    crowd_shift_idx = 1
else:
    env_type = ENV_DICT["-d"]
    env_str = "Navigation%s-v0" % velocity_str
env = gym.make("fancy/" + env_str)
print("Observation space: ", env.observation_space.shape)
print("Acion space: ", env.action_space.shape)
obs_handler = ObsHandler(env_type, env.get_wrapper_attr("n_crowd"))
render = "-nr" not in sys.argv

N = 21 if "-ss" not in sys.argv else int(sys.argv[sys.argv.index("-ss") + 1])
M = 20 if "-ps" not in sys.argv else int(sys.argv[sys.argv.index("-ps") + 1])
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
if "-sc" in sys.argv:
    mpc_kwargs["stability_coeff"] = float(sys.argv[sys.argv.index("-sc") + 1])

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

if "-u" in sys.argv:
    mpc_kwargs["uncertainty"] = sys.argv[sys.argv.index("-u") + 1]
if "-r" in sys.argv:
    mpc_kwargs["relax_uncertainty"] = float(sys.argv[sys.argv.index("-r") + 1])

mpc_kwargs["horizon_tries"] = 0
if "-ht" in sys.argv:
    mpc_kwargs["horizon_tries"] = int(sys.argv[sys.argv.index("-ht") + 1])

n_agents = env.get_wrapper_attr("n_crowd") if "-mci" in sys.argv else 1
planner = Plan(plan_steps, DT, env.get_wrapper_attr("AGENT_MAX_VEL"))

mpc = [
    get_mpc(
        mpc_type,
        horizon=N,
        dt=DT,
        physical_space=env.get_wrapper_attr("PHYSICAL_SPACE")[crowd_shift_idx:][i],
        # add 0.01 in order to account for continuous collision avoidance
        const_dist_crowd=env.get_wrapper_attr("PHYSICAL_SPACE")[0] * 2 + 0.01001,
        radius_crowd=np.delete(
            env.get_wrapper_attr("PHYSICAL_SPACE")[crowd_shift_idx:], i
        ),
        agent_max_vel=env.get_wrapper_attr("AGENT_MAX_VEL"),
        agent_max_acc=env.get_wrapper_attr("MAX_ACC"),
        crowd_max_vel=env.get_wrapper_attr("CROWD_MAX_VEL"),
        crowd_max_acc=env.get_wrapper_attr("MAX_ACC"),
        n_crowd=env.get_wrapper_attr("n_crowd") if n_agents == 1 else
        env.get_wrapper_attr("n_crowd") - 1,
        **mpc_kwargs
    ) for i in range(n_agents)
]

steps = 1000 if not env.get_wrapper_attr("run_test_case") else 500
obs = env.reset()
plan = np.zeros((N, 2))
returns, ep_return, vels, action = [], 0, [], np.array([0, 0])
ep_count = 0
ep_step_count = 0
dataset = np.empty((
    steps,
    np.sum(env.observation_space.shape) * 2 + np.sum(env.action_space.shape) + 1 + 1 + 1
))


def mpc_get_action(mpc, plan, obs, q):
    control_plan, breaking_flag = mpc.get_action(plan, obs)
    q.put((control_plan, breaking_flag))


step_count = 0
progress_bar = tqdm(total=steps, desc="Processing")
count = step_count if gen_data else ep_count
old_breaking_flags = None
breaking_steps = np.array([200] * n_agents)  # 200 is too high, no breaking traj
tot_breaking_steps = 0
while count < steps:
    old_obs = obs[0].copy() if isinstance(obs, tuple) else obs.copy()
    obs = obs_handler(obs)
    if n_agents > 1:
        plan = []
        crowd_poss = env.get_wrapper_attr("_crowd_poss")
        for i, _obs in enumerate(obs):
            plan.append(planner.plan(_obs))
            mpc[i].current_pos = crowd_poss[i]
    else:
        plan = planner.plan(obs)
        mpc[0].current_pos = env.get_wrapper_attr("_agent_pos")
    # None if mpc_type == "simple" else env.get_wrapper_attr("set_trajectory")(
    #     *planner.prepare_plot(plan, plan_steps)
    # )
    # env.get_wrapper_attr("set_separating_planes")() if "Crowd" in env_type else None
    # env.get_wrapper_attr("set_casc_trajectory")(all_future_pos)
    breaking_flags = np.array([False] * n_agents)
    if n_agents > 1:
        actions = []
        output, processes, queues = [], [], []
        # for i, (_plan, _obs) in enumerate(zip(plan, obs)):
        #     control, breaking = mpc[i].get_action(_plan, _obs)
        #     actions.append(control[0])
        # actions = np.array(actions).flatten()
        for i, (_plan, _obs) in enumerate(zip(plan, obs)):
            q = mp.Queue()
            p = mp.Process(target=mpc_get_action, args=(mpc[i], _plan, _obs, q))
            processes.append(p)
            queues.append(q)
            p.start()
        for q in queues:
            results = q.get()
            output += results
        for p in processes:
            p.join()
        for i in range(n_agents):
            control_plan = output[i * 2]
            breaking_flag = output[i * 2 + 1]
            mpc[i].last_planned_traj = control_plan
            action = control_plan[0]
            actions.append(action)
            breaking_flags[i] = breaking_flag
            if old_breaking_flags is not None:  # at least second step
                if not old_breaking_flags[i] and breaking_flag:
                    # before no breaking now breaking
                    breaking_steps[i] = ep_step_count
                if old_breaking_flags[i] and not breaking_flag:
                    breaking_steps[i] *= -1  # (-) meaning that there was but not anymore
        actions = np.array(actions).flatten()
    else:
        control_plan, breaking_flag = mpc[0].get_action(plan, obs)
        actions = control_plan[0]  # only one agent so only one action
        breaking_flags[0] = breaking_flag
        if old_breaking_flags is not None:
            tot_breaking_steps += 1 if breaking_flag and not old_breaking_flags[0] else 0
    step_count += 1
    env.render() if render else None
    obs, reward, terminated, truncated, info = env.step(actions)
    if gen_data:
        dataset[step_count] = np.hstack([
            old_obs.flatten(),
            obs.flatten(),
            actions.flatten(),
            np.array(reward),
            np.array(terminated),
            np.array(truncated)
        ])
    ep_return += reward
    ep_step_count += 1
    old_breaking_flags = breaking_flags
    if terminated or truncated:
        # print("Breaking flags: ", breaking_flags)
        # print("Breaking steps: ", breaking_steps)
        breaking_steps = np.array([200] * n_agents)  # 200 is too high, no breaking traj
        old_breaking_flags = None
        env.render() if render else None
        if not(ep_count == steps - 1 and env.get_wrapper_attr("run_test_case")):
            obs = env.reset()
            for i in range(n_agents):
                mpc[i].reset()
        returns.append(ep_return)
        ep_return = 0
        ep_count += 1
        ep_step_count = 0
        progress_bar.update(1) if not gen_data else None
    progress_bar.update(1) if gen_data else None
    count = step_count if gen_data else ep_count
if gen_data:
    np.save("dataset_" + env_str + ".npy", dataset)
print("Mean: ", np.mean(returns))
print("Number of episodes", ep_count)
print("Diffs: ", result.stdout)
print("Total breaking instances: ", tot_breaking_steps)
print("Stats:")
(
    col_rate,
    col_speed,
    col_agent_speed,
    avg_intersect_area,
    avg_intersect_area_percent,
    avg_col_severity,
    freezing_instances,
    avg_ttg,
    success_rate
) = env.stats()
exp_name = exp_name + ".csv"
path = Path.home() / "Documents" / "RAM" / "results" / exp_name
has_header = False
if path.is_file():
    with open(path, 'r', newline='') as csvfile:
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(csvfile.read(2048))
with open(path, 'a', newline='') as csvfile:
    fieldnames = [
        'return', 'ttg', 'success_rate',
        'col_rate', 'col_speed', 'col_agent_speed',
        'col_intersection_area', 'col_intersection_percent',
        'col_severity_index', 'breaking_instances', 'freezing_instances'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not has_header:
        writer.writeheader()
    writer.writerow({
        "return": np.mean(returns),
        "ttg": avg_ttg,
        "success_rate": success_rate,
        "col_rate": col_rate,
        "col_speed": col_speed,
        "col_agent_speed": col_agent_speed,
        "col_intersection_area": avg_intersect_area,
        "col_intersection_percent": avg_intersect_area_percent,
        "col_severity_index": avg_col_severity,
        "breaking_instances": tot_breaking_steps,
        "freezing_instances": freezing_instances,
    })
