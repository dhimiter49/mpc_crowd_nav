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
import mpc.abstract as mpc_ab
from plan import Plan
from obs_handler import ObsHandler
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


MPC_DICT = {
    "-d": "simple",  # simple plan just minimizes all future distances to the goal
    "-lp": "linear_plan",  # straight line to goal with sampling distance based on max vel
    "-v": "velocity_control",  # linea_plan byt with velocity as control
    "-sqp": "sequential",  # sequential QP
    "-cs": "cascading",  # linear_plan for cascading MPC
    "-vcs": "velocity_control_cascading"  # linear_plan with vel control for cascading MPC
}

ENV_DICT = {
    "-d": "Navigation",
    "-c": "CrowdNavigationStatic",
    "-mc": "CrowdNavigation",
    "-mcc": "CrowdNavigationConst",
    "-mcco": "CrowdNavigationConst",
    "-mcs": "CrowdNavigation",
    "-mco": "CrowdNavigation",
    "-mci": "CrowdNavigationInter",
}

PLAN_DICT = {
    "-lp": "Position",  # Objective (based on reference plan) uses only positions
    "-lpv": "PositionVelocity",  # Objective uses positions and velocities
    "-vp": "Velocity",  # Objective uses only velocities
}


# result = subprocess.run(["git", "diff"], capture_output=True, text=True)

###############################  READING INPUT PARAMETERS  ###############################
gen_data = "-gd" in sys.argv  # option to generate data from MPC
velocity_str = "Vel" if "-v" or "-sqp" in sys.argv else ""
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
elif "-mcco" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigationConstOneWay%s-v0" % velocity_str
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


####################################  SETTING UP MPC #####################################
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
elif "-sqp" in sys.argv:
    mpc_type = MPC_DICT["-sqp"]
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

if "-ns" in sys.argv:
    mpc_kwargs["passive_safety"] = False

n_agents = env.get_wrapper_attr("n_crowd") if "-mci" in sys.argv else 1
planner = Plan(plan_steps, DT, env.get_wrapper_attr("AGENT_MAX_VEL"))

augment_radius = float(sys.argv[sys.argv.index("-ar") + 1]) if "-ar" in sys.argv else 1.

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
        ) * augment_radius,
        agent_max_vel=env.get_wrapper_attr("AGENT_MAX_VEL"),
        agent_max_acc=env.get_wrapper_attr("MAX_ACC"),
        crowd_max_vel=env.get_wrapper_attr("CROWD_MAX_VEL"),
        crowd_max_acc=env.get_wrapper_attr("MAX_ACC"),
        **mpc_kwargs
    ) for i in range(n_agents)
]


def mpc_get_action(mpc, plan, obs, q):
    # Function that just runs an MPC step, used as target for multiprocessing
    control_plan, braking_flag = mpc.get_action(plan, obs)
    q.put((control_plan, braking_flag))


#########################  ENVIRONMENT SETUP AND AUXILIARY VARS ##########################
steps = 1000 if not env.get_wrapper_attr("run_test_case") else 500
dataset = np.empty((
    steps,
    np.sum(env.observation_space.shape) * 2 + np.sum(env.action_space.shape) + 1 + 1 + 1
)) if gen_data else None
obs = env.reset()
plan = np.zeros((N, 2))
returns, ep_return, vels, action = [], 0, [], np.array([0, 0])
ep_count = 0
ep_step_count = 0
step_count = 0
tot_braking_steps = 0
count = step_count if gen_data else ep_count
old_braking_flags = None
braking_steps = np.array([200] * n_agents)  # 200 is too high, no braking traj
progress_bar = tqdm(total=steps, desc="Processing")

# Main loop
while count < steps:
    old_obs = obs[0].copy() if isinstance(obs, tuple) else obs.copy()
    obs = obs_handler(obs)  # change observation to necessary format for mpc

    # get plan(s)
    if n_agents > 1:
        plan = []
        crowd_poss = env.get_wrapper_attr("_crowd_poss")
        for i, _obs in enumerate(obs):
            plan.append(planner.plan(_obs))
            mpc[i].current_pos = crowd_poss[i]
    else:
        plan = planner.plan(obs)
        mpc[0].current_pos = env.get_wrapper_attr("_agent_pos")

    # Visualize robot trajecotry and the separating constraints between crowd and agent
    # None if mpc_type == "simple" else env.get_wrapper_attr("set_trajectory")(
    #     *planner.prepare_plot(plan, plan_steps)
    # )
    # env.get_wrapper_attr("set_separating_planes")() if "Crowd" in env_type else None
    # env.get_wrapper_attr("set_casc_trajectory")(all_future_pos)

    # Compute MPC nect action and if a braking trajectory is executed
    braking_flags = np.array([False] * n_agents)
    if n_agents > 1:
        actions = []
        output, processes, queues = [], [], []
        # for i, (_plan, _obs) in enumerate(zip(plan, obs)):
        #     control, braking = mpc[i].get_action(_plan, _obs)
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
            braking_flag = output[i * 2 + 1]
            mpc[i].last_planned_traj = control_plan
            action = control_plan[0]
            actions.append(action)
            braking_flags[i] = braking_flag
            if old_braking_flags is not None:  # at least second step
                if not old_braking_flags[i] and braking_flag:
                    # before no braking now braking
                    braking_steps[i] = ep_step_count
                if old_braking_flags[i] and not braking_flag:
                    braking_steps[i] *= -1  # (-) meaning that there was but not anymore
        actions = np.array(actions).flatten()
    else:
        control_plan, braking_flag = mpc[0].get_action(plan, obs)
        # traj = mpc[0].traj_from_plan(obs[2])
        # env.set_trajectory(traj)
        actions = control_plan[0]  # only one agent so only one action
        braking_flags[0] = braking_flag
        if old_braking_flags is not None:
            tot_braking_steps += 1 if braking_flag and not old_braking_flags[0] else 0

    # take step in the environment
    env.render() if render else None
    obs, reward, terminated, truncated, info = env.step(actions)

    # update auxiliary variables
    step_count += 1
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
    old_braking_flags = braking_flags
    if terminated or truncated:
        # print("braking flags: ", braking_flags)
        # print("braking steps: ", braking_steps)
        braking_steps = np.array([200] * n_agents)  # 200 is too high, no braking traj
        old_braking_flags = None
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

# Save data if generating data
if gen_data:
    np.save("dataset_" + env_str + ".npy", dataset)

# Print and save results
# print("Diffs: ", result.stdout)
# print("Constraints", mpc_ab.CONST_DIM / mpc_ab.CONST_STEPS)
print("Mean: ", np.mean(returns))
print("Number of episodes", ep_count)
print("Total braking instances: ", tot_braking_steps)
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
        'col_severity_index', 'braking_instances', 'freezing_instances'
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
        "braking_instances": tot_braking_steps,
        "freezing_instances": freezing_instances,
    })
