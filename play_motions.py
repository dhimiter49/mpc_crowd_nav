import sys
import numpy as np
import fancy_gym
import gymnasium as gym


motion_data_path = sys.argv[1]
motion_data = np.load(motion_data_path)
# traj_viz = np.load("full_traj_20_100_og.npy")
traj_viz = None


indexes_for_thirty = []
all_solutions_thirty = np.array([])


ENV_DICT = {
    "-d": "Navigation",
    "-c": "CrowdNavigationStatic",
    "-mc": "CrowdNavigation",
    "-mcc": "CrowdNavigationConst",
    "-mcco": "CrowdNavigationConst",
    "-mccs": "CrowdNavigationConst",
    "-mcs": "CrowdNavigation",
    "-mco": "CrowdNavigation",
    "-mci": "CrowdNavigationInter",
}

velocity_str = "Vel" if "-v" or "-sqp" in sys.argv else ""
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
elif "-mccs" in sys.argv:
    env_type = ENV_DICT["-mc"]
    env_str = "CrowdNavigationConstOneWayShort%s-v0" % velocity_str
else:
    env_type = ENV_DICT["-d"]
    env_str = "Navigation%s-v0" % velocity_str

mult_plan = 1 if "-mp" not in sys.argv else int(sys.argv[sys.argv.index("-mp") + 1])
render = "-r" in sys.argv
plot_all_plans = "-ap" in sys.argv
calculate_best_time = "-bt" in sys.argv
best_time_out_of = mult_plan
if calculate_best_time:
    mult_plan = 1
env = gym.make("fancy/" + env_str)
DT = env.unwrapped.dt
# env = gym.wrappers.RecordVideo(
#     env=env,
#     video_folder="/home/dhimiter/Videos",
#     name_prefix="test-video",
#     episode_trigger=lambda x: x % 2 == 0
# )
env.reset()
n_crowd = env.unwrapped.n_crowd
# env.start_video_recorder()
plan_to_motion_time_distance = []
motion_time = []
all_solutions = []
successful_motion_time = []
motions_found = []
all_valid_idxs = []
motion_best_time = [[] for _ in range(len(motion_data) // best_time_out_of)]
n_rrt_paths = 0
solution_found = 0
n_all_zero_actions = 0

special_render = []

print("There are " + str(len(motion_data) // mult_plan) + " episodes.")
for i in range(0, len(motion_data), mult_plan):
    trajectory_data = motion_data[i]
    agent_pos = trajectory_data[:2]
    crowd_poss = trajectory_data[2:2 + n_crowd * 2]
    agent_vel = trajectory_data[2 + n_crowd * 2:4 + n_crowd * 2]
    crowd_vels = trajectory_data[4 + n_crowd * 2:4 + n_crowd * 4]
    goal_pos = trajectory_data[4 + n_crowd * 4:6 + n_crowd * 4]
    env.hard_set_vars(
        {
            "_agent_pos": agent_pos,
            "_agent_vel": agent_vel,
            "_crowd_poss": crowd_poss.reshape(n_crowd, 2),
            "_crowd_vels": crowd_vels.reshape(n_crowd, 2),
            "_goal_pos": goal_pos
        }
    )
    horizon = len(trajectory_data[6 + n_crowd * 4:]) // 4
    all_actions = [
        motion_data[i + j][6 + n_crowd * 4 + horizon * 2:].reshape(-1, 2)
        for j in range(mult_plan)
    ]
    n_all_zero_actions += 1 if np.all(np.array(all_actions) == 0) else 0
    all_plans = [
        motion_data[i + j][6 + n_crowd * 4:6 + n_crowd * 4 + horizon * 2]
        .reshape(-1, 2, order='F')
        for j in range(mult_plan)
    ]
    for p in all_plans:
        n_rrt_paths += 0 if np.all(p == 0) else 1
    positions = [
        np.concatenate([np.array([[0, 0]]), np.cumsum(a * DT, axis=0)])
        for a in all_actions
    ]
    dist = np.array([np.sum(np.linalg.norm(p[1:] - p[:-1], axis=-1)) for p in positions])
    non_stat_idx = np.where(dist > 1e-5)[0]
    to_goal_idx = np.where(
        np.linalg.norm(np.array(positions)[:, -1] + agent_pos - goal_pos, axis=-1) < 0.4
    )[0]
    valid_idx = list(set(non_stat_idx) & set(to_goal_idx))
    all_valid_idxs += list(np.array(valid_idx) + i)

    dist = dist[valid_idx]
    positions = np.array(positions)[valid_idx]
    all_valid_actions = np.array(all_actions)[valid_idx]
    all_valid_plans = np.array(all_plans)[valid_idx]
    solution_found += 1 if len(valid_idx) > 0 else 0
    # if len(valid_idx) == 0:
    #     print("aaaa", i)
    # n_motions_found.append(len(all_valid_plans))
    sorted_dist = np.argsort(dist)
    temp_pos = np.array(positions)[np.flip(sorted_dist)]
    idx = sorted_dist[0] if len(sorted_dist) > 0 else 0
    env.get_wrapper_attr("set_all_motions")(
        temp_pos[-1:]
    )

    best_motion_idx = sorted_dist[0] if len(sorted_dist) > 0 else 0
    plan = motion_data[i + best_motion_idx][6 + n_crowd * 4:6 + n_crowd * 4 + horizon * 2]
    plan = np.array([plan[:len(plan) // 2], plan[len(plan) // 2:]]).T
    if traj_viz is not None:

        env.get_wrapper_attr("set_casc_trajectory")(traj_viz[i + idx])
    if plot_all_plans:
        env.get_wrapper_attr("set_trajectory")(
            np.array(all_valid_plans)[np.flip(sorted_dist)]
        )
    else:
        env.get_wrapper_attr("set_trajectory")(plan)
    actions = all_valid_actions[best_motion_idx] if len(sorted_dist) > 0\
        else all_actions[best_motion_idx]
    p_time = np.where(np.linalg.norm(plan[1:] - plan[-1], axis=-1) == 0)[0][0] * 0.1
    a_time = 0
    for a in actions:
        a_time += 0.1
        env.render() if render or i in special_render else None
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break

    a_time = p_time if np.all(actions == 0) else a_time

    # print("Difference between action time and plan time: " + str(a_time - p_time) + "s")
    plan_to_motion_time_distance.append(a_time - p_time)
    all_solutions.append(a_time)
    if not np.all(actions == 0):
        motion_time.append(a_time)
        if len(sorted_dist) > 0:
            successful_motion_time.append(a_time)
        if calculate_best_time:
            motion_best_time[i // best_time_out_of].append(a_time)
    env.render() if render or i in special_render else None
    env.reset()
# print(plan_to_motion_time_distance)
print(np.mean(plan_to_motion_time_distance))
# print(np.std(plan_to_motion_time_distance))
# print(motion_time)
# print(all_solutions)

# time only for successful motion that get to the goal
# print(len(successful_motion_time))
# relevant_idx = list(set(indexes_for_thirty) & set(all_valid_idxs))
# print("Faster by", np.mean(
#     all_solutions_thirty[relevant_idx] - np.array(all_solutions)[relevant_idx]
# ))
# print(successful_motion_time)
# print(np.mean(successful_motion_time))

# print(motion_best_time)
# print(n_motions_found)
print("Number of valid plans", n_rrt_paths)
print("All zero actions, no qp solution: " + str(n_all_zero_actions))
print(
    "Solution found for: ", solution_found,
    " out of ", len(motion_data) // mult_plan, " episodes")
# print(np.mean(motion_time))
if calculate_best_time:
    print(np.mean([np.min(ls) for ls in motion_best_time if len(ls) > 0]))

# env.close_video_recorder()
# env.close()
