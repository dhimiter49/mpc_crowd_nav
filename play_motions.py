import sys
import numpy as np
import fancy_gym
import gymnasium as gym


motion_data_path = sys.argv[1]
motion_data = np.load(motion_data_path)


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
else:
    env_type = ENV_DICT["-d"]
    env_str = "Navigation%s-v0" % velocity_str

mult_plan = 1 if "-mp" not in sys.argv else int(sys.argv[sys.argv.index("-mp") + 1])
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
    if len(valid_idx) == 0:
        continue

    dist = dist[valid_idx]
    positions = np.array(positions)[valid_idx]
    all_actions = np.array(all_actions)[valid_idx]
    sorted_dist = np.argsort(dist)
    env.get_wrapper_attr("set_all_motions")(np.array(positions)[np.argsort(dist)])

    plan = motion_data[i + sorted_dist[0]][6 + n_crowd * 4:6 + n_crowd * 4 + horizon * 2]
    plan = np.array([plan[:len(plan) // 2], plan[len(plan) // 2:]]).T
    env.get_wrapper_attr("set_trajectory")(plan)
    actions = all_actions[sorted_dist[0]]
    for a in actions:
        env.render()
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break

    env.render()
    env.reset()

# env.close_video_recorder()
# env.close()
