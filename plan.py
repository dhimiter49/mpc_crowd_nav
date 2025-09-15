import numpy as np


class Plan:
    def __init__(self, horizon: int, dt: float, max_vel: float):
        self.N = horizon
        self.DT = dt
        self.MAX_VEL = max_vel


    def plan(self, obs):
        return self.__call__(obs)


    def __call__(self, obs):
        goal, _, _, _, _, _ = obs
        steps = np.zeros((self.N, 2))
        dist = np.linalg.norm(goal)
        vels = np.stack([(self.MAX_VEL) / dist * goal] * self.N).reshape(self.N, 2)
        if self.MAX_VEL * self.DT >= dist:
            # go directly to goal
            oneD_steps = np.array([dist])
        else:
            # steps to goal, multplying max_vel by 2 seems to work better
            oneD_steps = np.arange(
                self.MAX_VEL * self.DT, dist, 2 * self.MAX_VEL * self.DT
            )
        twoD_steps = np.array([goal * i / dist for i in oneD_steps])
        n_steps = min(self.N, len(oneD_steps))
        steps[:n_steps, :] = twoD_steps[:n_steps]
        steps[n_steps:, :] += goal
        vels_steps = int(dist / (self.MAX_VEL * self.DT))
        vels[vels_steps:, :] = np.zeros(2)
        return np.hstack([steps[:, 0], steps[:, 1]]), np.hstack([vels[:, 0], vels[:, 1]])


    def prepare_plot(self, plan, N):
        pos_plan, vel_plan = plan
        steps = np.zeros((N, 2))
        steps_vel = np.zeros((N, 2))
        steps[:, 0] = pos_plan[:N]
        steps[:, 1] = pos_plan[N:]
        steps_vel[:, 0] = vel_plan[:N]
        steps_vel[:, 1] = vel_plan[N:]
        return np.concatenate([np.zeros((1, 2)), steps]), steps_vel


class Sample_Plan:
    def __init__(self, horizon: int, dt: float, max_vel: float):
        self.N = horizon
        self.DT = dt
        self.MAX_VEL = max_vel
        self.N_SAMPLES = 10000
        self.child_angles = np.array([-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4])
        self.rot_matrices = np.array(
            [
                np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                for angle in self.child_angles
            ]
        )


    def plan(self, obs):
        return self.__call__(obs)


    def __call__(self, obs):
        goal, crowd_poss, _, crowd_vels, _, _ = obs
        goal_dir = goal / np.linalg.norm(goal)
        inv_goal_dir = -goal_dir
        # generate points
        traj_time = int(np.linalg.norm(goal) // self.MAX_VEL + 1)
        traj_steps = -(-traj_time // self.DT)
        samples_from_origin = [[[0, 0]]]
        samples_from_goal = [[goal]]
        for _ in range(min(int(traj_steps // 2), self.N)):
            samples_from_origin.append([])
            for last_sample in samples_from_origin[-2]:
                samples_from_origin[-1].extend(
                    last_sample + self.rot_matrices @ goal_dir * self.MAX_VEL * self.DT
                )
        if self.N > traj_steps // 2:
            for _ in range(int(traj_steps // 2)):
                samples_from_goal.append([])
                for last_sample in samples_from_goal[-2]:
                    samples_from_goal[-1].extend(
                        last_sample + self.rot_matrices @ inv_goal_dir * self.MAX_VEL *
                        self.DT
                    )
        trajectories = []
        for i, last_sample in enumerate(samples_from_origin[-1]):
            traj = [
                e[int(i // 5 ** (self.N - j))]
                for j, e in enumerate(samples_from_origin[:-1])
            ]
            traj.append(last_sample)
            trajectories.append(traj)
        trajectories = np.array(trajectories)
        trajectories = trajectories[:, 1:, :]
        reward = self.objecive_function(trajectories, goal, crowd_poss, crowd_vels)

        pos_traj = trajectories[np.argmax(reward)].flatten()
        return pos_traj, np.zeros(pos_traj.shape)


    def objecive_function(self, trajectories, goal, crowd_poss, crowd_vels):
        dist_goal = np.sum(np.linalg.norm(trajectories, axis=-1), axis=-1)
        crowd_poss = crowd_poss.reshape(-1, 2)
        crowd_vels = crowd_vels.reshape(-1, 2)
        crowd_poss = np.stack([crowd_poss] * self.N) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N, 0) * self.DT,
            np.arange(1, self.N + 1)
        )
        crowd_poss = np.transpose(crowd_poss, (1, 0, 2))
        diff_crowd = np.repeat(trajectories, len(crowd_poss), axis=1).reshape(
            trajectories.shape[0], len(crowd_poss), self.N, 2
        ) - crowd_poss
        dist_crowd = 0.0 * np.sum(np.linalg.norm(diff_crowd, axis=-1), axis=(-1, -2))

        reward = dist_goal + dist_crowd
        return reward


    def prepare_plot(self, plan, N):
        pos_plan, vel_plan = plan
        steps = np.zeros((N, 2))
        steps_vel = np.zeros((N, 2))
        steps[:, 0] = pos_plan[:N]
        steps[:, 1] = pos_plan[N:]
        steps_vel[:, 0] = vel_plan[:N]
        steps_vel[:, 1] = vel_plan[N:]
        return np.concatenate([np.zeros((1, 2)), steps]), steps_vel
