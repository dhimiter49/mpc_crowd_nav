import numpy as np
from fast_3Drrt import Obstacle, RRTSpatioTemporal


class Plan:
    def __init__(self, horizon: int, dt: float, max_vel: float):
        self.N = horizon
        self.DT = dt
        self.MAX_VEL = max_vel


    def plan(self, goal, current_pos):
        return self.__call__(goal, current_pos)


    def __call__(self, obs, _):
        # for this simple plan we only need the relative goal position to the agent
        goal, _, _, _, _, _ = obs

        # steps representing the plan (trajectory)
        steps = np.zeros((self.N, 2))
        dist = np.linalg.norm(goal)

        # velocity steps representgin also the plan (trajectory)
        vels = np.stack([(self.MAX_VEL) / dist * goal] * self.N).reshape(self.N, 2)

        # find the steps in 1D based on the maximum velocity, if too close than use
        # directly the goal position
        if self.MAX_VEL * self.DT >= dist:
            # go directly to goal
            oneD_steps = np.array([dist])
        else:
            # steps to goal, multplying max_vel by 2 seems to work better
            oneD_steps = np.arange(
                self.MAX_VEL * self.DT, dist, 2 * self.MAX_VEL * self.DT
            )

        # change from 1D steps to 2D using the goal direction
        twoD_steps = np.array([goal * i / dist for i in oneD_steps])

        # project the steps overshotting the goal to the goal
        n_steps = min(self.N, len(oneD_steps))
        steps[:n_steps, :] = twoD_steps[:n_steps]
        steps[n_steps:, :] += goal
        vels_steps = int(dist / (self.MAX_VEL * self.DT))
        vels[vels_steps:, :] = np.zeros(2)
        return np.hstack([steps[:, 0], steps[:, 1]]), np.hstack([vels[:, 0], vels[:, 1]])


    def prepare_plot(self, plan, N):
        # plots the plan
        pos_plan, vel_plan = plan
        steps = np.zeros((N, 2))
        steps_vel = np.zeros((N, 2))
        steps[:, 0] = pos_plan[:N]
        steps[:, 1] = pos_plan[N:]
        steps_vel[:, 0] = vel_plan[:N]
        steps_vel[:, 1] = vel_plan[N:]
        return np.concatenate([np.zeros((1, 2)), steps]), steps_vel


    def reset(self):
        pass


class RRT_Plan(Plan):
    def __init__(self, horizon: int, dt: float, max_vel: float, const_ctrl: bool):
        super().__init__(horizon, dt, max_vel)
        self.path = None
        self.const_ctrl = const_ctrl
        self.step = 0


    def __call__(self, obs, current_pos):
        if self.path is None:
            goal, crowd_poss, _, crowd_vels, walls, radii = obs
            crowd_poss = crowd_poss.reshape(-1, 2)
            crowd_vels = crowd_vels.reshape(-1, 2)
            if radii is None:
                radii = [0.4] * len(crowd_poss)
            obstacles = []
            for pos, vel, radius in zip(crowd_poss, crowd_vels, radii):
                obstacles.append(Obstacle(
                    pos[0], pos[1],
                    0, self.N * self.DT,
                    vel[0], vel[1],
                    radius
                ))
            left, right, down, up = walls
            rectangles = [
                (-left, 0, 0, 0.01, 100),
                (right, 0, 0, 0.01, 100),
                (0, -down, 0, 100, 0.01),
                (0, up, 0, 100, 0.1),
            ]

            self.ref_pos = current_pos
            rrt = RRTSpatioTemporal(
                start=(0, 0, 0),
                goal=tuple(goal),
                obstacles=obstacles,
                rectangles=rectangles,
                t_range=(0, self.N * self.DT),
                robot_radius=0.42,
                v_max=self.MAX_VEL,
                dt=0.1,
                max_iter=40000,
                goal_bias=0.1,
                goal_tolerance_xy=0.5,
            )

            while True:
                if rrt.build():
                    path = rrt.get_path()
                    break
            path = np.array(path)
            self.rrt_path = path
            max_time = path[-1][2]

            sample_time = np.arange(0, max_time, self.DT)
            interpol_x = np.interp(sample_time, path[:, 2], path[:, 0])[1:]
            interpol_y = np.interp(sample_time, path[:, 2], path[:, 1])[1:]
            traj_len = len(interpol_x)
            if traj_len < self.N:
                interpol_x = np.concatenate([
                    interpol_x, np.repeat(interpol_x[-1], self.N - traj_len)
                ])
                interpol_y = np.concatenate([
                    interpol_y, np.repeat(interpol_y[-1], self.N - traj_len)
                ])
            self.path = np.concatenate([interpol_x[:self.N], interpol_y[:self.N]])
            path = self.path.copy()
        else:
            path = np.array([self.path[:self.N], self.path[self.N:]]).T + self.ref_pos
            # closest_index = np.argmin(np.linalg.norm(path - current_pos, axis=-1))
            # path = path[closest_index:]
            path = path[self.step:]
            traj_len = len(path)
            if traj_len < self.N:
                path = np.concatenate([path, np.stack([path[-1]] * (self.N - traj_len))])
            path -= current_pos
            path = path.flatten('F')
        self.step += 1
        return path, path * 0


    def reset(self):
        self.step = 0
        self.path = None
