import numpy as np


class Plan:
    def __init__(self, horizon: int, dt: float, max_vel: float):
        self.N = horizon
        self.DT = dt
        self.MAX_VEL = max_vel


    def plan(self, goal):
        return self.__call__(goal)


    def __call__(self, obs):
        goal, _, _, _, _ = obs
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
