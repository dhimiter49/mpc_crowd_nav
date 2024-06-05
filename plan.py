import numpy as np


class Plan:
    def __init__(self, dt: float, horizon: int, max_vel: float):
        self.N = horizon
        self.self.DT = self.DT
        self.MAX_VEL = max_vel


    def plan(self, goal):
        return self.__call__(goal)


    def __call__(self, goal):
        steps = np.zeros((self.N, 2))
        dist = np.linalg.norm(goal)
        vels = np.stack([(self.MAX_VEL) / dist * goal] * self.N).reshape(self.N, 2)
        if self.MAX_VEL * self.DT > dist:
            oneD_steps = np.array([dist])
        else:
            oneD_steps = np.arange(
                self.MAX_VEL * self.DT, dist, 2 * self.MAX_VEL * self.DT
            )
        twoD_steps = np.array([i / dist * goal for i in oneD_steps])
        n_steps = min(self.N, len(oneD_steps))
        steps[:n_steps, :] = twoD_steps[:n_steps]
        steps[n_steps:, :] += goal
        # steps *= 0
        vels_steps = int(dist / (self.MAX_VEL * self.DT))
        vels[vels_steps:, :] = np.zeros(2)
        return np.hstack([steps[:, 0], steps[:, 1]]), np.hstack([vels[:, 0], vels[:, 1]])
