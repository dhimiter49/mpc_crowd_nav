import numpy as np
from plan import Plan


class ConstController:
    """
    Constant controller finds the next control input given a current and goal position and
    the time interval for which to go from the current position to the goal.
    """
    def __init__(self, dt: float, planner: Plan):
        self.DT = dt
        self.current_time = 0.
        self.current_pos = None
        self.planner = planner
        self.last_planned_traj = None


    def get_action(self, *args):
        pos_time_plan = self.planner.time_path
        dist = pos_time_plan[1:, :2] - pos_time_plan[:-1, :2]
        vel = dist / self.DT
        vel = np.concatenate([vel, np.zeros((1, 2))])

        return vel, False


    def reset(self):
        self.current_time = 0.
