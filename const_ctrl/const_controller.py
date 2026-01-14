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


    def get_action(self, *args):
        pos_time_plan = self.planner.rrt_path
        time_checkpoints = pos_time_plan[:, 2]
        last_index = np.where(self.current_time >= time_checkpoints)[0][-1]

        if last_index < len(pos_time_plan) - 1:
            time_interval = pos_time_plan[last_index + 1, 2] -\
                pos_time_plan[last_index, 2]
            vel = (pos_time_plan[last_index + 1, :2] - pos_time_plan[last_index, :2]) /\
                time_interval
        else:
            vel = np.zeros(2)
        input()
        self.current_time += self.DT
        return [vel], False


    def reset(self):
        self.current_time = 0.
