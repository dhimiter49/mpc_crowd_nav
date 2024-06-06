import numpy as np
import scipy

from mpc.abstract_mpc import AbstractMPC


class MPCVel(AbstractMPC):
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            agent_max_vel,
            agent_max_acc,
            n_crowd,
        )
        self.stability_coeff = 0.5
