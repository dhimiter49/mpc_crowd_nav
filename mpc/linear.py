from typing import Union
import numpy as np
from mpc.acc import MPCAcc
import scipy


class MPCLinear(MPCAcc):
    """
    Extended acceleration control MPC to use (linear) plans.
    """
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        const_dist_crowd: float,
        agent_max_vel: float,
        agent_max_acc: float,
        crowd_max_vel: float,
        crowd_max_acc: float,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        stability_coeff: float = 0.25,
        plan_type: str = "Position",
        horizon_tries: int = 0,
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            const_dist_crowd,
            agent_max_vel,
            agent_max_acc,
            crowd_max_vel,
            crowd_max_acc,
            uncertainty,
            radius_crowd,
            horizon_tries=horizon_tries,
        )
        self.plan_type = plan_type
        self.stability_coeff = stability_coeff

        if self.plan_type == "PositionVelocity":
            self.vel_coeff = 0.2
            self.mat_Q = scipy.sparse.csc_matrix(
                self.mat_pos_acc.T @ self.mat_pos_acc +
                self.vel_coeff * self.mat_vel_acc.T @ self.mat_vel_acc +
                self.stability_coeff * np.eye(2 * self.N)
            )
            self.vec_p = lambda _, plan, vel_plan, vel: (
                (-plan + self.vec_pos_vel * np.repeat(vel, self.N)).T @
                self.mat_pos_acc + (-vel_plan).T  @ self.mat_vel_acc
            )
        elif self.plan_type == "Position":
            self.stability_coeff = 0.25
            self.mat_Q = scipy.sparse.csc_matrix(
                self.mat_pos_acc.T @ self.mat_pos_acc +
                self.stability_coeff * self.mat_vel_acc.T @ self.mat_vel_acc
            )
            self.vec_p = lambda __1__, plan, __2__, vel: (
                (-plan + self.vec_pos_vel * np.repeat(vel, self.N)).T @
                self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
                self.mat_vel_acc
            )
        elif self.plan_type == "Velocity":
            self.mat_Q = scipy.sparse.csc_matrix(self.mat_vel_acc.T @ self.mat_vel_acc)
            self.vec_p = lambda __1__, __2__, vel_plan, __3__: \
                (-vel_plan).T @ self.mat_vel_acc


    def __call__(self, **kwargs):
        super().__call__(**kwargs)
