import numpy as np
from qpsolvers import solve_qp
from mpc.mpc import MPC
import scipy


class MPCLinear(MPC):
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
        plan_type: str = "Position",
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            agent_max_vel,
            agent_max_acc,
            n_crowd,
        )
        self.plan_type = plan_type

        if self.plan_type == "PositionVelocity":
            self.stability_coeff = 0.26
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
            self.vec_p = lambda _1, plan, _2, vel: (
                (-plan + self.vec_pos_vel * np.repeat(vel, self.N)).T @
                self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
                self.mat_vel_acc
            )
        elif self.plan_type == "Velocity":
            self.mat_Q = scipy.sparse.csc_matrix(self.mat_vel_acc.T @ self.mat_vel_acc)
            self.vec_p = lambda _1, _2, vel_plan, vel: (-vel_plan).T @ self.mat_vel_acc


    def __call__(self, plan, obs):
        return super().__call__(plan, obs)
