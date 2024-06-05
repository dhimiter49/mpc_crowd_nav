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
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            agent_max_vel,
            agent_max_acc
        )
        self.stability_coeff = 0.25

        self.mat_Q = scipy.sparse.csc_matrix(
            self.mat_pos_acc.T @ self.mat_pos_acc +
            self.stability_coeff * self.mat_vel_acc.T @ self.mat_vel_acc
        )
        self.vec_p = lambda plan, vel: (
            (-plan + self.vec_pos_vel * np.repeat(vel, self.N)).T @
            self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
            self.mat_vel_acc
        )


    def __call__(self, plan, obs):
        pos_plan, _ = plan
        _, vel, walls = obs

        const_M, const_b = [], []
        wall_eqs = self.wall_eq(walls)
        if len(wall_eqs) != 0:
            self.lin_pos_constraint(const_M, const_b, wall_eqs, vel)
        idxs = self.relevant_idxs(vel)
        const_M.append(self.mat_acc_const)
        const_b.append(self.vec_acc_const)
        const_M.append(self.mat_vel_const[idxs])
        const_b.append(self.vec_vel_const(vel, idxs))

        # term_const_M, term_const_b = self.terminal_const(vel)

        acc = solve_qp(
            self.mat_Q, self.vec_p(pos_plan, vel),
            # lb=-acc_b, ub=acc_b,
            G=scipy.sparse.csc_matrix(np.vstack(const_M)), h=np.hstack(const_b),
            # A=term_const_M, b=term_const_b,
            solver="clarabel",
            tol_gap_abs=5e-5,
            tol_gap_rel=5e-5,
            tol_feas=1e-4,
            tol_infeas_abs=5e-5,
            tol_infeas_rel=5e-5,
            tol_ktratio=1e-4
        )
        return np.array([acc[:self.N], acc[self.N:]]).T
