import numpy as np
from qpsolvers import solve_qp
import scipy

from mpc.utils import gen_polygon


class AbstractMPC:
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        agent_max_vel: float,
        agent_max_acc: float,
    ):
        self.N = horizon
        self.DT = dt
        self.PHYSICAL_SPACE = physical_space
        self.AGENT_MAX_VEL = agent_max_vel
        self.AGENT_MAX_ACC = agent_max_acc
        self.MAX_TIME_STOP = self.AGENT_MAX_VEL / self.AGENT_MAX_ACC
        self.MAX_DIST_STOP = self.MAX_TIME_STOP ** 2 * self.AGENT_MAX_ACC * 0.5
        self.MAX_DIST_STOP_CROWD = 2 * self.MAX_DIST_STOP

        self.circle_lin_sides = 8
        self.POLYGON_ACC_LINES = gen_polygon(self.AGENT_MAX_ACC, self.circle_lin_sides)
        self.POLYGON_VEL_LINES = gen_polygon(self.AGENT_MAX_VEL, self.circle_lin_sides)

        self.vel_coeff = 0.2
        self.stability_coeff = 0.25


    def get_action(self, plan, obs):
        return self.__call__(plan, obs)


    def __call__(self, plan, obs):
        raise NotImplementedError


    def wall_eq(self, wall_dist):
        """
        Reutrns the equation for all four walls knowing that the index are:
             2    To represent this in the format ax+by+c, a nd b are one of [0, 1, -1],
           1   0  e.g. for index 0 in the graph to the left b=0 and a=-1 while for index
             3    1, a=1 while c is the distance to the wall.
        """
        eqs = np.stack(
            [
                np.array([-1, 1, 0, 0]),
                np.array([0, 0, -1, 1]),
                wall_dist - 1.1 * self.PHYSICAL_SPACE
            ],
            axis=1
        )
        return eqs[wall_dist < self.MAX_DIST_STOP * 0.6]


    def relevant_idxs(self, vel):
        angle = np.arctan2(vel[1], vel[0])
        angle = 2 * np.pi + angle if angle < 0 else angle
        angle_idx = angle // (2 * np.pi / self.circle_lin_sides)
        idxs = [
            angle_idx,
            (angle_idx + 1) % self.circle_lin_sides,
            (angle_idx - 1) % self.circle_lin_sides
        ]
        idxs = np.hstack(list(idxs) * self.N) + np.repeat(
            np.arange(0, self.N * self.circle_lin_sides, self.circle_lin_sides), 3
        )
        return np.array(idxs, dtype=int)


class MPC(AbstractMPC):
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
        self.stability_coeff = 0.5

        self.vec_pos_vel = np.hstack([np.arange(1, self.N + 1)] * 2) * self.DT

        self.mat_pos_acc = scipy.linalg.toeplitz(
            np.array([(2 * i - 1) / 2 * self.DT ** 2 for i in range(1, self.N + 1)]),
            np.zeros(self.N)
        )
        self.mat_pos_acc = np.stack([
            np.hstack([self.mat_pos_acc, self.mat_pos_acc * 0]),
            np.hstack([self.mat_pos_acc * 0, self.mat_pos_acc])
        ]).reshape(2 * self.N, 2 * self.N)

        self.mat_vel_acc = self.DT * scipy.linalg.toeplitz(
            np.ones(self.N), np.zeros(self.N)
        )
        self.mat_vel_acc = np.stack([
            np.hstack([self.mat_vel_acc, self.mat_vel_acc * 0]),
            np.hstack([self.mat_vel_acc * 0, self.mat_vel_acc])
        ]).reshape(2 * self.N, 2 * self.N)

        self.mat_Q = scipy.sparse.csc_matrix(
            self.mat_pos_acc.T @ self.mat_pos_acc +
            self.stability_coeff * self.mat_vel_acc.T @ self.mat_vel_acc
        )
        self.vec_p = lambda goal, vel: (
            (-np.repeat(goal, self.N) + self.vec_pos_vel * np.repeat(vel, self.N)).T @
            self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
            self.mat_vel_acc
        )

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const()
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const()


    def gen_vel_const(self):
        M_v_ = np.vstack([np.eye(self.N) * -line[0] for line in self.POLYGON_VEL_LINES])
        M_v_ = np.hstack(
            [M_v_, np.vstack([np.eye(self.N)] * len(self.POLYGON_VEL_LINES))]
        )
        sgn_vel = np.ones(len(self.POLYGON_VEL_LINES))
        sgn_vel[len(self.POLYGON_VEL_LINES) // 2:] = -1
        sgn_vel = np.repeat(sgn_vel, self.N)
        b_v_ = np.repeat(self.POLYGON_VEL_LINES[:, 1], self.N)


        def vel_vec_const(vel, idxs=None):
            idxs = np.arange(len(sgn_vel)) if idxs is None else idxs
            return sgn_vel[idxs] * (b_v_[idxs] - M_v_[idxs] @ np.repeat(vel, self.N))

        return ((M_v_ @ self.mat_vel_acc).T * sgn_vel).T, vel_vec_const


    def gen_acc_const(self):
        # acceleration/control constraint using the inner polygon of a circle with radius
        # AGENT_MAX_ACC
        M_a_ = np.vstack([np.eye(self.N) * -line[0] for line in self.POLYGON_ACC_LINES])
        M_a_ = np.hstack(
            [M_a_, np.vstack([np.eye(self.N)] * len(self.POLYGON_ACC_LINES))]
        )
        sgn_acc = np.ones(len(self.POLYGON_ACC_LINES))
        sgn_acc[len(self.POLYGON_ACC_LINES) // 2:] = -1
        sgn_acc = np.repeat(sgn_acc, self.N)
        b_a_ = np.repeat(self.POLYGON_ACC_LINES[:, 1], self.N)

        return (M_a_.T * sgn_acc).T, sgn_acc * b_a_


    def terminal_const(self, vel):
        return self.mat_vel_acc[[self.N - 1, 2 * self.N - 1], :], -vel


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([np.eye(self.N) * line[0], np.eye(self.N) * line[1]])
            limit = -mat_line @ (self.vec_pos_vel * np.repeat(vel, self.N)) - line[2]

            const_M.append(-mat_line @ self.mat_pos_acc)
            const_b.append(-limit)


    def __call__(self, _, obs):
        goal, vel, walls = obs

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
            self.mat_Q, self.vec_p(goal, vel),
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
