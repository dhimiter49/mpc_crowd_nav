import numpy as np
import scipy

from mpc.abstract import AbstractMPC


class MPCAcc(AbstractMPC):
    """
    Using acceleration control. The objective is implemented directly as the position of
    the goal.
    """
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        const_dist_crowd: float,
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        radius: Union[float, None] = None,
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            const_dist_crowd,
            agent_max_vel,
            agent_max_acc,
            n_crowd,
            uncertainty,
            radius_crowd,
            radius,
        )
        self.stability_coeff = 0.3

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
        self.vec_p = lambda goal, _1, _2, vel: (
            (-np.repeat(goal, self.N) + self.vec_pos_vel * np.repeat(vel, self.N)).T @
            self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
            self.mat_vel_acc
        )

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const(self.N)
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N)


    def gen_vel_const(self, horizon):
        M_v_, b_v_, sgn_vel = super().gen_vel_param(horizon)


        def vel_vec_const(vel, idxs=None):
            idxs = np.arange(len(sgn_vel)) if idxs is None else idxs
            return sgn_vel[idxs] * (b_v_[idxs] - M_v_[idxs] @ np.repeat(vel, horizon))


        def vel_mat_const(idxs):
            return ((M_v_ @ self.mat_vel_acc).T * sgn_vel).T[idxs]

        return vel_mat_const, vel_vec_const


    def gen_acc_const(self, horizon):
        M_a_, b_a_, sgn_acc = super().gen_acc_param(horizon)


        def vec_const(_):
            return sgn_acc * b_a_
        return (M_a_.T * sgn_acc).T, vec_const


    def gen_crowd_const(self, const_M, const_b, crowd_poss, vel):
        for i, member in enumerate(range(crowd_poss.shape[1])):
            if hasattr(self, "member_indeces"):
                idx = np.where(i < self.member_indeces)[0][0]
                dist_to_keep = self.CONST_DIST_CROWD[idx]
            else:
                dist_to_keep = self.CONST_DIST_CROWD
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, vel)
            if ignore:
                continue
            mat_crowd = np.hstack([
                np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + self.vec_pos_vel * np.repeat(vel, self.N)
            ) - np.array([dist_to_keep] * self.N)
            mat_crowd_control = -mat_crowd @ self.mat_pos_acc
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def find_relevant_idxs(self, vel):
        """
        Shape relevant indexes according to problem.
        """
        idxs = super().relevant_idxs(vel)
        idxs = np.hstack(list(idxs) * self.N) + np.repeat(
            np.arange(0, self.N * self.circle_lin_sides, self.circle_lin_sides), 3
        )
        return np.array(idxs, dtype=int)


    def terminal_const(self, vel):
        return self.mat_vel_acc[[self.N - 1, 2 * self.N - 1], :], -vel


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([np.eye(self.N) * line[0], np.eye(self.N) * line[1]])
            limit = -mat_line @ (self.vec_pos_vel * np.repeat(vel, self.N)) - line[2]

            const_M.append(-mat_line @ self.mat_pos_acc)
            const_b.append(-limit)


    def __call__(self, plan, obs):
        acc = self.core_mpc(plan, obs)
        breaking = acc is None
        if breaking:
            # print("Executing last computed braking trajectory!")
            acc = np.zeros(2 * self.N)
            acc[0:self.N - 1] = self.last_planned_traj[1:, 0]
            acc[self.N:2 * self.N - 1] = self.last_planned_traj[1:, 1]

        action = np.array([acc[:self.N], acc[self.N:]]).T
        self.last_planned_traj = action
        return action, breaking
