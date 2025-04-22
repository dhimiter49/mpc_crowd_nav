import numpy as np
import scipy

from mpc.abstract import AbstractMPC


class MPCVel(AbstractMPC):
    """
    Using velocity control. Directly implemens a (linear) planner.
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
        plan_type: str = "Position",
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            const_dist_crowd,
            agent_max_vel,
            agent_max_acc,
            n_crowd,
        )
        self.stability_coeff = 0.25
        self.avoid_crowd_coeff = 0.01
        self.plan_type = plan_type

        self.mat_pos_vel = scipy.linalg.toeplitz(
            np.ones(self.N), np.zeros(self.N)
        ) * self.DT
        np.fill_diagonal(self.mat_pos_vel, 1 / 2 * self.DT)
        self.mat_pos_vel = self.mat_pos_vel[:, :-1]  # last dim is explicitly zero, ignore
        self.mat_pos_vel = np.stack([
            np.hstack([self.mat_pos_vel, self.mat_pos_vel * 0]),
            np.hstack([self.mat_pos_vel * 0, self.mat_pos_vel])
        ]).reshape(2 * self.N, 2 * (self.N - 1))

        acc_from_vel = np.zeros(self.N)
        acc_from_vel[:2] = np.array([1, -1])
        self.mat_acc_vel = scipy.linalg.toeplitz(acc_from_vel, np.zeros(self.N)) / self.DT
        self.mat_acc_vel = self.mat_acc_vel[:, :-1]
        self.mat_acc_vel = np.stack([
            np.hstack([self.mat_acc_vel, self.mat_acc_vel * 0]),
            np.hstack([self.mat_acc_vel * 0, self.mat_acc_vel])
        ]).reshape(2 * self.N, 2 * (self.N - 1))


        if self.plan_type == "Position":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.mat_pos_vel.T @ self.mat_pos_vel +
                self.stability_coeff * np.eye(2 * (self.N - 1))
            )
            # self.mat_Q = scipy.sparse.csc_matrix(
            #     (1 - self.avoid_crowd_coff) * self.mat_pos_vel.T @ self.mat_pos_vel +
            #     self.stability_coeff * np.eye(2 * (self.N - 1))
            # )


            def vec_p(_1, plan, _2, vel, crowd=None):
                goal_obj = (-plan + 0.5 * self.DT * np.repeat(vel, self.N)).T @\
                    self.mat_pos_vel

                # far_crowd_obj = 0
                # for member in np.moveaxis(crowd, 0, 1):
                #     far_crowd_obj -= (
                #         -member.flatten('F') + 0.5 * self.DT * np.repeat(vel, self.N)
                #     ).T @ self.mat_pos_vel
                # return goal_obj + self.avoid_crowd_coeff * far_crowd_obj
                return goal_obj
        elif self.plan_type == "Velocity":
            self.mat_Q = scipy.sparse.csc_matrix(np.eye(2 * (self.N - 1)))


            def vec_p(_1, _2, plan_vels, vel):
                plan_vels[:self.N] += vel[0]
                plan_vels[self.N:] += vel[1]
                plan_vels = np.delete(plan_vels, [self.N - 1, 2 * self.N - 1])
                return -plan_vels.T
        elif self.plan_type == "PositionVelocity":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.mat_pos_vel.T @ self.mat_pos_vel +
                self.stability_coeff * np.eye(2 * (self.N - 1))
            )


            def vec_p(_, plan_pos, plan_vels, vel):
                plan_vels[:self.N] += vel[0]
                plan_vels[self.N:] += vel[1]
                plan_vels = np.delete(plan_vels, [self.N - 1, 2 * self.N - 1])
                return (-plan_pos + 0.5 * self.DT * np.repeat(vel, self.N)).T @ \
                    self.mat_pos_vel - self.stability_coeff * plan_vels.T
        self.vec_p = vec_p

        if type(self).__name__ == "MPCVel":
            self.mat_vel_const, self.vec_vel_const = self.gen_vel_const(self.N - 1)
            self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N)


    def gen_vel_const(self, horizon):
        M_v_, b_v_, sgn_vel = super().gen_vel_param(horizon)


        def mat_vel_const(idxs):
            return (M_v_.T * sgn_vel).T[idxs]


        def vec_vel_const(_, idxs):
            return (sgn_vel * b_v_)[idxs]


        return mat_vel_const, vec_vel_const


    def gen_acc_const(self, horizon):
        M_a_, b_a_, sgn_acc = super().gen_acc_param(horizon)


        def acc_vec_const(agent_vel):
            agent_vel_ = np.zeros(2 * (horizon))
            agent_vel_[0], agent_vel_[horizon] = agent_vel
            return sgn_acc * (b_a_ + M_a_ @ agent_vel_ / self.DT)

        return ((M_a_ @ self.mat_acc_vel).T * sgn_acc).T, acc_vec_const


    def gen_crowd_const(self, const_M, const_b, crowd_poss, agent_vel):
        for member in range(self.n_crowd):
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, agent_vel)
            if ignore:
                continue
            mat_crowd = np.hstack([
                np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + 0.5 * self.DT * np.repeat(agent_vel, self.N)
            ) - np.array([self.CONST_DIST_CROWD] * self.N)
            mat_crowd_control = -mat_crowd @ self.mat_pos_vel
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def terminal_const(self, vel):
        # Explicti conditioning of last step of control velocity
        return None, None


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([np.eye(self.N) * line[0], np.eye(self.N) * line[1]])
            limit = -mat_line @ (0.5 * self.DT * np.repeat(vel, self.N)) - line[2]

            const_M.append(-mat_line @ self.mat_pos_vel)
            const_b.append(-limit)


    def find_relevant_idxs(self, vel):
        idxs = super().relevant_idxs(vel)
        idxs = np.hstack(list(idxs) * (self.N - 1)) + np.repeat(
            np.arange(0, (self.N - 1) * self.circle_lin_sides, self.circle_lin_sides), 3
        )
        return np.array(idxs, dtype=int)


    def __call__(self, plan, obs):
        vel = self.core_mpc(plan, obs)
        if vel is None:
            # print("Executing last computed braking trajectory!")
            vel = self.last_planned_traj[1:].flatten("F")

        action = np.array([
            np.append(vel[:self.N - 1], 0), np.append(vel[self.N - 1:], 0)
        ]).T
        self.last_planned_traj = action
        return action
