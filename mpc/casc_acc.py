import numpy as np
import scipy

from mpc.acc import MPCAcc


class MPCCascAcc(MPCAcc):
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
        plan_length: int = 20,
        uncertainty: str = "",
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
        )
        self.M = plan_length
        self.plan_horizon = self.M
        self.plan_type = plan_type

        self.casc_vec_pos_vel = np.hstack([
            np.hstack([np.arange(i, self.N + i) for i in range(1, self.M + 1)])
        ] * 2) * self.DT

        mat_pos_acc = self.mat_pos_acc[:self.N, :self.N]
        self.casc_mat_pos_acc = np.zeros((self.M * self.N, self.M * self.N))
        for i in range(self.M):
            self.casc_mat_pos_acc[
                i * self.N:i * self.N + self.N, i * self.N:i * self.N + self.N
            ] = mat_pos_acc
            for j in range(i):
                self.casc_mat_pos_acc[
                    i * self.N:(i + 1) * self.N, j * self.N
                ] = np.repeat(self.DT ** 2 * (2 * (i + 1 - j) - 1) / 2, self.N)
        self.casc_mat_pos_acc = np.stack([
            np.hstack([self.casc_mat_pos_acc, self.casc_mat_pos_acc * 0]),
            np.hstack([self.casc_mat_pos_acc * 0, self.casc_mat_pos_acc])
        ]).reshape(2 * self.M * self.N, 2 * self.M * self.N)

        mat_vel_acc = self.mat_vel_acc[:self.N, :self.N]
        self.casc_mat_vel_acc = np.zeros((self.M * self.N, self.M * self.N))
        for i in range(self.M):
            self.casc_mat_vel_acc[
                i * self.N:i * self.N + self.N, i * self.N:i * self.N + self.N
            ] = mat_vel_acc
            for j in range(i):
                self.casc_mat_vel_acc[
                    i * self.N:(i + 1) * self.N, j * self.N
                ] = np.repeat(self.DT * 1, self.N)
        self.casc_mat_vel_acc = np.stack([
            np.hstack([self.casc_mat_vel_acc, self.casc_mat_vel_acc * 0]),
            np.hstack([self.casc_mat_vel_acc * 0, self.casc_mat_vel_acc])
        ]).reshape(2 * self.M * self.N, 2 * self.M * self.N)

        filter_plan = np.zeros(self.N, dtype=int)
        filter_plan[0] = 1
        filter_plan = np.hstack([np.hstack([filter_plan] * self.M)] * 2)
        filter_terminal_break = np.zeros(self.N, dtype=int)
        filter_terminal_break[-1] = 1
        filter_terminal_break = np.hstack(
            [np.hstack([filter_terminal_break] * self.M)] * 2
        )

        self.casc_vec_pos_vel_plan = self.casc_vec_pos_vel[np.nonzero(filter_plan)]
        self.casc_mat_pos_acc_plan = self.casc_mat_pos_acc[np.nonzero(filter_plan)]
        self.casc_mat_vel_acc_plan = self.casc_mat_vel_acc[np.nonzero(filter_plan)]
        self.casc_mat_vel_acc_break = \
            self.casc_mat_vel_acc[np.nonzero(filter_terminal_break)]


        if self.plan_type == "Position":
            self.stability_coeff = 0.075
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_pos_acc_plan.T @ self.casc_mat_pos_acc_plan +
                self.stability_coeff * self.casc_mat_vel_acc_plan.T @ \
                self.casc_mat_vel_acc_plan
            )
            self.vec_p = lambda _1, plan, _2, vel: (
                -plan + self.casc_vec_pos_vel_plan * np.repeat(vel, self.M)
            ).T @ self.casc_mat_pos_acc_plan + \
                self.stability_coeff * np.repeat(vel, self.M) @ self.casc_mat_vel_acc_plan
        else:
            raise NotImplementedError

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const(self.N * self.M)
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N * self.M)


    def gen_vel_const(self, horizon):
        M_v_, b_v_, sgn_vel = self.gen_vel_param(horizon)


        def vel_vec_const(vel, idxs=None):
            idxs = np.arange(len(sgn_vel)) if idxs is None else idxs
            return sgn_vel[idxs] * (b_v_[idxs] - M_v_[idxs] @ np.repeat(vel, horizon))


        def vel_mat_const(idxs):
            return ((M_v_ @ self.casc_mat_vel_acc).T * sgn_vel).T[idxs]

        return vel_mat_const, vel_vec_const


    def gen_crowd_const(self, const_M, const_b, crowd_poss, vel):
        for member in range(self.n_crowd):
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, vel)
            if ignore:
                continue
            mat_crowd = np.hstack([
                np.eye(self.N * self.M) * vec[:, 0], np.eye(self.N * self.M) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + self.casc_vec_pos_vel *
                np.repeat(vel, self.N * self.M)
            ) - np.array([self.CONST_DIST_CROWD] * self.N * self.M)
            mat_crowd_control = -mat_crowd @ self.casc_mat_pos_acc
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def find_relevant_idxs(self, vel):
        idxs = self.relevant_idxs(vel)
        idxs = np.hstack(list(idxs) * self.N * self.M) + np.repeat(
            np.arange(0, self.N * self.M * self.circle_lin_sides, self.circle_lin_sides),
            3
        )
        return np.array(idxs, dtype=int)


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([
                np.eye(self.N * self.M) * line[0], np.eye(self.N * self.M) * line[1]
            ])
            limit = -mat_line @ (self.casc_vec_pos_vel * np.repeat(vel, self.N * self.M))\
                - line[2]

            const_M.append(-mat_line @ self.casc_mat_pos_acc)
            const_b.append(-limit)


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        crowd_vels.resize(self.n_crowd, 2) if crowd_vels is not None else None
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels
        horizon_crowd_poss = np.stack([crowd_poss] * (self.N + self.M)) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * (self.N + self.M), 0) * self.DT,
            np.arange(1, (self.N + self.M + 1))
        )
        return self.cascade_crowd_positions(horizon_crowd_poss)


    def cascade_crowd_positions(self, crowd_poss):
        """
        Take crowd positions and cascade them meaning from [1,..., M + N] converrt to
        [1, 2,.., N, 2,..., N + 1, 3,..., M, M + 2,..., M + N].

        Args:
            crowd_poss (numpy.ndarray): an array of size (n_crowd, 2) with the current
                positions of each member
        Return:
            (numpy.ndarray): the predicted positions of the crowd throughout the horizon
        """
        casc_crowd_poss = np.zeros((self.M * self.N,) + crowd_poss.shape[1:])
        for i in range(self.M):
            casc_crowd_poss[i * self.N:(i + 1) * self.N, :, :] =\
                crowd_poss[i:i + self.N, :, :]
        return casc_crowd_poss


    def terminal_const(self, vel):
        return self.casc_mat_vel_acc_break, -np.repeat(vel, self.M)


    def __call__(self, plan, obs):
        acc = self.core_mpc(plan, obs)
        if acc is None:
            # print("Executing last computed braking trajectory!")
            acc = np.zeros(2 * self.N)
            acc[0:self.N - 1] = self.last_planned_traj[1:, 0]
            acc[self.N:2 * self.N - 1] = self.last_planned_traj[1:, 1]
        else:
            acc = np.hstack([  # only next braking trajecotry is relevant
                acc[:self.N], acc[self.M * self.N:self.M * self.N + self.N]
            ])
        action = np.array([acc[:self.N], acc[self.N:]]).T
        self.last_planned_traj = action
        return action
