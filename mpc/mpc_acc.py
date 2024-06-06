import numpy as np
import scipy

from mpc.abstract_mpc import AbstractMPC


class MPCAcc(AbstractMPC):
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

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const()
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const()
        self.last_planned_traj = np.zeros((self.N, 2))


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


    def gen_crowd_const(self, const_M, const_b, crowd_poss, agent_vel):
        for member in range(self.n_crowd):
            poss = crowd_poss[:, member, :]
            dist = np.linalg.norm(poss, axis=-1)
            vec = -(poss.T / np.linalg.norm(poss, axis=-1)).T
            angle = np.arccos(np.clip(np.dot(-vec, agent_vel), -1, 1)) > np.pi / 4
            if (np.all(dist > self.MAX_DIST_STOP_CROWD) or
               (np.all(dist > self.MAX_DIST_STOP_CROWD / 2) and np.all(angle))):
                continue
            mat_crowd = np.hstack([
                np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + self.vec_pos_vel * np.repeat(agent_vel, self.N)
            ) - np.array([4 * self.PHYSICAL_SPACE] * self.N)
            mat_crowd_control = -mat_crowd @ self.mat_pos_acc
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        crowd_vels.resize(self.n_crowd, 2) if crowd_vels is not None else None
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels
        return np.stack([crowd_poss] * self.N) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N, 0) * self.DT,
            np.arange(0, self.N)
        )


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
        acc = super().__call__(plan, obs)
        if acc is None:
            print("Executing last computed braking trajectory!")
            acc = np.zeros(2 * self.N)
            acc[0:self.N - 1] = self.last_planned_traj[1:, 0]
            acc[self.N:2 * self.N - 1] = self.last_planned_traj[1:, 1]

        action = np.array([acc[:self.N], acc[self.N:]]).T
        self.last_planned_traj = action
        return action
