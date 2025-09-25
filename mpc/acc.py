from typing import Union
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
        crowd_max_vel: float,
        crowd_max_acc: float,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        stability_coeff: float = 0.3,
        horizon_tries: int = 0,
        relax_uncertainty: float = 1.,
    ):
        """
        Args:
            stability_coeff: stabilizes the MPC in order to have smoother control by
                penalizing jerky behaviour (big changes in control)
        """
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
            relax_uncertainty=relax_uncertainty,
        )
        self.stability_coeff = stability_coeff

        # (vec)tor to project initial (vel)ocity to future (pos)itions
        self.vec_pos_vel = np.hstack([np.arange(1, self.N + 1)] * 2) * self.DT

        # (mat)rix to project control (acc)eleration to future (pos)itions
        self.mat_pos_acc = scipy.linalg.toeplitz(
            np.array([(2 * i - 1) / 2 * self.DT ** 2 for i in range(1, self.N + 1)]),
            np.zeros(self.N)
        )
        self.mat_pos_acc = np.stack([
            np.hstack([self.mat_pos_acc, self.mat_pos_acc * 0]),
            np.hstack([self.mat_pos_acc * 0, self.mat_pos_acc])
        ]).reshape(2 * self.N, 2 * self.N)

        # (mat)rix to project control (acc)eleration to future (vel)ocities
        self.mat_vel_acc = self.DT * scipy.linalg.toeplitz(
            np.ones(self.N), np.zeros(self.N)
        )
        self.mat_vel_acc = np.stack([
            np.hstack([self.mat_vel_acc, self.mat_vel_acc * 0]),
            np.hstack([self.mat_vel_acc * 0, self.mat_vel_acc])
        ]).reshape(2 * self.N, 2 * self.N)

        # Q matrix in the quadratic objective
        self.mat_Q = scipy.sparse.csc_matrix(
            self.mat_pos_acc.T @ self.mat_pos_acc +
            self.stability_coeff * self.mat_vel_acc.T @ self.mat_vel_acc
        )

        # p vector (linear term) in the quadratic objective
        self.vec_p = lambda goal, _1, _2, vel: (
            (-np.repeat(goal, self.N) + self.vec_pos_vel * np.repeat(vel, self.N)).T @
            self.mat_pos_acc + self.stability_coeff * np.repeat(vel, self.N) @
            self.mat_vel_acc
        )

        # (mat)rix and (vec)tor to represent the velocity constraints
        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const(self.N)
        # (mat)rix and (vec)tor to represent the acceleration constraints
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N)


    def gen_vel_const(self, horizon):
        """
        Genreate the velocity constraints (see Section 1.6 in theory/mpc)
        """
        M_v_, b_v_, sgn_vel = super().gen_vel_param(horizon)


        def vel_vec_const(vel, idxs=None):
            idxs = np.arange(len(sgn_vel)) if idxs is None else idxs
            return sgn_vel[idxs] * (b_v_[idxs] - M_v_[idxs] @ np.repeat(vel, horizon))


        def vel_mat_const(idxs):
            return ((M_v_ @ self.mat_vel_acc).T * sgn_vel).T[idxs].squeeze()

        return vel_mat_const, vel_vec_const


    def gen_acc_const(self, horizon):
        """
        Genreate the acceleration constraints (see Section 1.6 in theory/mpc)
        """
        M_a_, b_a_, sgn_acc = super().gen_acc_param(horizon)


        def vec_const(_):
            return sgn_acc * b_a_
        return (M_a_.T * sgn_acc).T, vec_const


    def gen_crowd_const(self, const_M, const_b, crowd_poss, vel, crowd_vels=None):
        """
        Generate crowd constraints (see Section 1.5 in theory/mpc)
        """
        for i, member in enumerate(range(crowd_poss.shape[1])):
            # if considering uncertainty update the distance to the crowd
            if hasattr(self, "member_indeces"):
                idx = np.where(i < self.member_indeces)[0][0]
                dist_to_keep = self.CONST_DIST_CROWD[idx]
            else:
                dist_to_keep = self.CONST_DIST_CROWD

            # ignore if two far and different direction
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, vel)
            if ignore:
                continue

            # constraint formula
            mat_crowd = np.hstack([
                np.eye(self.N) * vec[:, 0], np.eye(self.N) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + self.vec_pos_vel * np.repeat(vel, self.N)
            ) - np.array([dist_to_keep] * self.N)
            mat_crowd_control = -mat_crowd @ self.mat_pos_acc

            # update/add constraints
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def find_relevant_idxs(self, vel):
        """
        Shape relevant indexes according to problem. The purpose of this is to avoid
        constraints that are already covered by other constraints for vel and acc.
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
        """
        Run mpc and if the results is None use the last computed braking trajectory.
        """
        acc = self.core_mpc(plan, obs)
        braking = acc is None
        if braking:
            # print("Executing last computed braking trajectory!")
            acc = np.zeros(2 * self.N)
            acc[0:self.N - 1] = self.last_planned_traj[1:, 0]
            acc[self.N:2 * self.N - 1] = self.last_planned_traj[1:, 1]

        action = np.array([acc[:self.N], acc[self.N:]]).T
        self.last_planned_traj = action.copy()
        return action, braking
