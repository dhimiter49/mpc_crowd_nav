from typing import Union
import numpy as np
import scipy

from mpc.vel import MPCVel


class MPCCascVel(MPCVel):
    def __init__(
        self,
        horizon: int,  # represent braking horizon
        dt: float,
        physical_space: float,
        const_dist_crowd: float,
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
        plan_type: str = "Position",
        plan_length: int = 20,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        stability_coeff: float = 0.2,
        horizon_tries: int = 0,
    ):
        super().__init__(
            horizon,
            dt,
            physical_space,
            const_dist_crowd,
            agent_max_vel,
            agent_max_acc,
            n_crowd,
            uncertainty=uncertainty,
            radius_crowd=radius_crowd,
            horizon_tries=horizon_tries,
        )
        self.M = plan_length
        self.plan_horizon = self.M
        self.plan_type = plan_type
        self.stability_coeff = stability_coeff

        mat_pos_vel = self.mat_pos_vel[:self.N, :self.N - 1]
        self.casc_mat_pos_vel = np.zeros((self.M * self.N, self.M * (self.N - 1)))
        for i in range(self.M):
            self.casc_mat_pos_vel[
                i * self.N:(i + 1) * self.N,
                i * (self.N - 1):i * (self.N - 1) + self.N - 1
            ] = mat_pos_vel
            for j in range(i):
                self.casc_mat_pos_vel[
                    i * self.N:(i + 1) * self.N, j * (self.N - 1)
                ] = np.ones(self.N) * self.DT
        self.casc_mat_pos_vel = np.stack([
            np.hstack([self.casc_mat_pos_vel, self.casc_mat_pos_vel * 0]),
            np.hstack([self.casc_mat_pos_vel * 0, self.casc_mat_pos_vel])
        ]).reshape(2 * self.M * self.N, 2 * self.M * (self.N - 1))

        mat_acc_vel = self.mat_acc_vel[:self.N, :self.N - 1]
        self.casc_mat_acc_vel = np.zeros((self.M * self.N, self.M * (self.N - 1)))
        for i in range(self.M):
            self.casc_mat_acc_vel[
                i * self.N:i * self.N + self.N,
                i * (self.N - 1):i * (self.N - 1) + self.N - 1
            ] = mat_acc_vel
            if i > 0:
                self.casc_mat_acc_vel[i * self.N, (i - 1) * (self.N - 1)] = -1 / self.DT
        self.casc_mat_acc_vel = np.stack([
            np.hstack([self.casc_mat_acc_vel, self.casc_mat_acc_vel * 0]),
            np.hstack([self.casc_mat_acc_vel * 0, self.casc_mat_acc_vel])
        ]).reshape(2 * self.M * self.N, 2 * self.M * (self.N - 1))
        self.mat_acc_vel = self.casc_mat_acc_vel

        filter_plan = np.zeros(self.N, dtype=int)
        filter_plan[0] = 1
        filter_plan = np.hstack([np.hstack([filter_plan] * self.M)] * 2)

        self.casc_mat_pos_vel_plan = self.casc_mat_pos_vel[np.nonzero(filter_plan)]

        filter_plan = np.zeros(self.N - 1, dtype=int)
        filter_plan[0] = 1
        filter_plan = np.hstack([np.hstack([filter_plan] * self.M)] * 2)
        self.casc_mat_vel_plan = np.eye(
            2 * self.M * (self.N - 1)
        )[np.nonzero(filter_plan)]


        if self.plan_type == "Position":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_pos_vel_plan.T @ self.casc_mat_pos_vel_plan +
                self.stability_coeff * np.eye(2 * self.M * (self.N - 1))
            )
            self.vec_p = lambda _1, plan, _2, vel: (
                -plan + 0.5 * self.DT * np.repeat(vel, self.M)
            ).T @ self.casc_mat_pos_vel_plan
        elif self.plan_type == "Velocity":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_vel_plan.T @ self.casc_mat_vel_plan
            )


            def vec_p(_1, _2, plan_vels, vel):
                plan_vels[:self.M] += vel[0]
                plan_vels[self.M:] += vel[1]
                return -plan_vels.T @ self.casc_mat_vel_plan
            self.vec_p = vec_p
        elif self.plan_type == "PositionVelocity":
            self.vel_coeff = 0.5
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_pos_vel_plan.T @ self.casc_mat_pos_vel_plan +
                self.vel_coeff * self.casc_mat_vel_plan.T @ self.casc_mat_vel_plan
            )


            def vec_p(_, plan_pos, plan_vels, vel):
                plan_vels[:self.M] += vel[0]
                plan_vels[self.M:] += vel[1]
                return (-plan_pos + 0.5 * self.DT * np.repeat(vel, self.M)).T @ \
                    self.casc_mat_pos_vel_plan - self.vel_coeff * plan_vels.T  @\
                    self.casc_mat_vel_plan
            self.vec_p = vec_p
        else:
            raise NotImplementedError

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const((self.N - 1) * self.M)
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N * self.M)
        self.last_planned_traj_casc = np.zeros((self.M * (self.N - 1) * 2))


    def gen_crowd_const(self, const_M, const_b, crowd_poss, vel):
        for member in range(crowd_poss.shape[1]):
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, vel)
            if ignore:
                continue
            mat_crowd = np.hstack([
                np.eye(self.N * self.M) * vec[:, 0], np.eye(self.N * self.M) * vec[:, 1]
            ])
            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + 0.5 * self.DT * np.repeat(vel, self.N * self.M)
            ) - np.array([self.CONST_DIST_CROWD] * self.N * self.M)
            mat_crowd_control = -mat_crowd @ self.casc_mat_pos_vel
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def find_relevant_idxs(self, vel):
        idxs = self.relevant_idxs(vel)
        idxs = np.hstack(list(idxs) * (self.N - 1) * self.M) + np.repeat(
            np.arange(
                0, (self.N - 1) * self.M * self.circle_lin_sides, self.circle_lin_sides
            ),
            3
        )
        return np.array(idxs, dtype=int)


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([
                np.eye(self.N * self.M) * line[0], np.eye(self.N * self.M) * line[1]
            ])
            limit = -mat_line @ (0.5 * self.DT * np.repeat(vel, self.N * self.M))\
                - line[2]

            const_M.append(-mat_line @ self.casc_mat_pos_vel)
            const_b.append(-limit)


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        crowd_vels.resize(self.num_crowd, 2) if crowd_vels is not None else None
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels
        horizon_crowd_poss = np.stack([crowd_poss] * (self.N + self.M)) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * (self.N + self.M), 0) * self.DT,
            np.arange(1, (self.N + self.M + 1))
        )
        return self.cascade_crowd_positions(horizon_crowd_poss)


    def cascade_crowd_positions(self, crowd_poss):
        """
        Take crowd positions and cascade them meaning from [1,..., M + N] convert to
        [1, 2,.., N, 2,..., N + 1, 3,..., M + N - 1, M + 1,..., M + N].

        Args:
            crowd_poss (numpy.ndarray): an array of size (num_crowd, 2) with the current
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
        return None, None


    def __call__(self, plan, obs):
        # goal, crowd_poss, agent_vel, crowd_vels, walls = obs
        # crowd_poss = self.calculate_crowd_poss(
        #     crowd_poss.reshape(self.num_crowd, 2), crowd_vels
        # )
        vel = self.core_mpc(plan, obs)
        breaking = vel is None
        if breaking:
            # print("Executing last computed braking trajectory!")
            vel = self.last_planned_traj[1:].flatten("F")
        else:
            # self.last_planned_traj_casc = vel
            vel = np.hstack([  # only next braking trajecotry is relevant
                vel[:self.N - 1],
                vel[self.M * (self.N - 1):self.M * (self.N - 1) + (self.N - 1)]
            ])
        action = np.array([
            np.append(vel[:self.N - 1], 0), np.append(vel[self.N - 1:], 0)
        ]).T
        # all_future_pos = 0.5 * self.DT * np.repeat(agent_vel, self.M * self.N) +\
        #     self.casc_mat_pos_vel @ self.last_planned_traj_casc
        # all_future_pos = np.array([
        #     all_future_pos[:self.N * self.M], all_future_pos[self.N * self.M:]
        # ]).T
        # print("Agent future pos", all_future_pos.shape)
        # print("Distances crowd member 1 and all future pos", np.linalg.norm(
        #     crowd_poss[:, 1, :] - all_future_pos, axis=-1
        # ))

        # print("Distances crowd member 1 and all future pos", np.linalg.norm(
        #     crowd_poss[:, 2, :] - all_future_pos, axis=-1
        # ))
        self.last_planned_traj = action.copy()
        # return action, all_future_pos
        return action, breaking
