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
        crowd_max_vel: float,
        crowd_max_acc: float,
        plan_type: str = "Position",
        plan_length: int = 20,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        stability_coeff: float = 0.2,
        horizon_tries: int = 0,
        relax_uncertainty: float = 1.,
        passive_safety: bool = True,
        use_plan: bool = False,
        use_always_plan: bool = False,
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
            uncertainty=uncertainty,
            radius_crowd=radius_crowd,
            horizon_tries=horizon_tries,
            relax_uncertainty=relax_uncertainty,
            use_plan=use_plan,
            use_always_plan=use_always_plan,
        )
        self.M = plan_length
        self.MAX_DIST_STOP_CROWD = self.CROWD_MAX_VEL * (self.M + self.N) * self.DT * 2.
        self.plan_horizon = self.M
        self.plan_type = plan_type
        self.stability_coeff = stability_coeff
        self.passive_safety = passive_safety
        self.N_control = self.N - 1 if self.passive_safety else self.N
        self.last_planned_traj = np.zeros((self.M + self.N_control, 2))

        mat_pos_vel = self.mat_pos_vel[:self.N, :self.N_control]
        self.casc_mat_pos_vel = np.zeros((self.M * self.N, self.M * (self.N_control)))
        for i in range(self.M):
            self.casc_mat_pos_vel[
                i * self.N:(i + 1) * self.N,
                i * (self.N_control):i * (self.N_control) + self.N_control
            ] = mat_pos_vel
            for j in range(i):
                self.casc_mat_pos_vel[
                    i * self.N:(i + 1) * self.N, j * (self.N_control)
                ] = np.ones(self.N) * self.DT
        self.casc_mat_pos_vel = np.stack([
            np.hstack([self.casc_mat_pos_vel, self.casc_mat_pos_vel * 0]),
            np.hstack([self.casc_mat_pos_vel * 0, self.casc_mat_pos_vel])
        ]).reshape(2 * self.M * self.N, 2 * self.M * (self.N_control))
        self.mat_pos_vel = self.make_mat_pos_vel(self.M + self.N, self.M + self.N - 1)


        # (mat)rix to project control (vel)ocities to future (pos)itions only for crowd
        # since it is possible to have a shorter horizon only for the crowd constraint
        # for cascading we dont really dynamically change the horizon however the horizon
        # is onw step shorter since it is possible to crash on the last step since the
        # agent will have zero velocity (at least in passive safety)
        braking_idx_wout_last = np.array([
            i for i in range(self.M * self.N) if i % self.N != self.N - 1
        ])
        self.casc_mat_pos_vel_crowd = np.concatenate([
            self.casc_mat_pos_vel[braking_idx_wout_last],
            self.casc_mat_pos_vel[self.N * self.M + braking_idx_wout_last]
        ])

        mat_acc_vel = self.mat_acc_vel[:self.N, :self.N_control]
        self.casc_mat_acc_vel = np.zeros((self.M * self.N, self.M * (self.N_control)))
        for i in range(self.M):
            self.casc_mat_acc_vel[
                i * self.N:i * self.N + self.N,
                i * (self.N_control):i * (self.N_control) + self.N_control
            ] = mat_acc_vel
            if i > 0:
                self.casc_mat_acc_vel[i * self.N, (i - 1) * (self.N_control)] = -1 /\
                    self.DT
        self.casc_mat_acc_vel = np.stack([
            np.hstack([self.casc_mat_acc_vel, self.casc_mat_acc_vel * 0]),
            np.hstack([self.casc_mat_acc_vel * 0, self.casc_mat_acc_vel])
        ]).reshape(2 * self.M * self.N, 2 * self.M * (self.N_control))
        self.mat_acc_vel = self.casc_mat_acc_vel

        filter_plan = np.zeros(self.N, dtype=int)
        filter_plan[0] = 1
        filter_plan = np.hstack([np.hstack([filter_plan] * self.M)] * 2)

        self.casc_mat_pos_vel_plan = self.casc_mat_pos_vel[np.nonzero(filter_plan)]

        filter_plan = np.zeros(self.N_control, dtype=int)
        filter_plan[0] = 1
        filter_plan = np.hstack([np.hstack([filter_plan] * self.M)] * 2)
        self.casc_mat_vel_plan = np.eye(
            2 * self.M * (self.N_control)
        )[np.nonzero(filter_plan)]


        if self.plan_type == "Position":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_pos_vel_plan.T @ self.casc_mat_pos_vel_plan +
                self.stability_coeff * np.eye(2 * self.M * (self.N_control))
            )
            self.vec_p = lambda __1__, plan, __2__, vel: (
                -plan + 0.5 * self.DT * np.repeat(vel, self.M)
            ).T @ self.casc_mat_pos_vel_plan
        elif self.plan_type == "Velocity":
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_vel_plan.T @ self.casc_mat_vel_plan
            )


            def vec_vel_p(__1__, __2__, plan_vels, vel):
                plan_vels[:self.M] += vel[0]
                plan_vels[self.M:] += vel[1]
                return -plan_vels.T @ self.casc_mat_vel_plan
            self.vec_p = vec_vel_p
        elif self.plan_type == "PositionVelocity":
            self.vel_coeff = 0.5
            self.mat_Q = scipy.sparse.csc_matrix(
                self.casc_mat_pos_vel_plan.T @ self.casc_mat_pos_vel_plan +
                self.vel_coeff * self.casc_mat_vel_plan.T @ self.casc_mat_vel_plan
            )


            def vec_posvel_p(_, plan_pos, plan_vels, vel):
                plan_vels[:self.M] += vel[0]
                plan_vels[self.M:] += vel[1]
                return (-plan_pos + 0.5 * self.DT * np.repeat(vel, self.M)).T @ \
                    self.casc_mat_pos_vel_plan - self.vel_coeff * plan_vels.T  @\
                    self.casc_mat_vel_plan
            self.vec_p = vec_posvel_p
        else:
            raise NotImplementedError

        self.gen_vel_const((self.N_control) * self.M)
        self.gen_acc_const(self.N * self.M)
        self.last_planned_traj_casc = np.zeros((self.M * (self.N_control) * 2))


    def gen_crowd_const(self, **kwargs):
        const_M = kwargs["const_M"]
        const_b = kwargs["const_b"]
        crowd_poss = kwargs["crowd_poss"]
        vel = kwargs["agent_vel"]
        crowd_vels = kwargs["crowd_vels"]
        plan = kwargs["plan"]
        for i, member in enumerate(range(crowd_poss.shape[1])):
            dist_to_keep = self.CONST_DIST_CROWD
            if (isinstance(self.CONST_DIST_CROWD, np.ndarray) and
               len(self.CONST_DIST_CROWD.shape) >= 2 or hasattr(self, "member_indeces")):
                idx = i
                if hasattr(self, "member_indeces"):
                    idx = np.where(i < self.member_indeces)[0][0]
                if isinstance(self.CONST_DIST_CROWD, np.ndarray):
                    dist_to_keep = self.CONST_DIST_CROWD[idx]
            else:
                if isinstance(self.CONST_DIST_CROWD, np.ndarray):
                    dist_to_keep = self.CONST_DIST_CROWD.copy()
            poss, vec, ignore = self.ignore_crowd_member(
                crowd_poss, member, vel, plan
            )
            if ignore:
                continue
            mat_crowd = np.hstack([
                np.eye((self.N - 1) * self.M) * vec[:, 0],
                np.eye((self.N - 1) * self.M) * vec[:, 1]
            ])

            # if considering uncertainty update the distance to the crowd
            if isinstance(dist_to_keep, float):
                dist_to_keep = [dist_to_keep] * (self.N - 1) * self.M
            # relaxing the uncertainty constraint
            if self.uncertainty == "rdist":
                # lower uncertainty in the future for low velocities
                assert isinstance(crowd_vels, np.ndarray)
                assert isinstance(dist_to_keep, np.ndarray)
                speed = np.linalg.norm(crowd_vels[i])
                step_diff = (
                    dist_to_keep - np.ones_like(dist_to_keep) * dist_to_keep[0]
                ) / dist_to_keep * self.relax_uncertainty
                dist_to_keep *= (1. - step_diff * (1. - speed / self.CROWD_MAX_VEL))

            vec_crowd = mat_crowd @ (
                -poss.flatten("F") + 0.5 * self.DT * np.repeat(vel, (self.N - 1) * self.M)
            ) - np.array(dist_to_keep)
            mat_crowd_control = -mat_crowd @ self.casc_mat_pos_vel_crowd
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def find_relevant_idxs(self, vel):
        idxs = self.relevant_idxs(vel)
        idxs = np.hstack(list(idxs) * (self.N_control) * self.M) + np.repeat(
            np.arange(
                0,
                (self.N_control) * self.M * self.circle_lin_sides,
                self.circle_lin_sides
            ),
            3
        )
        return np.array(idxs, dtype=int)


    def lin_pos_constraint(self, **kwargs):
        const_M = kwargs["const_M"]
        const_b = kwargs["const_b"]
        line_eq = kwargs["line_eq"]
        vel = kwargs["vel"]
        for line in line_eq:
            mat_line = np.hstack([
                np.eye(self.N * self.M) * line[0], np.eye(self.N * self.M) * line[1]
            ])
            limit = -mat_line @ (0.5 * self.DT * np.repeat(vel, self.N * self.M))\
                - line[2]

            const_M.append(-mat_line @ self.casc_mat_pos_vel)
            const_b.append(-limit)


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        crowd_vels = crowd_vels.reshape(-1, 2) if crowd_vels is not None else None
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels
        horizon_crowd_poss = np.stack([crowd_poss] * (self.N + self.M - 1)) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * (self.N + self.M - 1), 0) * self.DT,
            np.arange(1, (self.N + self.M))
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
        casc_crowd_poss = np.zeros((self.M * (self.N - 1),) + crowd_poss.shape[1:])
        for i in range(self.M):
            casc_crowd_poss[i * (self.N - 1):(i + 1) * (self.N - 1), :, :] =\
                crowd_poss[i:i + self.N - 1, :, :]
        return casc_crowd_poss


    def traj_from_plan(self, current_vel):
        # all_future_pos = np.repeat(self.current_pos, self.M * self.N) +\
        #     0.5 * self.DT * np.repeat(current_vel, self.M * self.N) +\
        #     self.casc_mat_pos_vel @ self.last_planned_traj_casc
        # all_future_pos = np.array([
        #     all_future_pos[:self.N * self.M], all_future_pos[self.N * self.M:]
        # ]).T
        all_future_pos = np.repeat(self.current_pos, self.M + self.N) +\
            0.5 * self.DT * np.repeat(current_vel, self.M + self.N) +\
            self.mat_pos_vel @ self.last_planned_traj.flatten('F')
        all_future_pos = np.array([
            all_future_pos[:self.N + self.M], all_future_pos[self.N + self.M:]
        ]).T
        return all_future_pos


    def __call__(self, **kwargs):
        plan = kwargs["plan"]
        obs = kwargs["obs"]
        # goal, crowd_poss, agent_vel, crowd_vels, walls = obs
        # crowd_poss = self.calculate_crowd_poss(
        #     crowd_poss.reshape(-1, 2), crowd_vels
        # )
        vel = self.core_mpc(plan, obs)
        _, _, current_vel, _, _, _ = obs
        braking = vel is None
        if braking:
            # print("Executing last computed braking trajectory!")
            vel = self.last_planned_traj[1:].flatten("F")
            if not self.passive_safety:
                N_control = len(self.last_planned_traj)
                vel = np.zeros(2 * N_control)
                vel[:N_control - 1] = self.last_planned_traj[1:, 0]
                vel[N_control - 1] = self.last_planned_traj[-1, 0]
                vel[N_control:2 * N_control - 1] = self.last_planned_traj[1:, 1]
                vel[2 * N_control - 1] = self.last_planned_traj[-1:, 1]
            self.last_planned_traj_casc = np.concatenate([
                self.last_planned_traj_casc[self.N_control:self.M * (self.N_control)],
                np.zeros(self.N_control),
                self.last_planned_traj_casc[(self.M + 1) * (self.N_control):],
                np.zeros(self.N_control),
            ])
        else:
            self.last_planned_traj_casc = vel
            idx_x = np.concatenate([
                [i * (self.N_control) for i in range(self.M)],
                np.arange(
                    (self.M - 1) * (self.N_control) + 1, self.M * (self.N_control)
                )
            ])
            idx_y = (self.M * (self.N_control)) + idx_x
            vel = np.hstack([vel[idx_x], vel[idx_y]])
        if self.passive_safety:
            action = np.array([
                np.append(vel[:self.M + self.N_control - 1], 0),
                np.append(vel[self.M + self.N_control - 1:], 0)
            ]).T
        else:
            action = np.array([
                vel[:self.M + self.N_control - 1], vel[self.M + self.N_control - 1:]
            ]).T
        self.last_planned_traj = action.copy()
        self.last_traj = self.traj_from_plan(current_vel)
        self.set_action(action, braking)


    def reset(self):
        self.last_planned_traj_casc = np.zeros((self.M * (self.N_control) * 2))
        super().reset()
