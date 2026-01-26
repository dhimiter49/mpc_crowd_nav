from typing import Union
import numpy as np
import scipy

from mpc.vel import MPCVel


class MPC_SQP_Vel(MPCVel):
    """
    Reiterate the QP using Newton method in order to handle non-linear constraints (SQP).
    In the end, linear constraints are used (option for non linear crowd constraint).
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
        horizon_tries: int = 0,
        relax_uncertainty: float = 1.,
        passive_safety: bool = True,
        lin_crowd_const: bool = True,
        **_
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
            passive_safety=passive_safety,
        )
        self.all_tries = []  # keep track of how many tries
        self.lin_crowd_const = lin_crowd_const
        self.sqp_loops = 30
        self.last_sqp_solution = np.zeros(2 * self.N_control)
        mat_pos_vel_quad = self.mat_pos_vel.T @ self.mat_pos_vel
        self.mat_Q = scipy.sparse.csc_matrix(mat_pos_vel_quad)


        def vec_p(_1, plan, _2, vel, **_):
            goal_obj = mat_pos_vel_quad @ self.last_sqp_solution +\
                (-plan + 0.5 * self.DT * np.repeat(vel, self.N)).T  @ self.mat_pos_vel
            return goal_obj
        self.vec_p = vec_p

        self.mat_vel_const, self.vec_vel_const = self.gen_vel_const(self.N_control)
        self.mat_acc_const, self.vec_acc_const = self.gen_acc_const(self.N)


    def lin_pos_constraint(self, const_M, const_b, line_eq, vel):
        """The linear position constraint is given by the equation ax+yb+c"""
        for line in line_eq:
            mat_line = np.hstack([np.eye(self.N) * line[0], np.eye(self.N) * line[1]])
            limit = -mat_line @ self.mat_pos_vel @ self.last_sqp_solution -\
                mat_line @ (0.5 * self.DT * np.repeat(vel, self.N)) - line[2]

            const_M.append(-mat_line @ self.mat_pos_vel)
            const_b.append(-limit)


    def gen_acc_const(self, horizon):
        M_a_, b_a_, sgn_acc = super().gen_acc_param(horizon)


        def acc_vec_const(agent_vel):
            agent_vel_ = np.zeros(2 * (horizon))
            agent_vel_[0], agent_vel_[horizon] = agent_vel
            return -np.einsum("ij,i->ij", M_a_ @ self.mat_acc_vel, sgn_acc) @\
                self.last_sqp_solution + sgn_acc * (b_a_ + M_a_ @ agent_vel_ / self.DT)

        return np.einsum("ij,i->ij", M_a_ @ self.mat_acc_vel, sgn_acc), acc_vec_const


    def gen_vel_const(self, horizon):
        M_v_, b_v_, sgn_vel = super().gen_vel_param(horizon)


        def mat_vel_const(idxs):
            ret = -np.einsum("ij,i->ij", M_v_, sgn_vel)
            return ret if idxs is None else ret[idxs]


        def vec_vel_const(_, idxs):
            ret = np.einsum("ij,i->ij", M_v_, sgn_vel) @ self.last_sqp_solution +\
                sgn_vel * b_v_
            return ret if idxs is None else ret[idxs]


        return mat_vel_const, vec_vel_const


    def gen_crowd_const(self, const_M, const_b, crowd_poss, agent_vel, crowd_vels=None):
        for i, member in enumerate(range(crowd_poss.shape[1])):
            # if considering uncertainty update the distance to the crowd
            if (not isinstance(self.CONST_DIST_CROWD, float) and
               len(self.CONST_DIST_CROWD.shape) >= 2 or hasattr(self, "member_indeces")):
                idx = i
                if hasattr(self, "member_indeces"):
                    idx = np.where(i < self.member_indeces)[0][0]
                dist_to_keep = self.CONST_DIST_CROWD[idx]
            else:
                if isinstance(self.CONST_DIST_CROWD, np.ndarray):
                    dist_to_keep = self.CONST_DIST_CROWD.copy()
                else:
                    dist_to_keep = self.CONST_DIST_CROWD

            # ignore if two far and different direction
            poss, vec, ignore = self.ignore_crowd_member(crowd_poss, member, agent_vel)
            if ignore:
                continue

            # if considering uncertainty update the distance to the crowd
            if isinstance(dist_to_keep, float):
                dist_to_keep = [dist_to_keep] * self.N_crowd
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

            # constraint formula
            if self.lin_crowd_const:
                mat_crowd = np.hstack([
                    np.eye(self.N_crowd) * vec[:, 0], np.eye(self.N_crowd) * vec[:, 1]
                ])
                # derivative part
                vec_crowd = mat_crowd @ self.mat_pos_vel_crowd @ self.last_sqp_solution
                # old part
                vec_crowd += mat_crowd @ (
                    -poss.flatten("F") +
                    0.5 * self.DT * np.repeat(agent_vel, self.N_crowd)
                ) - np.array(dist_to_keep)
                mat_crowd_control = -mat_crowd @ self.mat_pos_vel_crowd
            else:
                poss = -poss
                agent_vel_par = np.stack([agent_vel] * self.N).reshape(self.N, 2)
                mat_pos_vel_crowd_xy = self.mat_pos_vel_crowd.reshape(
                    (self.N, 2, -1), order='F'
                )
                term_dist_crowd = 2 * np.einsum("ij,ijk->ik", poss, mat_pos_vel_crowd_xy)
                term_agent_vel = self.DT * np.einsum(
                    "ij,ijk->ik", agent_vel_par, mat_pos_vel_crowd_xy
                )
                term_quadratic = 2 * np.matmul(
                    np.transpose(mat_pos_vel_crowd_xy, (0, 2, 1)), mat_pos_vel_crowd_xy
                ) @ self.last_sqp_solution
                vec_crowd = (term_dist_crowd + term_agent_vel) @ self.last_sqp_solution +\
                    0.5 * np.einsum(
                        "i,ji->j",
                        self.last_sqp_solution,
                        term_quadratic
                    ) + np.einsum(
                        "ij,ij->i", poss, poss
                    ) + self.DT * np.einsum(
                        "ij,ij->i", poss, agent_vel_par
                    ) + 0.25 * self.DT ** 2 * np.einsum(
                        "ij,ij->i", agent_vel_par, agent_vel_par
                    ) - np.array(dist_to_keep) ** 2
                mat_crowd_control = -(term_dist_crowd + term_agent_vel + term_quadratic)

            # update/add constraints
            const_M.append(mat_crowd_control)
            const_b.append(vec_crowd)


    def __call__(self, plan, obs):
        tries = self.sqp_loops
        braking = False
        _, _, current_vel, _, _, _ = obs
        last_action = None
        while (
            (
                tries == self.sqp_loops or
                np.linalg.norm(self.last_sqp_solution - last_action) > 1e-5
            ) and tries > 0 and not braking
        ):
            if tries < self.sqp_loops:
                self.last_sqp_solution = last_action
            step = self.core_mpc(plan, obs)
            braking = step is None

            if braking:
                # print("Executing last computed braking trajectory!")
                vel = self.last_planned_traj[1:].flatten("F")
                if not self.passive_safety:
                    N_control = len(self.last_planned_traj)
                    vel = np.zeros(2 * N_control)
                    vel[:N_control - 1] = self.last_planned_traj[1:, 0]
                    vel[N_control - 1] = self.last_planned_traj[-1, 0]
                    vel[N_control:2 * N_control - 1] = self.last_planned_traj[1:, 1]
                    vel[1 * N_control - 1] = self.last_planned_traj[-1:, 1]
            else:
                vel = self.last_sqp_solution + step

            if self.passive_safety:
                action = np.array([
                    np.append(vel[:len(vel) // 2], 0), np.append(vel[len(vel) // 2:], 0)
                ]).T
                last_action = action[:-1].flatten('F')
            else:
                action = np.array([vel[:len(vel) // 2], vel[len(vel) // 2:]]).T
                last_action = action.flatten('F')
            self.last_planned_traj = action.copy()
            self.last_pos = self.current_pos
            tries -= 1
        # if not braking:
        #     self.all_tries.append(self.sqp_loops - tries)
        self.last_traj = self.traj_from_plan(current_vel)
        return action, braking


    def reset(self):
        super().reset()
        self.last_sqp_solution = np.zeros(2 * self.N_control)
