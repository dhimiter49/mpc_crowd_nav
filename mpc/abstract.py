from typing import Union
import numpy as np
from qpsolvers import solve_qp
import scipy

from mpc.utils import gen_polygon


class AbstractMPC:
    """
    Abstract MPC for crowd navigation. Solves a qp problem by using the defined quadratic
    matrices and constraints in the child classes.
    """
    def __init__(
        self,
        horizon: int,
        dt: float,
        physical_space: float,
        const_dist_crowd: Union[float, list[float]],
        agent_max_vel: float,
        agent_max_acc: float,
        crowd_max_vel: float,
        crowd_max_acc: float,
        uncertainty: str = "",
        radius_crowd: Union[list[float], None] = None,
        horizon_tries: int = 0,
        horizon_crowd_pred: Union[int, None] = None,
        relax_uncertainty: float = 1.,
    ):
        """
        Args:
            physical_space: the radius of the robot
            const_dist_crowd: distance to keep to other members of the crowd, either
                constant, same for everyone or a list, different for everyon
            uncertainty: enums of strings, 'dir' only direction varies, 'vel' direction
                and speed varies, 'dist' calculates only based on distance, 'rdist'
                relaxed distance that lowers the distance for steps in the future
            radius_crowd: radii of the different members of the crowd
            horizon_tire: if a solution is not found with the current horizon it might
                mean that there are too many constraints, by lowering the horizon it is
                possible to find a solution
            horizon_crowd_pred: this only lowers the horizon for the crowd constraints and
                not the objective or the other constraints
            relax_uncertainty: if uncertainty is "rdist" than choose the factor for which
                to relax it, 0 no relaxation and 1 is no ucertainty
        """
        # vars
        self.N = horizon
        self.plan_horizon = self.N
        self.horizon_tries = horizon_tries
        self.short_hor_only_crowd = False
        self.N_crowd = self.N if horizon_crowd_pred is None else horizon_crowd_pred
        self.DT = dt
        self.PHYSICAL_SPACE = physical_space
        self.AGENT_MAX_VEL = agent_max_vel
        self.AGENT_MAX_ACC = agent_max_acc
        self.CROWD_MAX_VEL = crowd_max_vel
        self.CROWD_MAX_ACC = crowd_max_acc
        self.MAX_TIME_STOP = self.AGENT_MAX_VEL / self.AGENT_MAX_ACC
        self.MAX_DIST_STOP = 2 * self.AGENT_MAX_VEL * self.MAX_TIME_STOP
        self.MAX_DIST_STOP_CROWD = 6  # arbitrary, be careful on variable radii

        # different constant distance crowd depending on different radii
        if radius_crowd is not None:
            self.radius_crowd = radius_crowd
            if np.all(radius_crowd == radius_crowd[0]):
                self.CONST_DIST_CROWD = 2 * self.PHYSICAL_SPACE + 0.01001
            else:
                # 0.01 takes care of the continuity in the real analog world while the
                # collision are checked discretely in time
                self.CONST_DIST_CROWD = np.array(radius_crowd) + self.PHYSICAL_SPACE +\
                    0.01
        else:
            self.CONST_DIST_CROWD = const_dist_crowd

        # if considering uncertainty then need to change the safety distances during the
        # future trajectory
        self.uncertainty = uncertainty
        self.relax_uncertainty = relax_uncertainty
        if self.uncertainty == "dist" or self.uncertainty == "rdist":
            if radius_crowd is not None:
                self.CONST_DIST_CROWD = np.expand_dims(
                    self.CONST_DIST_CROWD, -1
                ).repeat(self.N_crowd, -1)
            else:
                self.CONST_DIST_CROWD = self.CONST_DIST_CROWD * np.ones(self.N_crowd)
            self.CONST_DIST_CROWD += max(
                self.AGENT_MAX_VEL * self.DT, self.AGENT_MAX_ACC * self.DT ** 2,
                self.CROWD_MAX_VEL * self.DT, self.CROWD_MAX_ACC * self.DT ** 2
            ) * np.arange(1, self.N_crowd + 1)

        # linearization of the acceleration and velocity limits
        self.circle_lin_sides = 8
        self.POLYGON_ACC_LINES = gen_polygon(self.AGENT_MAX_ACC, self.circle_lin_sides)
        self.POLYGON_VEL_LINES = gen_polygon(self.AGENT_MAX_VEL, self.circle_lin_sides)

        # other vars
        self.last_planned_traj = np.zeros((self.N, 2))
        self.current_pos = None
        self.pos_horizon = None
        self.last_pos = None


    def get_action(self, plan, obs):
        return self.__call__(plan, obs)


    def core_mpc(self, plan, obs):
        pos_plan, vel_plan = plan
        goal, crowd_poss, vel, crowd_vels, walls, radii = obs

        # Read the radii again, maybe they have changed, in that case,
        # update the safety distances
        if radii is not None and len(radii) != 0:
            self.PHYSICAL_SPACE = radii[0]
            self.CONST_DIST_CROWD = self.PHYSICAL_SPACE + radii[1:]
            if self.uncertainty == "dist" or self.uncertainty == "rdist":
                self.CONST_DIST_CROWD = np.expand_dims(self.CONST_DIST_CROWD, -1).repeat(
                    self.N_crowd, -1
                )
                self.CONST_DIST_CROWD += self.AGENT_MAX_VEL * self.DT *\
                    np.arange(1, self.N_crowd + 1)
        vel_plan[:self.plan_horizon] -= vel[0]
        vel_plan[self.plan_horizon:] -= vel[1]

        # Constraints
        const_M, const_b = [], []
        if crowd_poss is not None:
            crowd_poss = self.calculate_crowd_poss(
                crowd_poss.reshape(-1, 2), crowd_vels
            )
            self.gen_crowd_const(const_M, const_b, crowd_poss, vel, crowd_vels)
        crowd_const_dim = len(const_M)
        wall_eqs = self.wall_eq(walls)
        if len(wall_eqs) != 0:
            self.lin_pos_constraint(const_M, const_b, wall_eqs, vel)
        wall_const_dim = len(const_M) - crowd_const_dim
        # idxs = self.find_relevant_idxs(vel)
        const_M.append(self.mat_acc_const)
        const_b.append(self.vec_acc_const(vel))
        const_M.append(self.mat_vel_const(None))
        const_b.append(self.vec_vel_const(vel, None))
        acc_vel_const_dim = len(const_M) - crowd_const_dim - wall_const_dim

        term_const_M, term_const_b = self.terminal_const(vel)

        # QP solver
        opt_V = self.vec_p(goal, pos_plan, vel_plan, vel)
        const_M = scipy.sparse.csr_matrix(np.vstack(const_M))
        const_b = np.hstack(const_b)
        solution = solve_qp(
            # self.mat_Q, self.vec_p(goal, pos_plan, vel_plan, vel, crowd_poss),
            self.mat_Q, opt_V,
            # lb=-acc_b, ub=acc_b,
            G=const_M, h=const_b,
            A=term_const_M, b=term_const_b,
            solver="clarabel",
            # eps_abs=1e-4,
            # eps_duality_gap_abs=1e-4,
            tol_gap_abs=5e-5,
            tol_gap_rel=5e-5,
            tol_feas=1e-4,
            tol_infeas_abs=5e-5,
            tol_infeas_rel=5e-5,
            tol_ktratio=1e-4
        )

        # Shorten horizon main loop
        if solution is None and self.horizon_tries > 0:
            horizon = self.N
            full_dim = crowd_const_dim + wall_const_dim + acc_vel_const_dim
            opt_Q = self.mat_Q.toarray() if not self.short_hor_only_crowd else self.mat_Q
            short_dims = crowd_const_dim if self.short_hor_only_crowd else full_dim
            while self.horizon_tries > 0:
                shorten_by = horizon // 2
                # print("Using a shorter crowd horizon of: ", horizon - shorten_by)
                del_idx = list(np.array([
                    np.arange(horizon - shorten_by, horizon) + horizon * i
                    for i in range(0, short_dims)
                ]).flatten())
                const_M = np.delete(const_M.toarray(), del_idx, axis=0)
                const_b = np.delete(const_b, del_idx, axis=0)

                if not self.short_hor_only_crowd:
                    # remove indeces from the objective
                    obj_horizon = horizon - 1 if "Vel" in type(self).__name__ else horizon
                    del_idx = list(np.array([
                        np.arange(obj_horizon - shorten_by, obj_horizon) + obj_horizon * i
                        for i in range(0, 2)  # 2 dims x and y
                    ]).flatten())
                    opt_Q = np.delete(opt_Q, del_idx, axis=0)
                    opt_Q = np.delete(opt_Q, del_idx, axis=1)
                    const_M = np.delete(const_M, del_idx, axis=1)
                    opt_V = np.delete(opt_V, del_idx, axis=0)
                    opt_Q = scipy.sparse.csr_matrix(opt_Q)

                const_M = scipy.sparse.csr_matrix(const_M)

                # try again with shorter crowd horizon
                solution = solve_qp(
                    opt_Q, opt_V,
                    G=const_M, h=const_b,
                    A=term_const_M, b=term_const_b,
                    solver="clarabel",
                    tol_gap_abs=5e-5,
                    tol_gap_rel=5e-5,
                    tol_feas=1e-4,
                    tol_infeas_abs=5e-5,
                    tol_infeas_rel=5e-5,
                    tol_ktratio=1e-4
                )
                opt_Q = opt_Q.toarray() if not self.short_hor_only_crowd else opt_Q
                if solution is not None:
                    break
                horizon -= shorten_by
                self.horizon_tries -= 1
            self.horizon_tries = 3

        return solution  # control plan


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        """
        Based on the current crowd positions and constant velocities it is possible to
        compute all future positions.
        """
        crowd_vels = crowd_vels.reshape(-1, 2) if crowd_vels is not None else None
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels

        crowd_poss, crowd_vels = self.vel_uncertainty(crowd_poss, crowd_vels)

        # propagate position in the future based on the currrent position and current vel
        return np.stack([crowd_poss] * self.N_crowd) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N_crowd, 0) * self.DT,
            np.arange(1, self.N_crowd + 1)
        )


    def vel_uncertainty(self, crowd_poss, crowd_vels):
        """
        !!!Works in practice only for specific acceleration and velocity!!!

        If the model is uncertain about the velocities of the crowd then additional
        possible future velocities are added to the computation in order to copmute
        multiple different future trajectories of crowd member.

        Uncertainty is added in the direction of the given velocity and its absolute
        value. If the velocity given as a ratio to the maximum velocity is between:
            0 to 1/3, perturbations are mapped linearly between 2pi and pi/2
            1/3 to 2/3, perturbations are mapped linearly between pi/2 and pi/4
            2/3 to 1, perturbations are mapped linearly between pi/4 and pi/12
        The ranges are discretized, 5 for the first case and three for the other two.
        The speed is perturbed by the the given maximum acceleration, which means that
        the current velocity is perturbed maximally by the max_acc x time_step and can be
        either accelerating or decelerating. This means two more velocities are added.

        Does not support varying future velocities.
        """
        new_crowd_vels = []
        if self.uncertainty in ["dir", "vel"]:
            crowd_speeds_rel_max = np.linalg.norm(crowd_vels, axis=-1) /\
                self.AGENT_MAX_VEL
            alphas = np.where(
                crowd_speeds_rel_max <= 2 / 3,
                np.where(
                    crowd_speeds_rel_max <= 1 / 3,
                    2 * np.pi - 9 * np.pi / 2 * crowd_speeds_rel_max,
                    3 * np.pi / 4 - 3 * np.pi / 4 * crowd_speeds_rel_max,
                ),
                7 * np.pi / 12 - np.pi / 2 * crowd_speeds_rel_max
            )
            n_trajs = np.where(alphas >= np.pi / 2, 5, 3)  # 3 traj if <= 90, else 5
            n_trajs = n_trajs.reshape(-1)
            angles = alphas * (1 / (n_trajs - 1))
            all_dir_crowd_vels = np.repeat(crowd_vels, n_trajs, axis=0)
            all_dir_angles = np.repeat(angles, n_trajs, axis=0)

            # start from current angle (0) then remove alpha (-1) then add alpha (1)
            # then for five remove twice alpha (-2) and add twice alpha (2) ...
            mult_angles = np.array([0, -1, 1, -2, 2, -3, 3, -4, 4])
            all_dir_angles *= np.concatenate([mult_angles[:i] for i in n_trajs])
            dir_matrix = np.stack([
                np.cos(all_dir_angles), -np.sin(all_dir_angles),
                np.sin(all_dir_angles), np.cos(all_dir_angles)
            ], axis=-1).reshape(len(all_dir_crowd_vels), 2, 2)

            new_crowd_vels = np.einsum('ijk,ij->ij', dir_matrix, all_dir_crowd_vels)
            crowd_poss = np.repeat(crowd_poss, n_trajs, axis=0)
            crowd_vels = new_crowd_vels
            if hasattr(self, "radius_crowd"):
                self.member_indeces = np.cumsum(n_trajs)

        if self.uncertainty == "vel":
            crowd_poss = np.repeat(crowd_poss, 3, axis=0)
            new_crowd_vels = np.repeat(crowd_vels, 3, axis=0)
            uncertainty = np.stack([
                np.array([0, self.AGENT_MAX_ACC, self.AGENT_MAX_ACC]) / np.sqrt(2) *
                self.DT * np.array([1, 1, -1]),
            ] * len(crowd_vels)).reshape(-1, 1)
            new_crowd_vels += uncertainty
            crowd_vels = new_crowd_vels
            if hasattr(self, "radius_crowd"):
                self.member_indeces *= 3

        return crowd_poss, crowd_vels


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
        return eqs[wall_dist < self.MAX_DIST_STOP_CROWD]


    def relevant_idxs(self, vel):
        """
        !!! Use with caution, reachability of a velocity depends on acceleration !!!

        Relevant indexes when computing acceleration and velocity constraints. Based on
        the direction of the current velocity it is unecessary to add constraints to the
        qp-problem that address opposite directions. The relative indexes are defined as
        the three regions in which the current velocity falls in. In cases where the
        linearization of the velocity and acceleration constraint is split into 8 parts
        this means (360 / 8) * 3 = 135 degress are covered.
        This is defined arbitrariyl and works for velocity of 3m/s and acceleration of
        1.5m/s2.
        """
        angle = np.arctan2(vel[1], vel[0])
        angle = 2 * np.pi + angle if angle < 0 else angle
        angle_idx = angle // (2 * np.pi / self.circle_lin_sides)
        idxs = [
            angle_idx,
            (angle_idx + 1) % self.circle_lin_sides,
            (angle_idx - 1) % self.circle_lin_sides
        ]
        return idxs


    def ignore_crowd_member(self, crowd_poss, member, agent_vel):
        """
        Ignore crowd members that are too far or in another direction.

        Return:
            np.ndarray: relative position of member
            np.ndarray: direction vector of member
            bool: ignore flag
        """
        poss = crowd_poss[:, member, :]
        # print(np.where(np.linalg.norm(poss, axis=-1) <= 0.8)[0])
        zero_idx = np.where(np.linalg.norm(poss, axis=-1) == 0)[0]
        poss[zero_idx] += 1e-8
        vec = -(poss.T / np.linalg.norm(poss, axis=-1)).T
        dist = np.linalg.norm(poss, axis=-1)
        angle = np.arccos(np.clip(np.dot(-vec, agent_vel), -1, 1)) > np.pi / 2
        return poss, vec, (
            np.all(dist > self.MAX_DIST_STOP_CROWD) or
            (np.all(dist > self.MAX_DIST_STOP_CROWD / 2) and np.all(angle))
        )


    def gen_vel_param(self, horizon):
        """
        Parameters, matrix vector and sign that represent the lienear constraint for
        velocity.
        """
        M_v_ = np.vstack([np.eye(horizon) * -line[0] for line in self.POLYGON_VEL_LINES])
        M_v_ = np.hstack(
            [M_v_, np.vstack([np.eye(horizon)] * len(self.POLYGON_VEL_LINES))]
        )
        sgn_vel = np.ones(len(self.POLYGON_VEL_LINES))
        sgn_vel[len(self.POLYGON_VEL_LINES) // 2:] = -1
        sgn_vel = np.repeat(sgn_vel, horizon)
        b_v_ = np.repeat(self.POLYGON_VEL_LINES[:, 1], horizon)

        return M_v_, b_v_, sgn_vel


    def gen_acc_param(self, horizon):
        """
        Parameters, matrix vector and sign that represent the lienear constraint for
        acceleration.
        """
        M_a_ = np.vstack([np.eye(horizon) * -line[0] for line in self.POLYGON_ACC_LINES])
        M_a_ = np.hstack(
            [M_a_, np.vstack([np.eye(horizon)] * len(self.POLYGON_ACC_LINES))]
        )
        sgn_acc = np.ones(len(self.POLYGON_ACC_LINES))
        sgn_acc[len(self.POLYGON_ACC_LINES) // 2:] = -1
        sgn_acc = np.repeat(sgn_acc, horizon)
        b_a_ = np.repeat(self.POLYGON_ACC_LINES[:, 1], horizon)

        return M_a_, b_a_, sgn_acc


    def reset(self):
        self.last_planned_traj *= 0
        self.current_pos = None
