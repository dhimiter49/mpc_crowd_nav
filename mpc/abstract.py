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
        const_dist_crowd: float,
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
        uncertainty: str = "",
    ):
        self.N = horizon
        self.plan_horizon = self.N
        self.DT = dt
        self.PHYSICAL_SPACE = physical_space
        self.CONST_DIST_CROWD = const_dist_crowd
        self.AGENT_MAX_VEL = agent_max_vel
        self.AGENT_MAX_ACC = agent_max_acc
        self.MAX_TIME_STOP = self.AGENT_MAX_VEL / self.AGENT_MAX_ACC
        self.MAX_DIST_STOP = self.MAX_TIME_STOP ** 2 * self.AGENT_MAX_ACC * 0.5
        self.MAX_DIST_STOP_CROWD = 2 * self.MAX_DIST_STOP
        self.n_crowd = n_crowd
        self.uncertainty = uncertainty

        self.circle_lin_sides = 8
        self.POLYGON_ACC_LINES = gen_polygon(self.AGENT_MAX_ACC, self.circle_lin_sides)
        self.POLYGON_VEL_LINES = gen_polygon(self.AGENT_MAX_VEL, self.circle_lin_sides)

        self.vel_coeff = 0.2
        self.stability_coeff = 0.25
        self.last_planned_traj = np.zeros((self.N, 2))


    def get_action(self, plan, obs):
        return self.__call__(plan, obs)


    def core_mpc(self, plan, obs):
        pos_plan, vel_plan = plan
        goal, crowd_poss, vel, crowd_vels, walls = obs
        vel_plan[:self.plan_horizon] -= vel[0]
        vel_plan[self.plan_horizon:] -= vel[1]

        # Constraints
        const_M, const_b = [], []
        wall_eqs = self.wall_eq(walls)
        if len(wall_eqs) != 0:
            self.lin_pos_constraint(const_M, const_b, wall_eqs, vel)
        idxs = self.find_relevant_idxs(vel)
        const_M.append(self.mat_acc_const)
        const_b.append(self.vec_acc_const(vel))
        const_M.append(self.mat_vel_const(idxs))
        const_b.append(self.vec_vel_const(vel, idxs))
        if self.n_crowd > 0:
            crowd_poss = self.calculate_crowd_poss(
                crowd_poss.reshape(self.n_crowd, 2), crowd_vels
            )
            self.gen_crowd_const(const_M, const_b, crowd_poss, vel)

        term_const_M, term_const_b = self.terminal_const(vel)

        return solve_qp(
            # self.mat_Q, self.vec_p(goal, pos_plan, vel_plan, vel, crowd_poss),
            self.mat_Q, self.vec_p(goal, pos_plan, vel_plan, vel),
            # lb=-acc_b, ub=acc_b,
            G=scipy.sparse.csc_matrix(np.vstack(const_M)), h=np.hstack(const_b),
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


    def calculate_crowd_poss(self, crowd_poss, crowd_vels):
        """
        Based on the current crowd positions and constant velocities it is possible to
        compute all future positions.

        Does not support varying future velocities.
        """
        crowd_vels.resize(self.n_crowd, 2) if crowd_vels is not None else None
        crowd_vels = crowd_vels * 0 if crowd_vels is None else crowd_vels
        new_crowd_vels = []
        if self.uncertainty in ["dir", "vel"]:
            alphas = np.pi - 5 * np.pi / 6 * (
                np.linalg.norm(crowd_vels, axis=-1) / self.AGENT_MAX_VEL
            )
            n_trajs = np.where(alphas > np.pi / 2, 5, 3)  # 3 traj if less then 90, else 5
            n_trajs = n_trajs.reshape(self.n_crowd)
            angles = alphas * (1 / (n_trajs - 1))
            for i, vel in enumerate(crowd_vels):
                for j in range(n_trajs[i]):
                    angle = (j // 2 if j % 2 == 0 else - (j + 1) // 2) * angles[i]
                    new_crowd_vels.append(np.array([
                        np.cos(angle) * vel[0] - np.sin(angle) * vel[1],
                        np.sin(angle) * vel[0] + np.cos(angle) * vel[1],
                    ]))

            crowd_poss = np.repeat(crowd_poss, n_trajs, axis=0)
            new_crowd_vels = np.array(new_crowd_vels)
            crowd_vels = new_crowd_vels

        if self.uncertainty == "vel":
            crowd_poss = np.repeat(crowd_poss, 3, axis=0)
            new_crowd_vels = np.repeat(crowd_vels, 3, axis=0)
            for i in range(len(new_crowd_vels)):
                if i % 3 == 0:
                    continue
                if i % 3 == 1:
                    new_crowd_vels[i] -= np.linalg.norm(new_crowd_vels[i]) * 0.1
                if i % 3 == 2:
                    new_crowd_vels[i] += np.linalg.norm(new_crowd_vels[i]) * 0.1
            crowd_vels = new_crowd_vels

        return np.stack([crowd_poss] * self.N) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N, 0) * self.DT,
            np.arange(1, self.N + 1)
        )


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
        Relevant indexes when computing acceleration and velocity constraints. Based on
        the direction of the current velocity it is unecessary to add constraints to the
        qp-problem that address opposite directions. The relative indexes are defined as
        the three regions in which the current velocity falls in. In cases where the
        linearization of the velocity and acceleration constraint is split into 8 parts
        this means (360 / 8) * 3 = 135 degress are covered.
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
        angle = np.arccos(np.clip(np.dot(-vec, agent_vel), -1, 1)) > np.pi / 4
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
