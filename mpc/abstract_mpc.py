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
        agent_max_vel: float,
        agent_max_acc: float,
        n_crowd: int = 0,
    ):
        self.N = horizon
        self.DT = dt
        self.PHYSICAL_SPACE = physical_space
        self.AGENT_MAX_VEL = agent_max_vel
        self.AGENT_MAX_ACC = agent_max_acc
        self.MAX_TIME_STOP = self.AGENT_MAX_VEL / self.AGENT_MAX_ACC
        self.MAX_DIST_STOP = self.MAX_TIME_STOP ** 2 * self.AGENT_MAX_ACC * 0.5
        self.MAX_DIST_STOP_CROWD = 2 * self.MAX_DIST_STOP
        self.n_crowd = n_crowd

        self.circle_lin_sides = 8
        self.POLYGON_ACC_LINES = gen_polygon(self.AGENT_MAX_ACC, self.circle_lin_sides)
        self.POLYGON_VEL_LINES = gen_polygon(self.AGENT_MAX_VEL, self.circle_lin_sides)

        self.vel_coeff = 0.2
        self.stability_coeff = 0.25
        self.last_planned_traj = np.zeros((self.N, 2))


    def get_action(self, plan, obs):
        return self.__call__(plan, obs)


    def __call__(self, plan, obs):
        pos_plan, vel_plan = plan
        goal, crowd_poss, vel, crowd_vels, walls = obs
        vel_plan[:self.N] -= vel[0]
        vel_plan[self.N:] -= vel[1]

        # Constraints
        const_M, const_b = [], []
        wall_eqs = self.wall_eq(walls)
        if len(wall_eqs) != 0:
            self.lin_pos_constraint(const_M, const_b, wall_eqs, vel)
        idxs = self.relevant_idxs(vel)
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
            self.mat_Q, self.vec_p(goal, pos_plan, vel_plan, vel),
            # lb=-acc_b, ub=acc_b,
            G=scipy.sparse.csc_matrix(np.vstack(const_M)), h=np.hstack(const_b),
            A=term_const_M, b=term_const_b,
            solver="clarabel",
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
        crowd_vels = crowd_poss * 0 if crowd_vels is None else crowd_vels
        return np.stack([crowd_poss] * self.N) + np.einsum(
            'ijk,i->ijk',
            np.stack([crowd_vels] * self.N, 0) * self.DT,
            np.arange(0, self.N)
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
        return eqs[wall_dist < self.MAX_DIST_STOP * 0.6]


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
        idxs = np.hstack(list(idxs) * self.N) + np.repeat(
            np.arange(0, self.N * self.circle_lin_sides, self.circle_lin_sides), 3
        )
        return np.array(idxs, dtype=int)
