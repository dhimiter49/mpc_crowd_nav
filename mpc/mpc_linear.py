import numpy as np
from qpsolvers import solve_qp
from mpc.mpc import AbstractMPC


class MPCLinear(AbstractMPC):
    def __call__(self, plan, obs):
        return None
