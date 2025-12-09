from mpc.acc import MPCAcc
from mpc.vel import MPCVel
from mpc.sqp_vel import MPC_SQP_Vel
from mpc.linear import MPCLinear
from mpc.casc_acc import MPCCascAcc
from mpc.casc_vel import MPCCascVel
from mpc.sqp_casc_vel import MPC_SQP_CascVel

ALL_TYPES = [
    "simple", "linear_plan", "velocity_control", "cascading",
    "velocity_control_cascading", "sequential", "sequential_cascading"
]


def get_mpc(mpc_type: str, **kwargs):
    mpc_type = mpc_type.lower()
    if mpc_type == "simple":
        return MPCAcc(**kwargs)
    elif mpc_type == "linear_plan":
        return MPCLinear(**kwargs)
    elif mpc_type == "velocity_control":
        return MPCVel(**kwargs)
    elif mpc_type == "sequential":
        return MPC_SQP_Vel(**kwargs)
    elif mpc_type == "cascading":
        return MPCCascAcc(**kwargs)
    elif mpc_type == "velocity_control_cascading":
        return MPCCascVel(**kwargs)
    elif mpc_type == "sequential_cascading":
        return MPC_SQP_CascVel(**kwargs)
    else:
        raise ValueError(f"Specified mpc type {mpc_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
