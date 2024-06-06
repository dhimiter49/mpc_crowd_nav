from mpc.mpc_acc import MPCAcc
from mpc.mpc_vel import MPCVel
from mpc.mpc_linear import MPCLinear

ALL_TYPES = ["simple", "linear_plan", "velocity_control"]


def get_mpc(mpc_type: str, **kwargs):
    mpc_type = mpc_type.lower()
    if mpc_type == "simple":
        return MPCAcc(**kwargs)
    elif mpc_type == "linear_plan":
        return MPCLinear(**kwargs)
    elif mpc_type == "velocity_control":
        return MPCVel(**kwargs)
    else:
        raise ValueError(f"Specified mpc type {mpc_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
