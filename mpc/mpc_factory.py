from mpc.mpc import MPC
from mpc.mpc_linear import MPCLinear

ALL_TYPES = ["simple", "linear_plan", "velocity_control"]


def get_mpc(mpc_type: str, **kwargs):
    mpc_type = mpc_type.lower()
    if mpc_type == "simple":
        return MPC(**kwargs)
    elif mpc_type == "linear_plan":
        return MPCLinear(**kwargs)
    else:
        raise ValueError(f"Specified mpc type {mpc_type} not supported, "
                         f"please choose one of {ALL_TYPES}.")
