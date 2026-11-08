"""Editable HW4 entry: optimize the fixed FourRoom world-model computation."""
from model import build_model as build_baseline_model


def build_model(weights, architecture):
    """Return the model used by BOTH quick check and full planning evaluation.

    Preserve the fixed architecture, weights and JEPA component interface.
    Planning needs gradients through actions: do not use inference_mode or
    detach action-dependent results. Custom kernels need a correct backward.
    """
    return build_baseline_model(weights, architecture)
