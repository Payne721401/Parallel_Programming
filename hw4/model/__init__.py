"""Editable complete FourRoom H-JEPA model; action paths support autograd."""
from .factory import build_model
from .hjepa import HJEPA
from .jepa import JEPA

# Legacy entry spelling also returns the complete model, without inference_mode.
build_runner = build_model
__all__ = ["build_model", "build_runner", "HJEPA", "JEPA"]
