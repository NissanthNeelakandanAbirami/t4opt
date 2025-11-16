"""T4-OPT: Lightweight Agentic LLM Training & Optimization System."""

__version__ = "1.0.0"
__author__ = "T4-OPT"

from . import agents
from . import training
from . import quant
from . import eval
from . import utils

__all__ = ["agents", "training", "quant", "eval", "utils"]

