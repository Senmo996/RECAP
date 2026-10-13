"""RECAP inference, with no dependency on the original experiment repository."""

from .config import RecapConfig

__version__ = "0.1.0"
__all__ = ["RecapConfig", "RecapModel", "RecapOutput"]


def __getattr__(name):
    if name in ("RecapModel", "RecapOutput"):
        from .model import RecapModel, RecapOutput
        return {"RecapModel": RecapModel, "RecapOutput": RecapOutput}[name]
    raise AttributeError(name)
