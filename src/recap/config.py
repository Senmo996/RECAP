from dataclasses import dataclass


@dataclass(frozen=True)
class RecapConfig:
    """Main-paper defaults; the root is excluded from the candidate budget.

    Suffix insertion may add up to four nodes beyond the 63 transition nodes.
    The Python backend needs no compiler; cpp reproduces the main-run hot path.
    """

    top_k: int = 8
    depth: int = 10
    node_budget: int = 63
    suffix_tokens: int = 4
    fusion: str = "prior_confidence"
    backend: str = "python"

    def __post_init__(self):
        if min(self.top_k, self.depth, self.node_budget) < 1:
            raise ValueError("top_k, depth and node_budget must be positive")
        if not 0 <= self.suffix_tokens <= 4:
            raise ValueError("suffix_tokens must be between 0 and 4")
        if self.fusion not in ("prior_confidence", "fixed"):
            raise ValueError("fusion must be prior_confidence or fixed")
        if self.backend not in ("python", "cpp"):
            raise ValueError("backend must be python or cpp")
