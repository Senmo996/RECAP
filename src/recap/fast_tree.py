"""Optional lazy C++ tree builder; compilation stays in the working directory."""

import os
from pathlib import Path

_extension = None


def load_recap_extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        source = Path(__file__).with_name("recap_tree.cpp")
        build = Path(os.environ.get("RECAP_BUILD_DIR", "outputs/cache/recap_tree")).resolve()
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="recap_tree_v1", sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            build_directory=str(build), verbose=False,
        )
    return _extension
