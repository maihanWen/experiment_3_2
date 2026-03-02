#!/usr/bin/env python3
import os
import sys
import runpy
import types
import torch
from torch.profiler import profile, ProfilerActivity

def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))  # .../instrumented_toy_transformer_pkg

    # --- CRITICAL: create an alias package named "toy_transformer_pkg"
    # so imports like `from toy_transformer_pkg.data import ...` work,
    # even though the directory name is instrumented_toy_transformer_pkg.
    pkg_name = "toy_transformer_pkg"
    if pkg_name not in sys.modules:
        m = types.ModuleType(pkg_name)
        m.__path__ = [pkg_dir]  # treat this folder as the package root
        m.__file__ = os.path.join(pkg_dir, "__init__.py")
        sys.modules[pkg_name] = m

    # Also ensure current dir is importable (helps some import mechanisms)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    # Keep cwd inside package so relative file access works
    os.chdir(pkg_dir)

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    with profile(activities=acts, record_shapes=True, with_stack=True, acc_events=True) as prof:
        runpy.run_path(os.path.join(pkg_dir, "train.py"), run_name="__main__")

    prof.export_chrome_trace(os.path.join(pkg_dir, "trace.json"))
    print("Wrote trace.json in:", pkg_dir)

if __name__ == "__main__":
    main()
