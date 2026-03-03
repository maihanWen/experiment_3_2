#!/usr/bin/env python3
import os
import sys
import runpy
import torch
from torch.profiler import profile, ProfilerActivity

def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
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
