"""Local PLAN-1 execution shim; training semantics remain config-driven.

The sandbox cannot create DataLoader worker sockets, and the production CIFAR
loader has no subset knob. This shim makes those two smoke-only execution
constraints effective before dispatching to the unified main.py entry point.
"""

import os
import runpy
import sys

import numpy as np
from torch.utils.data import Subset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
METHOD_DIR = os.path.join(ROOT, "classification", "H-SFP")
sys.path.insert(0, ROOT)
sys.path.insert(0, METHOD_DIR)

import data  # noqa: E402
import hierarchy  # noqa: E402


_get_dataset = data.get_dataset


def _bounded_get_dataset(args):
    train, valid, test, groups = _get_dataset(args)
    cap = int(args["smoke_train_samples_per_client"])
    bounded_groups = {}
    for client_id, indices in groups.items():
        ordered = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
        positions = np.linspace(0, len(ordered) - 1, min(cap, len(ordered)), dtype=np.int64)
        bounded_groups[client_id] = ordered[positions]

    eval_cap = min(int(args["smoke_eval_samples"]), len(valid), len(test))
    eval_indices = list(range(eval_cap))
    return train, Subset(valid, eval_indices), Subset(test, eval_indices), bounded_groups


data.get_dataset = _bounded_get_dataset

_hierarchy_init = hierarchy.HierarchicalFL.__init__


def _zero_worker_init(self, *args, **kwargs):
    _hierarchy_init(self, *args, **kwargs)
    self.num_workers = 0


hierarchy.HierarchicalFL.__init__ = _zero_worker_init

os.chdir(ROOT)
runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")
