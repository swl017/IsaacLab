# file: mappo_rnn_groups.py
"""Parameter-ownership grouping for MAPPO-RNN.

Stock skrl multi-agent algorithms key optimizers and schedulers by `agent_id`.
That is correct only when every agent owns a disjoint set of trainable
parameter tensors. When two or more `uid` entries point to the same Python
model object — the homogeneous shared-policy/shared-value pattern used by
iris_ma6 — multiple independent Adam optimizers and KLAdaptiveLR controllers
end up driving the same parameter tensors. Each Adam keeps its own moment
buffers, the parameters are stepped once per uid per inner update, and the
schedulers compete over the same weights.

This module computes the partition of `uid`s into "ownership groups" by
parameter-tensor identity (`id(p)`), so the trainer can:

  - build exactly one optimizer per unique parameter set (shared groups
    collapse, disjoint agents stay separate);
  - aggregate per-uid losses inside a group before a single backward/step;
  - drive a single KLAdaptiveLR per group from the aggregated KL;
  - preserve the existing per-uid path automatically when models are
    heterogeneous (each uid lands in its own singleton group).

The trigger is parameter-tensor identity, not model identity, which means a
hybrid layout (shared encoder + per-agent heads) is detected the same way as
fully-shared models — any overlap between two uids' parameter id sets fuses
them into one group. Disjoint uids fall through unchanged.

This module is dependency-free apart from PyTorch so it can be unit-tested
without AppLauncher / Isaac Sim.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


def _trainable_params(module: Optional[nn.Module]) -> List[nn.Parameter]:
    """Return the list of `requires_grad=True` parameters of a module.

    A `None` module yields an empty list so we can call this uniformly on
    optional value/policy slots.
    """
    if module is None:
        return []
    return [p for p in module.parameters() if p.requires_grad]


def _uid_param_ids(policy: Optional[nn.Module], value: Optional[nn.Module]) -> Set[int]:
    """Return the set of `id(parameter)` covered by a uid's policy + value.

    Uses Python object identity (`id`) on the parameter tensors. Two uids
    sharing the same `nn.Module` instance — or the same individual parameter
    tensor — will produce overlapping sets and end up in the same ownership
    group. Two uids with structurally identical but distinct modules will
    produce disjoint sets and end up in separate groups.
    """
    ids: Set[int] = set()
    for p in _trainable_params(policy):
        ids.add(id(p))
    for p in _trainable_params(value):
        ids.add(id(p))
    return ids


@dataclass
class OwnershipGroup:
    """One group of `uid`s whose policy/value parameters overlap.

    Invariants enforced by `build_ownership_groups`:
      - `uids` is non-empty.
      - `param_ids` is the union of every contributing uid's parameter ids.
      - No two `OwnershipGroup`s returned by `build_ownership_groups` share
        any `id` in `param_ids` — partitioning is total over uids and
        disjoint over parameter ids.
    """

    uids: List[str]
    params: List[nn.Parameter]
    param_ids: Set[int]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.uids:
            raise ValueError("OwnershipGroup requires at least one uid")
        if not self.name:
            self.name = "+".join(self.uids)


def build_ownership_groups(
    possible_agents: Sequence[str],
    policies: Mapping[str, Optional[nn.Module]],
    values: Mapping[str, Optional[nn.Module]],
) -> List[OwnershipGroup]:
    """Partition `possible_agents` into disjoint parameter-ownership groups.

    Two uids land in the same group iff their (policy ∪ value) parameter-id
    sets overlap on at least one tensor. The detection is transitive: if uid
    A shares a parameter with B, and B shares a different parameter with C,
    all three end up in one group.

    The order of `OwnershipGroup`s in the returned list mirrors the order in
    which their first uid appears in `possible_agents`, so `for group in
    groups: for uid in group.uids:` iterates uids in the same order the
    caller would have used `for uid in possible_agents:`.

    Uids whose policy *and* value are both `None` (or have no trainable
    parameters) get their own degenerate singleton group with an empty
    parameter list — this matches the stock skrl behavior of skipping the
    optimizer construction for that uid, while keeping the bookkeeping
    uniform.
    """
    if not possible_agents:
        return []

    # Per-uid parameter id sets (cached once).
    uid_to_ids: Dict[str, Set[int]] = {
        uid: _uid_param_ids(policies.get(uid), values.get(uid)) for uid in possible_agents
    }

    # Union-find over uids keyed by parameter id overlap.
    parent: Dict[str, str] = {uid: uid for uid in possible_agents}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Map every parameter id to the first uid that introduced it. Subsequent
    # uids that re-introduce the same id get unioned with that uid.
    id_to_uid: Dict[int, str] = {}
    for uid in possible_agents:
        for pid in uid_to_ids[uid]:
            if pid in id_to_uid:
                union(id_to_uid[pid], uid)
            else:
                id_to_uid[pid] = uid

    # Materialize groups in the order their roots first appear.
    root_order: List[str] = []
    seen_roots: Set[str] = set()
    for uid in possible_agents:
        r = find(uid)
        if r not in seen_roots:
            seen_roots.add(r)
            root_order.append(r)

    root_to_uids: Dict[str, List[str]] = {r: [] for r in root_order}
    for uid in possible_agents:
        root_to_uids[find(uid)].append(uid)

    groups: List[OwnershipGroup] = []
    for root in root_order:
        uids_in_group = root_to_uids[root]
        # Collect the unique parameter tensors for the group, deduping by id.
        param_ids: Set[int] = set()
        params: List[nn.Parameter] = []
        for uid in uids_in_group:
            for p in itertools.chain(
                _trainable_params(policies.get(uid)),
                _trainable_params(values.get(uid)),
            ):
                pid = id(p)
                if pid not in param_ids:
                    param_ids.add(pid)
                    params.append(p)
        groups.append(
            OwnershipGroup(
                uids=uids_in_group,
                params=params,
                param_ids=param_ids,
                name="+".join(uids_in_group),
            )
        )

    return groups


def assert_groups_partition_params(groups: Sequence[OwnershipGroup]) -> None:
    """Sanity check: no parameter id appears in more than one group.

    The invariant is structural — if it ever fails, we have constructed
    overlapping optimizers and broken the whole point of this module. Cheap
    enough to call unconditionally during MAPPO_RNN.__init__.
    """
    seen: Dict[int, str] = {}
    for g in groups:
        for pid in g.param_ids:
            if pid in seen and seen[pid] != g.name:
                raise RuntimeError(
                    f"Parameter id {pid} appears in two ownership groups: "
                    f"{seen[pid]!r} and {g.name!r}"
                )
            seen[pid] = g.name


def homogeneous_config_check(
    group: OwnershipGroup,
    per_uid_value: Mapping[str, object],
    name: str,
) -> object:
    """Return the shared per-uid config value for a group, asserting equality.

    For groups with more than one uid (i.e. shared parameters across uids),
    per-uid hyperparameters that affect the optimization loop — number of
    learning epochs, number of mini-batches, KL early-stop threshold, etc. —
    must match across contributors, otherwise the aggregated step is
    ambiguous. For singleton groups this returns the lone value unchanged.
    """
    if len(group.uids) == 1:
        return per_uid_value[group.uids[0]]
    first = per_uid_value[group.uids[0]]
    for uid in group.uids[1:]:
        v = per_uid_value[uid]
        if v != first:
            raise ValueError(
                f"Ownership group {group.name!r} has mismatched per-uid "
                f"{name!r}: {group.uids[0]!r}={first!r} vs {uid!r}={v!r}. "
                f"Uids that share parameters must agree on {name!r}."
            )
    return first
