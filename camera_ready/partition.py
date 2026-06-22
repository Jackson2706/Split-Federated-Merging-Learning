"""
Dirichlet and two-level (edge/client) Dirichlet data partitioning.

The two-level scheme separates *inter-edge* heterogeneity (alpha_edge) from
*intra-edge* heterogeneity (alpha_client):

    1. For each class, sample an edge-level distribution ~ Dir(alpha_edge) over
       edges and split that class' samples across edges accordingly.
    2. Within each edge, for each class present, sample a client-level
       distribution ~ Dir(alpha_client) over that edge's clients and split.

Smaller alpha => more skew (more non-IID). Larger alpha => closer to uniform.

These functions are dataset-agnostic: they operate on an integer `labels` array
and return index groups, so they work for CIFAR-10/100, HAM10000, ImageNet, etc.

`two_level_dirichlet_partition` also returns the client->edge mapping so the
hierarchy can place each client under the edge it was actually drawn for
(instead of the default random assignment).
"""

import numpy as np


def _as_labels(labels):
    return np.asarray(labels).astype(np.int64).reshape(-1)


def _class_indices(labels, num_classes):
    return {c: np.where(labels == c)[0] for c in range(num_classes)}


def _dirichlet_split(idxs, num_parts, alpha, rng):
    """Split a 1-D index array into `num_parts` chunks via Dir(alpha)."""
    if len(idxs) == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_parts)]
    idxs = idxs.copy()
    rng.shuffle(idxs)
    props = rng.dirichlet(np.full(num_parts, alpha))
    cuts = (np.cumsum(props)[:-1] * len(idxs)).astype(int)
    return [np.asarray(s, dtype=np.int64) for s in np.split(idxs, cuts)]


def _ensure_min(user_groups, labels, num_users, min_size, rng):
    """Guarantee each client has >= min_size samples by donating from the largest."""
    if min_size <= 0:
        return user_groups
    sizes = {c: len(user_groups[c]) for c in range(num_users)}
    for c in range(num_users):
        while sizes[c] < min_size:
            donor = max(sizes, key=lambda k: sizes[k])
            if donor == c or sizes[donor] <= min_size:
                break
            take = user_groups[donor][:1]
            user_groups[donor] = user_groups[donor][1:]
            user_groups[c] = np.concatenate([user_groups[c], take])
            sizes[donor] -= 1
            sizes[c] += 1
    return user_groups


def dirichlet_partition(labels, num_users, alpha, seed=0, num_classes=None,
                        min_size=1):
    """Flat Dirichlet non-IID partition.

    Returns: dict {client_id: np.ndarray of sample indices}.
    """
    labels = _as_labels(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    user_groups = {c: np.array([], dtype=np.int64) for c in range(num_users)}
    for c, idxs in _class_indices(labels, num_classes).items():
        parts = _dirichlet_split(idxs, num_users, alpha, rng)
        for u in range(num_users):
            user_groups[u] = np.concatenate([user_groups[u], parts[u]])

    user_groups = _ensure_min(user_groups, labels, num_users, min_size, rng)
    for u in user_groups:
        rng.shuffle(user_groups[u])
    return user_groups


def two_level_dirichlet_partition(labels, num_users, num_edges, alpha_edge,
                                  alpha_client, seed=0, num_classes=None,
                                  min_size=1):
    """Two-level Dirichlet partition (edge-level then client-level).

    Returns:
        user_groups: dict {client_id: np.ndarray of sample indices}.
        client_to_edge: dict {client_id: edge_id}.
    """
    labels = _as_labels(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    # Deterministic, contiguous client->edge assignment (last edge takes remainder).
    clients_per_edge = num_users // num_edges
    edge_clients = {}
    client_to_edge = {}
    nxt = 0
    for e in range(num_edges):
        n = clients_per_edge if e < num_edges - 1 else (num_users - nxt)
        members = list(range(nxt, nxt + n))
        edge_clients[e] = members
        for cid in members:
            client_to_edge[cid] = e
        nxt += n

    # Level 1: split each class across edges via Dir(alpha_edge).
    edge_class_idx = {e: {} for e in range(num_edges)}
    for c, idxs in _class_indices(labels, num_classes).items():
        parts = _dirichlet_split(idxs, num_edges, alpha_edge, rng)
        for e in range(num_edges):
            edge_class_idx[e][c] = parts[e]

    # Level 2: within each edge, split each class across that edge's clients
    # via Dir(alpha_client).
    user_groups = {c: np.array([], dtype=np.int64) for c in range(num_users)}
    for e in range(num_edges):
        members = edge_clients[e]
        if len(members) == 0:
            continue
        for c in range(num_classes):
            idxs = edge_class_idx[e][c]
            parts = _dirichlet_split(idxs, len(members), alpha_client, rng)
            for j, cid in enumerate(members):
                user_groups[cid] = np.concatenate([user_groups[cid], parts[j]])

    user_groups = _ensure_min(user_groups, labels, num_users, min_size, rng)
    for u in user_groups:
        rng.shuffle(user_groups[u])
    return user_groups, client_to_edge


# --------------------------------------------------------------------------- #
# Config-driven entry point (used by each method's get_data.py)
# --------------------------------------------------------------------------- #
def _extract_labels(dataset):
    """Best-effort integer label extraction from a torch/torchvision dataset."""
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            return _as_labels(getattr(dataset, attr))
    # torch.utils.data.Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = _extract_labels(dataset.dataset)
        return base[np.asarray(dataset.indices)]
    # Fallback: iterate (slow; last resort for custom datasets).
    return _as_labels([int(y) for _, y in dataset])


def partition_from_config(dataset, args):
    """Build user_groups per `args["partition"]` ('dirichlet'|'two_level_dirichlet').

    For two-level, stashes the client->edge mapping in args["_client_to_edge"] so
    the hierarchy can honor it. Returns dict {client_id: np.ndarray of indices}.
    """
    labels = _extract_labels(dataset)
    num_users = int(args["num_users"])
    seed = int(args.get("seed", 0) or 0)
    num_classes = args.get("num_classes")
    part = args["partition"]

    if part == "dirichlet":
        alpha = float(args.get("dirichlet_alpha", args.get("alpha", 0.1)))
        return dirichlet_partition(labels, num_users, alpha, seed=seed,
                                   num_classes=num_classes)

    if part == "two_level_dirichlet":
        num_edges = args.get("num_edges")
        if num_edges is None:
            ms = args.get("mid_server", [1])
            num_edges = ms[0] if isinstance(ms, (list, tuple)) else int(ms)
        ug, c2e = two_level_dirichlet_partition(
            labels, num_users, int(num_edges),
            alpha_edge=float(args.get("alpha_edge", 0.1)),
            alpha_client=float(args.get("alpha_client", 0.1)),
            seed=seed, num_classes=num_classes)
        args["_client_to_edge"] = c2e
        return ug

    raise ValueError(f"unknown partition mode: {part}")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def partition_diagnostics(user_groups, labels, num_classes=None,
                          client_to_edge=None):
    """Compute partition diagnostics.

    Returns a dict with:
        samples_per_client       : {cid: int}
        classes_per_client       : {cid: int}  (number of distinct classes)
        class_hist_per_client    : {cid: [count per class]}
        class_hist_per_edge      : {eid: [count per class]}  (if client_to_edge)
    """
    labels = _as_labels(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    samples_per_client, classes_per_client, class_hist_per_client = {}, {}, {}
    for cid, idxs in user_groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        hist = np.bincount(labels[idxs], minlength=num_classes) if len(idxs) else \
            np.zeros(num_classes, dtype=np.int64)
        class_hist_per_client[cid] = hist.tolist()
        samples_per_client[cid] = int(len(idxs))
        classes_per_client[cid] = int((hist > 0).sum())

    out = {
        "num_classes": int(num_classes),
        "num_clients": len(user_groups),
        "samples_per_client": samples_per_client,
        "classes_per_client": classes_per_client,
        "class_hist_per_client": class_hist_per_client,
    }

    if client_to_edge is not None:
        edge_hist = {}
        for cid, idxs in user_groups.items():
            e = client_to_edge.get(cid)
            if e is None:
                continue
            idxs = np.asarray(idxs, dtype=np.int64)
            h = np.bincount(labels[idxs], minlength=num_classes) if len(idxs) else \
                np.zeros(num_classes, dtype=np.int64)
            edge_hist[e] = (edge_hist.get(e, np.zeros(num_classes, dtype=np.int64)) + h)
        out["class_hist_per_edge"] = {int(e): h.tolist() for e, h in edge_hist.items()}
        out["client_to_edge"] = {int(k): int(v) for k, v in client_to_edge.items()}
    return out


def plot_partition(diag, out_prefix, title=""):
    """Save partition-diagnostic figures (PNG + PDF). Requires matplotlib.

    Produces:
        <out_prefix>_edge_class_hist.{png,pdf}    (if edge info present)
        <out_prefix>_client_class_hist.{png,pdf}
        <out_prefix>_client_summary.{png,pdf}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_partition] matplotlib not available; skipping figures.")
        return []

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    saved = []
    num_classes = diag["num_classes"]

    def _save(fig, name):
        for ext in ("png", "pdf"):
            p = f"{out_prefix}_{name}.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            saved.append(p)
        plt.close(fig)

    # Edge x class heatmap
    if "class_hist_per_edge" in diag:
        edges = sorted(diag["class_hist_per_edge"].keys())
        mat = np.array([diag["class_hist_per_edge"][e] for e in edges])
        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.12), 0.5 * len(edges) + 1.5))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xlabel("class"); ax.set_ylabel("edge")
        ax.set_yticks(range(len(edges))); ax.set_yticklabels(edges)
        ax.set_title(f"{title} class histogram per edge".strip())
        fig.colorbar(im, ax=ax, label="#samples")
        _save(fig, "edge_class_hist")

    # Client x class heatmap (sorted by edge if available)
    cids = sorted(diag["class_hist_per_client"].keys(),
                  key=lambda c: (diag.get("client_to_edge", {}).get(c, 0), c))
    mat = np.array([diag["class_hist_per_client"][c] for c in cids])
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.12), max(3, len(cids) * 0.03)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("class"); ax.set_ylabel("client")
    ax.set_title(f"{title} class histogram per client".strip())
    fig.colorbar(im, ax=ax, label="#samples")
    _save(fig, "client_class_hist")

    # Summary bars: #samples and #classes per client
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3))
    a0.bar(range(len(cids)), [diag["samples_per_client"][c] for c in cids])
    a0.set_title("samples per client"); a0.set_xlabel("client"); a0.set_ylabel("#samples")
    a1.bar(range(len(cids)), [diag["classes_per_client"][c] for c in cids], color="tab:orange")
    a1.set_title("classes per client"); a1.set_xlabel("client"); a1.set_ylabel("#classes")
    _save(fig, "client_summary")
    return saved


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic 10-class dataset, no torchvision needed.
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=5000)

    print("== Flat Dirichlet (alpha=0.1), 20 clients ==")
    ug = dirichlet_partition(labels, num_users=20, alpha=0.1, seed=0)
    d = partition_diagnostics(ug, labels)
    assert sum(d["samples_per_client"].values()) == len(labels)
    print("  total samples:", sum(d["samples_per_client"].values()))
    print("  classes/client:", list(d["classes_per_client"].values()))

    print("== Two-level Dirichlet (ae=0.1, ac=1.0), 20 clients / 4 edges ==")
    ug2, c2e = two_level_dirichlet_partition(
        labels, num_users=20, num_edges=4, alpha_edge=0.1, alpha_client=1.0, seed=0)
    d2 = partition_diagnostics(ug2, labels, client_to_edge=c2e)
    assert sum(d2["samples_per_client"].values()) == len(labels)
    assert set(c2e.keys()) == set(range(20))
    print("  total samples:", sum(d2["samples_per_client"].values()))
    print("  client_to_edge:", c2e)
    for e in sorted(d2["class_hist_per_edge"]):
        h = np.array(d2["class_hist_per_edge"][e])
        print(f"  edge {e}: classes={int((h>0).sum())} samples={int(h.sum())}")
    print("OK")
