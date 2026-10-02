"""MiSFP packets: class-conditional diagonal-Gaussian mixtures + exact serialization.

A packet carries, for every represented class c, R_c components

    {class_id, component_id, count, mean[d], variance[d]}

plus a small header (feature space, source, round, precision).  Internally all
moments are float64 *population* (denominator n) statistics and variances are
kept UNFLOORED so that moment pooling stays exact; floors are applied only at
numerical evaluation / sampling boundaries.  Standard deviations are derived
only when sampling or when building the baseline-compatible transport tuple.

Component ids are local bookkeeping. They carry no cross-client identity and
aggregation never matches components by id.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import torch

MAGIC = b"MSFP"
PRECISIONS = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
_PREC_BYTES = {"float16": 2, "float32": 4, "float64": 8}
# Per-class record header: int32 class_id + int32 n_components.
CLASS_OVERHEAD_BYTES = 8
# Per-component scalars: int32 component_id + float32 count.
COMPONENT_SCALAR_BYTES = 8


class FeatureSpaceMismatch(ValueError):
    """Raised when packets from incompatible representation spaces are pooled."""


@dataclass(frozen=True)
class FeatureSpace:
    """Identity of the representation a packet lives in.

    ``space_id`` names the extraction regime (e.g. ``client_to_edge/local`` or
    ``client_to_edge/snapshot@r3``).  Equal metadata is a necessary guard, NOT
    evidence that two encoders produce aligned coordinates.
    """

    boundary: str
    dim: int
    feature_shape: tuple
    space_id: str
    model_version: str = ""

    def compatible_with(self, other: "FeatureSpace") -> bool:
        return (
            self.boundary == other.boundary
            and self.dim == other.dim
            and tuple(self.feature_shape) == tuple(other.feature_shape)
            and self.space_id == other.space_id
        )

    def to_dict(self):
        return {
            "boundary": self.boundary,
            "dim": int(self.dim),
            "feature_shape": [int(s) for s in self.feature_shape],
            "space_id": self.space_id,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["boundary"], int(d["dim"]), tuple(d["feature_shape"]),
                   d["space_id"], d.get("model_version", ""))


@dataclass
class ClassMixture:
    class_id: int
    counts: torch.Tensor  # [R] float64, underlying data mass (not MC sample count)
    means: torch.Tensor  # [R, d] float64
    variances: torch.Tensor  # [R, d] float64, population, unfloored, >= 0
    component_ids: Optional[torch.Tensor] = None  # [R] int64

    def __post_init__(self):
        self.counts = torch.as_tensor(self.counts, dtype=torch.float64).reshape(-1).cpu()
        self.means = torch.as_tensor(self.means, dtype=torch.float64).cpu()
        self.variances = torch.as_tensor(self.variances, dtype=torch.float64).cpu()
        if self.means.dim() == 1:
            self.means = self.means.unsqueeze(0)
        if self.variances.dim() == 1:
            self.variances = self.variances.unsqueeze(0)
        R = self.counts.shape[0]
        if R < 1:
            raise ValueError(f"class {self.class_id}: a mixture needs >= 1 component")
        if self.means.shape[0] != R or self.variances.shape != self.means.shape:
            raise ValueError(
                f"class {self.class_id}: shape mismatch counts={tuple(self.counts.shape)} "
                f"means={tuple(self.means.shape)} variances={tuple(self.variances.shape)}"
            )
        for name, t in (("counts", self.counts), ("means", self.means), ("variances", self.variances)):
            if not torch.isfinite(t).all():
                raise ValueError(f"class {self.class_id}: non-finite {name}")
        if (self.counts <= 0).any():
            raise ValueError(f"class {self.class_id}: component counts must be positive")
        # Float round-off in pooled second moments may produce -1e-17; clamp only
        # that, reject anything materially negative.
        if (self.variances < -1e-9 * (1.0 + self.variances.abs().max())).any():
            raise ValueError(f"class {self.class_id}: negative variance")
        self.variances = self.variances.clamp_min(0.0)
        if self.component_ids is None:
            self.component_ids = torch.arange(R, dtype=torch.int64)
        else:
            self.component_ids = torch.as_tensor(self.component_ids, dtype=torch.int64).reshape(-1).cpu()

    @property
    def R(self) -> int:
        return int(self.counts.shape[0])

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    @property
    def weights(self) -> torch.Tensor:
        return self.counts / self.counts.sum()

    def mixture_mean(self) -> torch.Tensor:
        return (self.weights[:, None] * self.means).sum(0)

    def mixture_variance(self) -> torch.Tensor:
        """Exact total (law-of-total-variance) diagonal variance of the mixture."""
        mu = self.mixture_mean()
        w = self.weights[:, None]
        return (w * (self.variances + (self.means - mu) ** 2)).sum(0)

    def collapse(self) -> "ClassMixture":
        """Moment-matched single Gaussian (count and first/second moments preserved)."""
        return ClassMixture(self.class_id, self.counts.sum().reshape(1),
                            self.mixture_mean()[None], self.mixture_variance()[None],
                            torch.zeros(1, dtype=torch.int64))

    def canonical(self) -> "ClassMixture":
        """Order components by (-count, mean, variance); re-id 0..R-1.

        Canonical ordering makes greedy merging independent of incoming
        component order / ids (permutation invariance)."""
        keys = []
        for r in range(self.R):
            keys.append((-float(self.counts[r]), tuple(self.means[r].tolist()),
                         tuple(self.variances[r].tolist()), r))
        order = [k[-1] for k in sorted(keys)]
        idx = torch.tensor(order, dtype=torch.long)
        return ClassMixture(self.class_id, self.counts[idx], self.means[idx],
                            self.variances[idx], torch.arange(self.R, dtype=torch.int64))

    def clone(self) -> "ClassMixture":
        return ClassMixture(self.class_id, self.counts.clone(), self.means.clone(),
                            self.variances.clone(), self.component_ids.clone())


@dataclass
class MixturePacket:
    space: FeatureSpace
    source: str
    round_idx: int
    classes: Dict[int, ClassMixture]
    precision: str = "float32"
    # Local log only: never serialized, never counted as payload.
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision!r}")
        for c, cm in self.classes.items():
            if int(c) != int(cm.class_id):
                raise ValueError(f"class key {c} != class_id {cm.class_id}")
            if cm.d != self.space.dim:
                raise FeatureSpaceMismatch(
                    f"class {c}: dim {cm.d} != packet space dim {self.space.dim}")

    def header_dict(self):
        return {"space": self.space.to_dict(), "source": self.source,
                "round": int(self.round_idx), "precision": self.precision}

    def header_bytes(self) -> bytes:
        return json.dumps(self.header_dict(), sort_keys=True, separators=(",", ":")).encode()

    def num_components(self) -> int:
        return sum(cm.R for cm in self.classes.values())


def component_nbytes(dim: int, precision: str) -> int:
    return COMPONENT_SCALAR_BYTES + 2 * dim * _PREC_BYTES[precision]


def packet_nbytes(packet: MixturePacket) -> int:
    """Exact serialized size; equals ``len(serialize_packet(packet))`` (tested)."""
    n = len(MAGIC) + 4 + len(packet.header_bytes())
    per_comp = component_nbytes(packet.space.dim, packet.precision)
    for cm in packet.classes.values():
        n += CLASS_OVERHEAD_BYTES + cm.R * per_comp
    return n


def serialize_packet(packet: MixturePacket) -> bytes:
    dt = np.dtype(PRECISIONS[packet.precision]).newbyteorder("<")
    header = packet.header_bytes()
    out = [MAGIC, struct.pack("<I", len(header)), header]
    for c in sorted(packet.classes):
        cm = packet.classes[c]
        out.append(struct.pack("<ii", int(c), cm.R))
        for r in range(cm.R):
            out.append(struct.pack("<if", int(cm.component_ids[r]), float(cm.counts[r])))
            mean = cm.means[r].numpy().astype(dt)
            var = cm.variances[r].numpy().astype(dt)
            if not (np.isfinite(mean).all() and np.isfinite(var).all()):
                raise OverflowError(
                    f"class {c} component {r} is not representable in {packet.precision}")
            out.append(mean.tobytes())
            out.append(var.tobytes())
    return b"".join(out)


def deserialize_packet(blob: bytes) -> MixturePacket:
    if blob[:4] != MAGIC:
        raise ValueError("not a MiSFP packet")
    (hlen,) = struct.unpack_from("<I", blob, 4)
    off = 8
    header = json.loads(blob[off:off + hlen].decode())
    off += hlen
    space = FeatureSpace.from_dict(header["space"])
    prec = header["precision"]
    dt = np.dtype(PRECISIONS[prec]).newbyteorder("<")
    d = space.dim
    vb = d * _PREC_BYTES[prec]
    classes = {}
    while off < len(blob):
        c, R = struct.unpack_from("<ii", blob, off)
        off += 8
        ids, counts, means, vars_ = [], [], [], []
        for _ in range(R):
            cid, cnt = struct.unpack_from("<if", blob, off)
            off += 8
            means.append(np.frombuffer(blob, dtype=dt, count=d, offset=off).astype(np.float64))
            off += vb
            vars_.append(np.frombuffer(blob, dtype=dt, count=d, offset=off).astype(np.float64))
            off += vb
            ids.append(cid)
            counts.append(cnt)
        classes[c] = ClassMixture(c, torch.tensor(counts, dtype=torch.float64),
                                  torch.from_numpy(np.stack(means)),
                                  torch.from_numpy(np.stack(vars_)),
                                  torch.tensor(ids, dtype=torch.int64))
    return MixturePacket(space, header["source"], header["round"], classes, prec)


def transmit(packet: MixturePacket) -> tuple:
    """Simulate the wire: returns (received packet, serialized byte count).

    The receiver only ever sees dequantized values, so precision effects
    (e.g. float16) are real, not just an accounting label."""
    blob = serialize_packet(packet)
    rx = deserialize_packet(blob)
    rx.meta = dict(packet.meta)
    return rx, len(blob)


class MixtureTransport(tuple):
    """Baseline-compatible ``(proto_dict, std_dict)`` view of a MiSFP packet.

    The unchanged H-SFP round loop calls ``get_proto_dist_size_MB`` on what a
    client/edge returns; tensors here use the wire precision so that count is the
    tensor share of the payload. The remaining serialized bytes (header, ids,
    counts) are added to the same route by MiSFP, so route totals equal the
    exact serialized size.
    """

    def __new__(cls, packet: MixturePacket, nbytes: int):
        dtype = {"float16": torch.float16, "float32": torch.float32,
                 "float64": torch.float64}[packet.precision]
        protos = {c: cm.means.to(dtype) for c, cm in packet.classes.items()}
        stds = {c: cm.variances.sqrt().to(dtype) for c, cm in packet.classes.items()}
        obj = super().__new__(cls, (protos, stds))
        obj.packet = packet
        obj.nbytes = int(nbytes)
        return obj

    def tensor_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in self for t in d.values())


def concat_mixtures(mixtures: Iterable[ClassMixture]) -> ClassMixture:
    ms = list(mixtures)
    if not ms:
        raise ValueError("nothing to concatenate")
    c = ms[0].class_id
    d = ms[0].d
    for m in ms:
        if m.class_id != c:
            raise ValueError("concatenating different classes")
        if m.d != d:
            raise FeatureSpaceMismatch("concatenating mixtures with different dims")
    counts = torch.cat([m.counts for m in ms])
    return ClassMixture(c, counts, torch.cat([m.means for m in ms]),
                        torch.cat([m.variances for m in ms]),
                        torch.arange(counts.shape[0], dtype=torch.int64))


def pool_packets(packets) -> Dict[int, ClassMixture]:
    """Pool same-class components of compatible packets (no collapsing)."""
    packets = [p for p in packets if p is not None]
    if not packets:
        return {}
    ref = packets[0].space
    for p in packets[1:]:
        if not ref.compatible_with(p.space):
            raise FeatureSpaceMismatch(
                f"cannot pool packets from different feature spaces: {ref} vs {p.space}")
    by_class: Dict[int, list] = {}
    for p in packets:
        for c, cm in p.classes.items():
            by_class.setdefault(int(c), []).append(cm)
    return {c: concat_mixtures(v) for c, v in sorted(by_class.items())}
