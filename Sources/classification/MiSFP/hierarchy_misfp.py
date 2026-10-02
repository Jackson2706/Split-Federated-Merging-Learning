"""MiSFP round logic on top of the preserved H-SFP ``HierarchicalFL``.

Only three phases are overridden; client training, client selection, model
aggregation (t1/t2), validation and checkpointing are inherited unchanged.

  client:  H-SFP local SupCon training (inherited) -> deterministic extraction
           pass -> per-class seeded k-means mixture -> budgeted compression ->
           serialized packet (receiver sees dequantized values).
  edge:    pool same-class components of compatible client packets -> optional
           cap -> synthesize (balanced class prior, within-class pi) -> edge
           SupCon training (H-SFP loop) -> re-estimate class/component
           statistics in the edge OUTPUT space -> budgeted compression -> packet.
  cloud:   pool edge packets -> synthesize -> supervised CE (H-SFP loop).

Requires the H-SFP method directory on ``sys.path`` (the MiSFP runner adds it).
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from ehsfp.communication import add_communication
from hierarchy import (HierarchicalFL, average_state_dicts, build_edge_ssl_transforms,
                       get_model_size_MB, supervised_contrastive_loss)
from misfp import (ClassMixture, FeatureSpace, FitConfig, MixturePacket, MixtureTransport,
                   between_source_dispersion, collapse_baseline_average, component_nbytes,
                   compress_classes, fit_class_mixture, make_generator, mixture_log_prob,
                   packet_nbytes, pool_packets, sample_component, synthesize, transmit)
from misfp.packets import CLASS_OVERHEAD_BYTES

MB = 1024.0 ** 2


def _autocast(device):
    return torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))


def global_client_model(hfl):
    """The encoder H-SFP validation grades (FedAvg of cached trained clients)."""
    if getattr(hfl, "client_sync", False):
        m = copy.deepcopy(hfl.client_model)
        m.load_state_dict(average_state_dicts(list(hfl.edge_client_state.values())))
        return m
    layer = hfl.structure[-1]
    cids = [c for c in dict.fromkeys(hfl.client_cache) if c in layer and layer[c] is not None]
    if not cids:
        return copy.deepcopy(hfl.client_model)
    m = copy.deepcopy(hfl.client_model)
    m.load_state_dict(average_state_dicts([layer[c].state_dict() for c in cids]))
    return m


def current_pipeline(hfl):
    """Composed model exactly as ``HierarchicalFL._run_validation`` builds it."""
    client = global_client_model(hfl)
    edge = hfl.structure[0][hfl.connectivity[-1][0]]
    cloud = hfl.structure[len(hfl.args["mid_server"])][0]
    return hfl._full_pipeline(client, edge, cloud)


class MiSFPHierarchicalFL(HierarchicalFL):
    def __init__(self, *a, misfp_cfg=None, misfp_log_path=None, **kw):
        super().__init__(*a, **kw)
        self.mcfg = misfp_cfg
        self.seed = int(self.args.get("seed", 0) or 0)
        self.fit_cfg = FitConfig(
            k_mode=misfp_cfg["misfp_local_k_mode"], k=int(misfp_cfg["misfp_local_k"]),
            min_component_support=int(misfp_cfg["misfp_min_component_support"]),
            kmeans_iters=int(misfp_cfg["misfp_kmeans_iters"]),
            kmeans_n_init=int(misfp_cfg["misfp_kmeans_n_init"]),
            reservoir_size=misfp_cfg["misfp_reservoir_size"],
            adaptive_val_fraction=float(misfp_cfg["misfp_adaptive_val_fraction"]),
            adaptive_min_val=int(misfp_cfg["misfp_adaptive_min_val"]),
            adaptive_min_improvement=float(misfp_cfg["misfp_adaptive_min_improvement"]),
            adaptive_penalty_per_component=float(misfp_cfg["misfp_adaptive_penalty_per_component"]),
            nll_rel_var_floor=float(misfp_cfg["misfp_nll_rel_var_floor"]),
            nll_abs_var_floor=float(misfp_cfg["misfp_nll_abs_var_floor"]),
        )
        self.refit_cfg = FitConfig(
            k_mode=misfp_cfg["misfp_boundary_refit_k_mode"],
            k=int(misfp_cfg["misfp_boundary_refit_k"]),
            min_component_support=int(misfp_cfg["misfp_min_component_support"]),
            kmeans_iters=int(misfp_cfg["misfp_kmeans_iters"]),
            kmeans_n_init=int(misfp_cfg["misfp_kmeans_n_init"]),
            reservoir_size=misfp_cfg["misfp_reservoir_size"],
            adaptive_val_fraction=float(misfp_cfg["misfp_adaptive_val_fraction"]),
            adaptive_min_val=int(misfp_cfg["misfp_adaptive_min_val"]),
            adaptive_min_improvement=float(misfp_cfg["misfp_adaptive_min_improvement"]),
            adaptive_penalty_per_component=float(misfp_cfg["misfp_adaptive_penalty_per_component"]),
            nll_rel_var_floor=float(misfp_cfg["misfp_nll_rel_var_floor"]),
            nll_abs_var_floor=float(misfp_cfg["misfp_nll_abs_var_floor"]),
        )
        self.var_floor = float(misfp_cfg["misfp_sample_var_floor"])
        self.precision = misfp_cfg["misfp_precision"]
        self.log_path = misfp_log_path
        self.totals = {
            "client_to_edge_payload_bytes": 0, "edge_to_cloud_payload_bytes": 0,
            "client_to_edge_packets": 0, "edge_to_cloud_packets": 0,
            "client_to_edge_components": 0, "edge_to_cloud_components": 0,
            "snapshot_broadcast_MB": 0.0, "synthetic_samples_edge": 0,
            "synthetic_samples_cloud": 0, "boundary_mc_samples": 0,
            "infeasible_budget_events": 0,
            "time_feature_extraction_s": 0.0, "time_local_fit_s": 0.0,
            "time_client_compress_s": 0.0, "time_edge_merge_s": 0.0,
            "time_edge_synthesis_s": 0.0, "time_edge_reestimate_s": 0.0,
            "time_cloud_merge_s": 0.0, "time_cloud_synthesis_s": 0.0,
            "time_snapshot_s": 0.0,
        }
        self._round = None
        self._snapshot = None
        self._snapshot_round = None
        self._last_client_pool = {}

    # ------------------------------------------------------------------ utils
    def _new_round_log(self):
        return {"round": self._current_round, "selected_clients": [], "client_fits": [],
                "client_compress": [], "edge": [], "cloud": None, "dispersion": [],
                "budget_status": []}

    def _ensure_round(self):
        if self._round is None or self._round["round"] != self._current_round:
            self._round = self._new_round_log()
            self._last_client_pool = {}

    def _space(self, boundary, dim, shape):
        if boundary == "client_to_edge":
            regime = ("snapshot@r%d" % self._current_round
                      if self.mcfg["misfp_extraction"] == "shared_snapshot" else "local")
        else:
            regime = "edge_out"
        return FeatureSpace(boundary, int(dim), tuple(int(s) for s in shape),
                            f"{boundary}/{regime}", model_version=f"r{self._current_round}")

    def _budget_bytes(self, spec, classes, header_len, dim):
        if spec is None:
            return None
        kind, val = spec
        if kind == "absolute":
            return int(val)
        k1 = 8 + header_len + len(classes) * (CLASS_OVERHEAD_BYTES + component_nbytes(dim, "float32"))
        return int(val * k1)

    def _compress_packet(self, packet: MixturePacket, cap, budget_spec):
        dim = packet.space.dim
        per = component_nbytes(dim, packet.precision)
        hlen = len(packet.header_bytes())

        def nbytes(classes):
            return 8 + hlen + sum(CLASS_OVERHEAD_BYTES + cm.R * per for cm in classes.values())

        budget = self._budget_bytes(budget_spec, packet.classes, hlen, dim)
        classes, rep = compress_classes(packet.classes, cap, budget, nbytes, per)
        if rep.status == "infeasible":
            self.totals["infeasible_budget_events"] += 1
            msg = (f"[MiSFP] infeasible budget for {packet.source}: need {rep.bytes_after} B "
                   f"(one component/class) > budget {budget} B")
            if self.mcfg["misfp_on_infeasible"] == "raise":
                raise RuntimeError(msg)
            print(msg)
        packet = MixturePacket(packet.space, packet.source, packet.round_idx, classes,
                               packet.precision, packet.meta)
        return packet, rep.summary(classes)

    def _send(self, packet, route):
        rx, nbytes = transmit(packet)
        assert nbytes == packet_nbytes(packet)
        transport = MixtureTransport(rx, nbytes)
        # The inherited loop adds transport.tensor_nbytes(); add the remainder so
        # the route total equals the exact serialized size.
        delta = nbytes - transport.tensor_nbytes()
        add_communication(self.comm_tracker, route, mb=delta / MB)
        key = "client_to_edge" if route == "client_to_edge_MB" else "edge_to_cloud"
        self.totals[f"{key}_payload_bytes"] += nbytes
        self.totals[f"{key}_packets"] += 1
        self.totals[f"{key}_components"] += rx.num_components()
        return transport

    def _round_snapshot(self):
        """Frozen round-specific shared encoder (diagnostic extraction control)."""
        if self._snapshot_round != self._current_round:
            t0 = time.perf_counter()
            self._snapshot = global_client_model(self).to(self.device).eval()
            self._snapshot_round = self._current_round
            self.totals["time_snapshot_s"] += time.perf_counter() - t0
        return self._snapshot

    @torch.no_grad()
    def _extract(self, model, dataset):
        """Deterministic full pass (no shuffle, no drop_last, no global RNG use)."""
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0,
                            generator=torch.Generator())
        feats, labels = [], []
        model.eval()
        with _autocast(self.device):
            for data, target in loader:
                out = model(data.to(self.device, non_blocking=True)).float()
                feats.append(out.cpu())
                labels.append(target.cpu())
        X = torch.cat(feats)
        return X, torch.cat(labels)

    # ---------------------------------------------------------------- client
    def _client_ssl_extraction_phase(self, cid, loader, ssl_epochs):
        self._ensure_round()
        snapshot = None
        if self.mcfg["misfp_extraction"] == "shared_snapshot":
            snapshot = self._round_snapshot()  # built before any client trains this round
        # Inherited H-SFP local training (+ its own extraction, which is discarded;
        # it keeps the global RNG stream identical to H-SFP's client phase).
        _baseline_packet, _ = super()._client_ssl_extraction_phase(cid, loader, ssl_epochs)
        model = snapshot if snapshot is not None else (
            self._work_client if getattr(self, "client_sync", False) else self.structure[-1][cid])
        if snapshot is not None:
            mb = get_model_size_MB(snapshot.state_dict())
            add_communication(self.comm_tracker, "edge_to_client_MB", mb=mb)
            self.totals["snapshot_broadcast_MB"] += mb

        t0 = time.perf_counter()
        X, y = self._extract(model, loader.dataset)
        self.totals["time_feature_extraction_s"] += time.perf_counter() - t0
        shape = tuple(X.shape[1:])
        Xf = X.reshape(X.shape[0], -1).double()

        t0 = time.perf_counter()
        classes, support = {}, {}
        for c in sorted(int(v) for v in torch.unique(y)):
            gen = make_generator(self.seed, "client_fit", self._current_round, int(cid), c)
            cm, info = fit_class_mixture(Xf[y == c], c, self.fit_cfg, gen)
            info["client"] = int(cid)
            self._round["client_fits"].append(info)
            if cm is not None:
                classes[c] = cm
                support[c] = int(round(cm.total))
        self.totals["time_local_fit_s"] += time.perf_counter() - t0
        self._round["selected_clients"].append(int(cid))
        if not classes:
            return None, {}

        t0 = time.perf_counter()
        packet = MixturePacket(self._space("client_to_edge", Xf.shape[1], shape),
                               f"client_{int(cid)}", self._current_round, classes, self.precision)
        packet, rep = self._compress_packet(packet, None, self.mcfg["_client_budget"])
        rep["client"] = int(cid)
        self._round["client_compress"].append(rep)
        self.totals["time_client_compress_s"] += time.perf_counter() - t0
        return self._send(packet, "client_to_edge_MB"), support

    # ------------------------------------------------------------------ edge
    def _train_edge_ssl(self, model, eid, features, labels, ssl_epochs):
        """The H-SFP edge SupCon loop (E-HSFP PRC / partial execution excluded)."""
        opt = self.optimizers[0][eid]
        optimizer, scaler = opt["optimizer"], opt["scaler"]
        ds = torch.utils.data.TensorDataset(features.cpu(), labels.cpu())
        loader = DataLoader(ds, batch_size=self.args["local_bs"], shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
        model.to(self.device).train()
        total_loss = 0.0
        for _ in range(ssl_epochs):
            total_loss = 0.0
            for f, l in loader:
                f, l = f.to(self.device), l.to(self.device)
                if self.ssl_transforms_edge is None:
                    self.ssl_transforms_edge = build_edge_ssl_transforms(f.shape[-1]).to(self.device)
                with _autocast(self.device):
                    z1 = model(self.ssl_transforms_edge(f))
                    z2 = model(self.ssl_transforms_edge(f))
                    z1 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    loss = supervised_contrastive_loss(torch.cat([z1, z2]), torch.cat([l, l]),
                                                       temperature=self.supcon_temp)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
        n_batches = max(len(loader), 1)
        print(f"Edge {eid}: SSL done, final loss: {total_loss / n_batches:.4f}")

    @torch.no_grad()
    def _edge_forward(self, model, X, shape):
        model.eval()
        outs = []
        with _autocast(self.device):
            for s in range(0, X.shape[0], 1024):
                xb = X[s:s + 1024].float().reshape(-1, *shape).to(self.device)
                o = model(xb)
                outs.append(torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(o), start_dim=1).float().cpu())
        return torch.cat(outs).double()

    def _edge_ssl_extraction_phase(self, eid, cids, client_outputs, ssl_epochs, syn_samples_per_class):
        self._ensure_round()
        model = self.structure[0][eid]
        packets = [client_outputs[c].packet for c in cids
                   if client_outputs.get(c) is not None]
        if not packets:
            print(f"Edge {eid}: no prototypes from clients, skipping.")
            return None, {}
        log = {"edge": int(eid), "sources": len(packets)}
        self._round["dispersion"].append({"edge": int(eid), **between_source_dispersion(packets)})

        # 1. pool (never by component id) + optional edge-input cap
        t0 = time.perf_counter()
        if self.mcfg["misfp_pooling"] == "baseline_average":
            per_class = {}
            for p in packets:
                for c, cm in p.classes.items():
                    per_class.setdefault(c, []).append(cm)
            pooled = {c: collapse_baseline_average(v) for c, v in sorted(per_class.items())}
        else:
            pooled = pool_packets(packets)
        for c, cm in pooled.items():  # keep for the held-out diagnostic
            self._last_client_pool.setdefault(c, []).append(cm)
        cap = self.mcfg["misfp_edge_pool_cap"]
        safety = self.mcfg["misfp_edge_pool_safety_cap"]
        cap = safety if cap is None else min(int(cap), int(safety))
        pooled, rep_in = compress_classes(pooled, cap)
        log["input"] = rep_in.summary(pooled)
        self.totals["time_edge_merge_s"] += time.perf_counter() - t0
        space_in = packets[0].space

        # 2. synthesize training inputs (balanced class prior, within-class pi)
        t0 = time.perf_counter()
        gen = make_generator(self.seed, "edge_train", self._current_round, int(eid))
        feats, labels, _ = synthesize(pooled, syn_samples_per_class, gen, self.var_floor,
                                      space_in.feature_shape)
        self.totals["time_edge_synthesis_s"] += time.perf_counter() - t0
        self.totals["synthetic_samples_edge"] += int(feats.shape[0])
        print(f"Edge {eid}: generated {feats.shape[0]} L1 samples from "
              f"{sum(cm.R for cm in pooled.values())} components. Starting SSL...")

        # 3. edge SupCon training (H-SFP loop)
        self._train_edge_ssl(model, eid, feats, labels, ssl_epochs)
        del feats, labels

        # 4. re-estimate statistics in the edge OUTPUT space. Component mass is
        # carried from the underlying data support, never from MC sample counts.
        t0 = time.perf_counter()
        out_classes = {}
        mode = self.mcfg["misfp_boundary_mode"]
        gen_b = make_generator(self.seed, "edge_boundary", self._current_round, int(eid))
        refits = []
        for c, cm in sorted(pooled.items()):
            if mode == "propagate":
                m = int(self.mcfg["misfp_boundary_samples_per_component"])
                Xin = torch.cat([sample_component(cm, r, m, gen_b, self.var_floor)
                                 for r in range(cm.R)])
                Z = self._edge_forward(model, Xin, space_in.feature_shape)
                self.totals["boundary_mc_samples"] += int(Z.shape[0])
                Z = Z.reshape(cm.R, m, -1)
                mu = Z.mean(1)
                var = ((Z - mu[:, None, :]) ** 2).mean(1)
                out_classes[c] = ClassMixture(c, cm.counts.clone(), mu, var)
            else:  # refit: plain draws from the target mixture, k-means in output space
                M = int(self.mcfg["misfp_boundary_samples_per_class"])
                from misfp import sample_mixture
                Xin, _ = sample_mixture(cm, M, gen_b, self.var_floor)
                Z = self._edge_forward(model, Xin, space_in.feature_shape)
                self.totals["boundary_mc_samples"] += int(Z.shape[0])
                g = make_generator(self.seed, "edge_refit", self._current_round, int(eid), c)
                fit, info = fit_class_mixture(Z, c, self.refit_cfg, g)
                refits.append(info["k_selected"])
                # MC estimate of component mass: assignment fraction x class support.
                out_classes[c] = ClassMixture(c, fit.counts / fit.counts.sum() * cm.total,
                                              fit.means, fit.variances)
        if refits:
            log["refit_mean_k"] = float(np.mean(refits))
        self.totals["time_edge_reestimate_s"] += time.perf_counter() - t0
        out_dim = next(iter(out_classes.values())).d
        if self.input_shape_cloud is None:
            self.input_shape_cloud = torch.Size([out_dim])

        # 5. compress to the edge->cloud cap / budget and transmit
        t0 = time.perf_counter()
        packet = MixturePacket(self._space("edge_to_cloud", out_dim, (out_dim,)), f"edge_{int(eid)}",
                               self._current_round, out_classes, self.precision)
        packet, rep_out = self._compress_packet(packet, self.mcfg["misfp_edge_out_cap"],
                                                self.mcfg["_edge_budget"])
        log["output"] = rep_out
        self.totals["time_edge_merge_s"] += time.perf_counter() - t0
        self._round["edge"].append(log)
        support = {c: int(round(cm.total)) for c, cm in packet.classes.items()}
        return self._send(packet, "edge_to_cloud_MB"), support

    # ----------------------------------------------------------------- cloud
    def _cloud_supervised_phase(self, cloud_id, edge_outputs, syn_epochs, syn_samples_per_class):
        self._ensure_round()
        model = self.structure[len(self.args["mid_server"])][cloud_id]
        opt = self.optimizers[len(self.args["mid_server"])][cloud_id]
        optimizer, scaler = opt["optimizer"], opt["scaler"]
        packets = [v.packet for v in edge_outputs.values() if v is not None]
        if not packets:
            print(f"Cloud {cloud_id}: no prototypes from edges, skipping.")
            return 0.0
        t0 = time.perf_counter()
        pooled = pool_packets(packets)
        pooled, rep = compress_classes(pooled, self.mcfg["misfp_cloud_pool_cap"])
        self.totals["time_cloud_merge_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        gen = make_generator(self.seed, "cloud_train", self._current_round, int(cloud_id))
        feats, labels, _ = synthesize(pooled, syn_samples_per_class, gen, self.var_floor,
                                      packets[0].space.feature_shape)
        self.totals["time_cloud_synthesis_s"] += time.perf_counter() - t0
        self.totals["synthetic_samples_cloud"] += int(feats.shape[0])
        self._round["cloud"] = rep.summary(pooled)
        print(f"Cloud {cloud_id}: generated {feats.shape[0]} L2 samples from "
              f"{sum(cm.R for cm in pooled.values())} components. Starting supervised...")

        ds = torch.utils.data.TensorDataset(feats.cpu(), labels.cpu())
        loader = DataLoader(ds, batch_size=self.args["local_bs"], shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
        model.to(self.device).train()
        total_loss = 0.0
        for _ in range(syn_epochs):
            epoch_loss = 0.0
            for f, l in loader:
                f, l = f.to(self.device), l.to(self.device, non_blocking=True)
                with _autocast(self.device):
                    loss = self.criterion(model(f), l)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            total_loss = epoch_loss / max(len(loader), 1)
        print(f"Cloud {cloud_id}: supervised done, final loss: {total_loss:.4f}")
        return total_loss

    # ------------------------------------------------- per-round log + diag
    def _run_validation(self, valid_dataset, best_val_top1, epoch):
        out = super()._run_validation(valid_dataset, best_val_top1, epoch)
        self._flush_round(epoch)
        if (self.mcfg["misfp_heldout_diagnostic"] and epoch == self._sched_total_epochs
                and self._last_client_pool):
            try:
                self._heldout_diagnostic(valid_dataset)
            except Exception as exc:  # diagnostic must never kill a run
                print(f"[MiSFP] held-out diagnostic failed: {exc!r}")
        return out

    def _flush_round(self, epoch):
        if self.log_path is None or self._round is None:
            return
        fits = self._round["client_fits"]
        ks = [f["k_selected"] for f in fits if f["k_selected"] > 0]
        fallbacks = {}
        for f in fits:
            if f["fallback"]:
                fallbacks[f["fallback"]] = fallbacks.get(f["fallback"], 0) + 1
        rec = {
            "round": epoch,
            "selected_clients": self._round["selected_clients"],
            "client_fit_summary": {
                "class_fits": len(fits),
                "mean_k": float(np.mean(ks)) if ks else None,
                "k_hist": {int(k): int(v) for k, v in zip(*np.unique(ks, return_counts=True))} if ks else {},
                "fallbacks": fallbacks,
                "fit_modes": sorted({f["fit_mode"] for f in fits}),
                "mean_class_n": float(np.mean([f["n"] for f in fits])) if fits else None,
            },
            "client_compress": self._round["client_compress"],
            "edge": self._round["edge"],
            "cloud": self._round["cloud"],
            "dispersion": self._round["dispersion"],
            "totals": dict(self.totals),
            "comm_tracker": dict(self.comm_tracker),
            "rss_bytes": psutil.Process().memory_info().rss,
            "cuda_max_allocated_bytes": (torch.cuda.max_memory_allocated()
                                         if torch.cuda.is_available() else 0),
        }
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    @torch.no_grad()
    def _heldout_diagnostic(self, valid_dataset):
        """Within-class held-out discrepancy at the client->edge boundary.

        DIAGNOSTIC ONLY, extra information access: uses held-out (validation)
        features/labels, never feeds training. Compares the last round's pooled
        received client mixture with its moment-collapsed single Gaussian. In the
        legacy local-encoder mode the packets came from per-client encoders while
        these features come from the averaged global encoder, so absolute values
        mix model mismatch with distribution fit."""
        enc = (self._snapshot if self.mcfg["misfp_extraction"] == "shared_snapshot"
               and self._snapshot is not None else global_client_model(self).to(self.device))
        X, y = self._extract(enc, valid_dataset)
        X = X.reshape(X.shape[0], -1).double()
        rows, pca_payload = [], {}
        for c, parts in sorted(self._last_client_pool.items()):
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                continue
            from misfp.packets import concat_mixtures
            mix = concat_mixtures(parts)
            single = mix.collapse()
            floor = max(float(self.mcfg["misfp_nll_abs_var_floor"]),
                        float(self.mcfg["misfp_nll_rel_var_floor"]) * float(single.variances.mean()))
            d = X.shape[1]
            rows.append({"class_id": int(c), "n_val": int(Xc.shape[0]), "components": mix.R,
                         "nll_mixture_per_dim": float(-mixture_log_prob(Xc, mix, floor).mean()) / d,
                         "nll_single_per_dim": float(-mixture_log_prob(Xc, single, floor).mean()) / d})
            pca_payload[int(c)] = (Xc.float().numpy(), mix.counts.numpy(), mix.means.numpy(),
                                   mix.variances.numpy())
        out_dir = os.path.dirname(self.log_path)
        with open(os.path.join(out_dir, "misfp_heldout_diagnostic.json"), "w") as fh:
            json.dump({"note": "diagnostic only; uses held-out validation features",
                       "extraction": self.mcfg["misfp_extraction"], "per_class": rows,
                       "mean_nll_mixture_per_dim": float(np.mean([r["nll_mixture_per_dim"] for r in rows])) if rows else None,
                       "mean_nll_single_per_dim": float(np.mean([r["nll_single_per_dim"] for r in rows])) if rows else None},
                      fh, indent=2)
        top = sorted(pca_payload, key=lambda c: -len(pca_payload[c][1]))[: int(self.mcfg["misfp_diag_pca_classes"])]
        np.savez_compressed(os.path.join(out_dir, "misfp_pca_payload.npz"),
                            **{f"c{c}_{k}": v for c in top
                               for k, v in zip(("real", "counts", "means", "vars"), pca_payload[c])})
