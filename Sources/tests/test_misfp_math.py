"""Mathematical correctness tests for MiSFP (misfp/)."""

import importlib.util
import io
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from misfp import (ClassMixture, FeatureSpace, FeatureSpaceMismatch, FitConfig, MixturePacket,
                   MixtureTransport, collapse_baseline_average, component_nbytes, compress_classes,
                   deserialize_packet, fit_class_mixture, get_misfp_config, make_generator,
                   merge_pair, mixture_log_prob, packet_nbytes, pool_packets, pool_two,
                   sample_mixture, serialize_packet, synthesize, transmit)
from misfp.merge import greedy_merge_to_cap
from misfp.packets import CLASS_OVERHEAD_BYTES

D = torch.float64


def _moments(X):
    return X.shape[0], X.mean(0), X.var(0, unbiased=False)


def _cm_from_data(c, chunks):
    stats = [_moments(x) for x in chunks]
    return ClassMixture(c, torch.tensor([float(s[0]) for s in stats]),
                        torch.stack([s[1] for s in stats]), torch.stack([s[2] for s in stats]))


def _space(dim, boundary="client_to_edge", sid="client_to_edge/local"):
    return FeatureSpace(boundary, dim, (dim,), sid)


class MomentPreservation(unittest.TestCase):
    def test_pool_two_matches_concatenated_data(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(37, 5, generator=g, dtype=D) * 2 + 1
        b = torch.randn(11, 5, generator=g, dtype=D) * 0.3 - 4
        n, mu, v = pool_two(*_moments(a), *_moments(b))
        n_ref, mu_ref, v_ref = _moments(torch.cat([a, b]))
        self.assertEqual(n, n_ref)
        torch.testing.assert_close(mu, mu_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(v, v_ref, rtol=1e-12, atol=1e-12)

    def test_any_merge_sequence_preserves_count_mean_second_moment(self):
        g = torch.Generator().manual_seed(1)
        chunks = [torch.randn(int(n), 4, generator=g, dtype=D) + 3 * k
                  for k, n in enumerate([5, 9, 3, 20, 7, 1])]
        cm = _cm_from_data(7, chunks)
        n_ref, mu_ref, v_ref = _moments(torch.cat(chunks))
        for cap in (5, 3, 2, 1):
            out = greedy_merge_to_cap(cm, cap)
            self.assertEqual(out.R, cap)
            self.assertAlmostEqual(out.total, n_ref, places=10)
            torch.testing.assert_close(out.mixture_mean(), mu_ref, rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(out.mixture_variance(), v_ref, rtol=1e-11, atol=1e-11)
        one = greedy_merge_to_cap(cm, 1)
        torch.testing.assert_close(one.means[0], mu_ref, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(one.variances[0], v_ref, rtol=1e-11, atol=1e-11)

    def test_collapse_equals_cap_one(self):
        g = torch.Generator().manual_seed(2)
        cm = _cm_from_data(0, [torch.randn(6, 3, generator=g, dtype=D) + i for i in range(4)])
        a, b = cm.collapse(), greedy_merge_to_cap(cm, 1)
        torch.testing.assert_close(a.means, b.means)
        torch.testing.assert_close(a.variances, b.variances)

    def test_streamed_fit_moments_equal_direct_cluster_moments(self):
        g = torch.Generator().manual_seed(3)
        X = torch.cat([torch.randn(40, 3, generator=g, dtype=D) - 5,
                       torch.randn(60, 3, generator=g, dtype=D) + 5])
        cfg = FitConfig(k=2, chunk_size=7)  # force multi-chunk streaming
        cm, info = fit_class_mixture(X, 3, cfg, make_generator(0))
        self.assertEqual(cm.R, 2)
        left = X[:40]  # the negative mode
        idx = int(cm.means[:, 0].argmin())
        torch.testing.assert_close(cm.means[idx], left.mean(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cm.variances[idx], left.var(0, unbiased=False),
                                   rtol=1e-10, atol=1e-12)
        self.assertEqual(sorted(cm.counts.tolist()), [40.0, 60.0])


class Sampling(unittest.TestCase):
    def test_component_proportions_and_moments(self):
        cm = ClassMixture(0, torch.tensor([1.0, 3.0]),
                          torch.tensor([[-2.0, 0.0], [2.0, 1.0]], dtype=D),
                          torch.tensor([[0.01, 0.04], [0.09, 0.25]], dtype=D))
        n = 200_000
        Z, comp = sample_mixture(cm, n, make_generator("s", 1))
        frac = (comp == 1).double().mean().item()
        se = (0.75 * 0.25 / n) ** 0.5
        self.assertLess(abs(frac - 0.75), 5 * se)
        torch.testing.assert_close(Z.mean(0), cm.mixture_mean(), rtol=0, atol=0.01)
        torch.testing.assert_close(Z.var(0, unbiased=False), cm.mixture_variance(), rtol=0.02, atol=0.01)
        for r in range(2):
            Zr = Z[comp == r]
            torch.testing.assert_close(Zr.mean(0), cm.means[r], rtol=0, atol=0.01)
            torch.testing.assert_close(Zr.var(0, unbiased=False), cm.variances[r], rtol=0.03, atol=1e-3)

    def test_synthesize_balanced_class_prior_and_labels(self):
        classes = {3: ClassMixture(3, torch.tensor([1.0]), torch.zeros(1, 2, dtype=D), torch.ones(1, 2, dtype=D)),
                   9: ClassMixture(9, torch.tensor([5.0, 5.0]), torch.ones(2, 2, dtype=D), torch.ones(2, 2, dtype=D))}
        X, y, comp = synthesize(classes, 7, make_generator(1), feature_shape=(2, 1, 1))
        self.assertEqual(tuple(X.shape), (14, 2, 1, 1))
        self.assertEqual(y.tolist(), [3] * 7 + [9] * 7)
        self.assertEqual(X.dtype, torch.float32)

    def test_seeded_sampling_is_reproducible(self):
        cm = ClassMixture(0, torch.tensor([1.0, 2.0]), torch.randn(2, 3, dtype=D), torch.ones(2, 3, dtype=D))
        a = sample_mixture(cm, 50, make_generator("x", 3))[0]
        b = sample_mixture(cm, 50, make_generator("x", 3))[0]
        self.assertTrue(torch.equal(a, b))


class DegenerateInputs(unittest.TestCase):
    def test_absent_class_and_nonfinite_rows(self):
        cfg = FitConfig(k=2)
        cm, info = fit_class_mixture(torch.full((3, 4), float("nan")), 1, cfg, make_generator(0))
        self.assertIsNone(cm)
        self.assertEqual(info["fallback"], "absent")
        X = torch.randn(10, 4, dtype=D)
        X[2, 1] = float("inf")
        cm, info = fit_class_mixture(X, 1, cfg, make_generator(0))
        self.assertEqual(info["dropped_nonfinite"], 1)
        self.assertAlmostEqual(cm.total, 9.0)

    def test_singleton_and_tiny_classes_fall_back_to_k1(self):
        cfg = FitConfig(k=4, min_component_support=3)
        cm, info = fit_class_mixture(torch.ones(1, 5, dtype=D), 0, cfg, make_generator(0))
        self.assertEqual(cm.R, 1)
        self.assertTrue(torch.equal(cm.variances, torch.zeros(1, 5, dtype=D)))
        self.assertEqual(info["fallback"], "insufficient_support")
        cm, info = fit_class_mixture(torch.randn(5, 2, dtype=D), 0, cfg, make_generator(0))
        self.assertEqual(cm.R, 1)  # 5 // 3 = 1

    def test_duplicate_features_limit_k(self):
        X = torch.cat([torch.zeros(10, 3, dtype=D), torch.ones(10, 3, dtype=D)])
        cm, info = fit_class_mixture(X, 0, FitConfig(k=4, min_component_support=2), make_generator(0))
        self.assertEqual(cm.R, 2)  # only two distinct rows
        self.assertTrue(torch.equal(cm.variances, torch.zeros(2, 3, dtype=D)))

    def test_min_support_enforced_for_every_component(self):
        g = torch.Generator().manual_seed(4)
        X = torch.cat([torch.randn(30, 2, generator=g, dtype=D), torch.tensor([[50.0, 50.0]], dtype=D)])
        cm, _ = fit_class_mixture(X, 0, FitConfig(k=3, min_component_support=3), make_generator(0))
        self.assertTrue((cm.counts >= 3).all())

    def test_variance_floor_only_at_evaluation_and_sampling(self):
        cm = ClassMixture(0, torch.tensor([4.0]), torch.zeros(1, 2, dtype=D), torch.zeros(1, 2, dtype=D))
        self.assertTrue(torch.equal(cm.variances, torch.zeros(1, 2, dtype=D)))  # stored unfloored
        lp = mixture_log_prob(torch.zeros(1, 2, dtype=D), cm, 1e-6)
        self.assertTrue(torch.isfinite(lp).all())
        Z, _ = sample_mixture(cm, 10, make_generator(0), var_floor=1e-4)
        self.assertGreater(float(Z.std()), 0)
        merged = merge_pair(ClassMixture(0, torch.tensor([1.0, 1.0]), torch.tensor([[0.0], [2.0]], dtype=D),
                                         torch.zeros(2, 1, dtype=D)), 0, 1)
        self.assertAlmostEqual(float(merged.variances[0, 0]), 1.0)  # exact, no floor leakage

    def test_invalid_mixtures_rejected(self):
        with self.assertRaises(ValueError):
            ClassMixture(0, torch.tensor([1.0]), torch.tensor([[float("nan")]], dtype=D), torch.ones(1, 1, dtype=D))
        with self.assertRaises(ValueError):
            ClassMixture(0, torch.tensor([0.0]), torch.zeros(1, 1, dtype=D), torch.ones(1, 1, dtype=D))
        with self.assertRaises(ValueError):
            ClassMixture(0, torch.tensor([1.0]), torch.zeros(1, 1, dtype=D), -torch.ones(1, 1, dtype=D))


class Budgets(unittest.TestCase):
    def _classes(self, Rs, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        return {c: ClassMixture(c, torch.rand(R, generator=g, dtype=D) + 1,
                                torch.randn(R, d, generator=g, dtype=D) * (c + 1),
                                torch.rand(R, d, generator=g, dtype=D))
                for c, R in Rs.items()}

    def _nbytes_fn(self, d, prec="float32", header=10):
        per = component_nbytes(d, prec)
        return (lambda cl: 8 + header + sum(CLASS_OVERHEAD_BYTES + cm.R * per for cm in cl.values())), per

    def test_budget_enforced_and_classes_kept(self):
        classes = self._classes({0: 4, 1: 3, 2: 1})
        fn, per = self._nbytes_fn(4)
        budget = fn(classes) - 3 * per
        out, rep = compress_classes(classes, None, budget, fn, per)
        self.assertEqual(rep.status, "ok")
        self.assertLessEqual(fn(out), budget)
        self.assertEqual(sorted(out), [0, 1, 2])
        self.assertEqual(rep.components_after, 5)

    def test_infeasible_budget_reported_not_dropped(self):
        classes = self._classes({0: 3, 1: 2})
        fn, per = self._nbytes_fn(4)
        out, rep = compress_classes(classes, None, 10, fn, per)
        self.assertEqual(rep.status, "infeasible")
        self.assertEqual(sorted(out), [0, 1])
        self.assertTrue(all(cm.R == 1 for cm in out.values()))

    def test_global_budget_merges_cheapest_pair_first(self):
        far = ClassMixture(0, torch.tensor([10.0, 10.0]), torch.tensor([[-5.0], [5.0]], dtype=D), torch.ones(2, 1, dtype=D))
        near = ClassMixture(1, torch.tensor([10.0, 10.0]), torch.tensor([[0.0], [0.1]], dtype=D), torch.ones(2, 1, dtype=D))
        fn, per = self._nbytes_fn(1)
        out, rep = compress_classes({0: far, 1: near}, None, fn({0: far, 1: near}) - per, fn, per)
        self.assertEqual(out[0].R, 2)
        self.assertEqual(out[1].R, 1)

    def test_deterministic_tie_breaking(self):
        # Two identical classes: a tie between classes resolves to the lower class id.
        cm = ClassMixture(0, torch.tensor([1.0, 1.0]), torch.tensor([[0.0], [1.0]], dtype=D), torch.ones(2, 1, dtype=D))
        cm1 = ClassMixture(1, cm.counts, cm.means, cm.variances)
        fn, per = self._nbytes_fn(1)
        for _ in range(3):
            out, _ = compress_classes({1: cm1, 0: cm}, None, fn({0: cm, 1: cm1}) - per, fn, per)
            self.assertEqual((out[0].R, out[1].R), (1, 2))

    def test_permutation_and_component_id_invariance(self):
        cm = self._classes({0: 6})[0]
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        shuffled = ClassMixture(0, cm.counts[perm], cm.means[perm], cm.variances[perm],
                                torch.tensor([90, 17, 4, 55, 3, 8]))
        for cap in (4, 2, 1):
            a, b = greedy_merge_to_cap(cm, cap), greedy_merge_to_cap(shuffled, cap)
            torch.testing.assert_close(a.counts, b.counts)
            torch.testing.assert_close(a.means, b.means)
            torch.testing.assert_close(a.variances, b.variances)


class SerializationAndCompat(unittest.TestCase):
    def _packet(self, prec="float32", d=6):
        g = torch.Generator().manual_seed(5)
        classes = {c: ClassMixture(c, torch.tensor([3.0, 1.5])[: R], torch.randn(R, d, generator=g, dtype=D),
                                   torch.rand(R, d, generator=g, dtype=D))
                   for c, R in ((0, 2), (4, 1))}
        return MixturePacket(_space(d), "client_7", 3, classes, prec)

    def test_roundtrip_and_exact_byte_count(self):
        for prec in ("float32", "float16"):
            p = self._packet(prec)
            blob = serialize_packet(p)
            self.assertEqual(len(blob), packet_nbytes(p))
            q = deserialize_packet(blob)
            self.assertEqual(q.space, p.space)
            self.assertEqual(sorted(q.classes), [0, 4])
            tol = 1e-6 if prec == "float32" else 2e-3
            for c in p.classes:
                torch.testing.assert_close(q.classes[c].means, p.classes[c].means, rtol=tol, atol=tol)
                torch.testing.assert_close(q.classes[c].counts, p.classes[c].counts)
            # the transmitted (dequantized) packet re-serializes bit-identically
            self.assertEqual(serialize_packet(q), blob)

    def test_fp16_halves_moment_bytes(self):
        self.assertEqual(component_nbytes(128, "float32") - 8, 2 * (component_nbytes(128, "float16") - 8))

    def test_transport_accounting_equals_serialized_bytes(self):
        rx, n = transmit(self._packet())
        t = MixtureTransport(rx, n)
        self.assertLess(t.tensor_nbytes(), n)
        protos, stds = t
        self.assertEqual(tuple(protos[0].shape), (2, 6))
        torch.testing.assert_close(stds[0].double() ** 2, rx.classes[0].variances, rtol=1e-6, atol=1e-7)

    def test_checkpoint_resume_reproducibility(self):
        """Packets + generator state saved mid-stream reproduce later synthesis exactly."""
        p, _ = transmit(self._packet())  # a node's state is the RECEIVED packet
        g = make_generator("resume", 1)
        synthesize(p.classes, 5, g)  # consume some stream
        buf = io.BytesIO()
        torch.save({"packet": serialize_packet(p), "gen": g.get_state()}, buf)
        expect = synthesize(p.classes, 9, g)[0]
        buf.seek(0)
        ck = torch.load(buf)
        g2 = torch.Generator()
        g2.set_state(ck["gen"])
        got = synthesize(deserialize_packet(ck["packet"]).classes, 9, g2)[0]
        self.assertTrue(torch.equal(expect, got))

    def test_incompatible_spaces_refused(self):
        a = self._packet()
        b = MixturePacket(_space(6, sid="client_to_edge/snapshot@r3"), "client_1", 3, a.classes)
        with self.assertRaises(FeatureSpaceMismatch):
            pool_packets([a, b])
        c = MixturePacket(_space(5), "client_2", 3,
                          {0: ClassMixture(0, torch.tensor([1.0]), torch.zeros(1, 5, dtype=D), torch.ones(1, 5, dtype=D))})
        with self.assertRaises(FeatureSpaceMismatch):
            pool_packets([a, c])
        with self.assertRaises(FeatureSpaceMismatch):
            MixturePacket(_space(5), "x", 0, a.classes)

    def test_pooling_keeps_all_same_class_components(self):
        a, b = self._packet(), self._packet()
        pooled = pool_packets([a, b])
        self.assertEqual(pooled[0].R, 4)
        self.assertEqual(pooled[4].R, 2)


def _load_hsfp_hierarchy():
    """Import H-SFP's hierarchy.py as a standalone module (no sys.path changes)."""
    sys.path.insert(0, os.path.join(ROOT, "classification", "H-SFP"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_hsfp_hierarchy", os.path.join(ROOT, "classification", "H-SFP", "hierarchy.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


class BaselineAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = _load_hsfp_hierarchy()

    def test_k1_client_packet_equals_hsfp_mean_and_population_std(self):
        g = torch.Generator().manual_seed(6)
        X = torch.randn(40, 8, generator=g)
        y = torch.randint(0, 3, (40,), generator=g)
        protos, stds = self.h.calculate_prototypes_and_distribution(X, y)
        for c in protos:
            cm, _ = fit_class_mixture(X[y == c], c, FitConfig(k=1), make_generator(0))
            torch.testing.assert_close(cm.means[0].float(), protos[c], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(cm.variances[0].sqrt().float(), stds[c], rtol=1e-5, atol=1e-6)

    def test_baseline_average_pooling_reproduces_hsfp_aggregation(self):
        g = torch.Generator().manual_seed(7)
        outs, per_source = {}, {}
        for s, n in enumerate([3, 10, 25]):
            X = torch.randn(n, 4, generator=g) + s
            mu, sd = X.mean(0), X.std(0, unbiased=False)
            outs[s] = ({0: mu}, {0: sd})
            per_source[s] = ClassMixture(0, torch.tensor([float(n)]), mu[None].double(), (sd ** 2)[None].double())
        # H-SFP aggregation_mode="average" (hierarchy.py / ehsfp.aggregation):
        ref_mu = torch.stack([outs[s][0][0] for s in outs]).mean(0)
        ref_sd = torch.sqrt(torch.stack([outs[s][1][0] ** 2 for s in outs]).mean(0))
        got = collapse_baseline_average(list(per_source.values()))
        torch.testing.assert_close(got.means[0].float(), ref_mu, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(got.variances[0].sqrt().float(), ref_sd, rtol=1e-6, atol=1e-6)
        # ...and it differs from exact moment pooling whenever sources disagree,
        # which is the documented H-SFP vs MiSFP-K1 difference.
        exact = greedy_merge_to_cap(ClassMixture(0, torch.cat([m.counts for m in per_source.values()]),
                                                 torch.cat([m.means for m in per_source.values()]),
                                                 torch.cat([m.variances for m in per_source.values()])), 1)
        self.assertGreater(float((exact.variances - got.variances).abs().max()), 1e-3)


class ConfigResolution(unittest.TestCase):
    def test_variants_and_ehsfp_guard(self):
        self.assertEqual(get_misfp_config({"misfp_variant": "k2"})["misfp_local_k"], 2)
        self.assertEqual(get_misfp_config({"misfp_variant": "adaptive"})["misfp_local_k_mode"], "adaptive")
        self.assertEqual(get_misfp_config({"misfp_variant": "edge"})["misfp_edge_out_cap"], 4)
        with self.assertRaises(ValueError):
            get_misfp_config({"misfp_variant": "k2"}, {"use_episodic_memory": True})
        with self.assertRaises(ValueError):
            get_misfp_config({"misfp_variant": "k2", "misfp_pooling": "baseline_average"})
        with self.assertRaises(ValueError):
            get_misfp_config({"misfp_variant": "mc-hsfp"})
        cfg = get_misfp_config({"misfp_variant": "k2", "misfp_client_budget": "match_k1_fp32:1.5"})
        self.assertEqual(cfg["_client_budget"], ("match_k1_fp32", 1.5))


class AdaptiveSelection(unittest.TestCase):
    def test_bimodal_selects_two_unimodal_selects_one(self):
        g = torch.Generator().manual_seed(8)
        bi = torch.cat([torch.randn(60, 2, generator=g, dtype=D) * 0.1 - 2,
                        torch.randn(60, 2, generator=g, dtype=D) * 0.1 + 2])
        uni = torch.randn(120, 2, generator=g, dtype=D)
        cfg = FitConfig(k_mode="adaptive", k=4)
        cm_bi, info_bi = fit_class_mixture(bi, 0, cfg, make_generator(1))
        cm_uni, info_uni = fit_class_mixture(uni, 0, cfg, make_generator(1))
        self.assertGreaterEqual(cm_bi.R, 2)
        self.assertEqual(cm_uni.R, 1)
        self.assertIsNotNone(info_bi["nll_val"])

    def test_insufficient_validation_support_falls_back(self):
        cm, info = fit_class_mixture(torch.randn(6, 2, dtype=D), 0, FitConfig(k_mode="adaptive", k=4),
                                     make_generator(0))
        self.assertEqual(cm.R, 1)
        self.assertEqual(info["fallback"], "insufficient_validation_support")


if __name__ == "__main__":
    unittest.main()
