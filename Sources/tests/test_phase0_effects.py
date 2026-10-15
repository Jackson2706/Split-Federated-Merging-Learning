import json
import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from ehsfp import (
    ABLATION_PRESETS, EHSFP_DEFAULTS, EpisodicPrototypeMemory,
    PrototypeDropout, PrototypeRecord, PrototypeReliabilityNetwork,
    RuntimeCounters, build_reliability_features, get_ehsfp_config,
    mix_current_and_memory, prepare_run_dir,
    prototype_replay_consistency_loss, reliability_weighted_aggregate,
    resolved_config_hash, validate_checkpoint_config_hash,
)
from ehsfp import aggregation as aggregation_module
from ehsfp import losses as losses_module
from ehsfp import memory as memory_module
from ehsfp import reliability as reliability_module


def _prc_gradient(memory_mu):
    representation = torch.nn.Linear(2, 1, bias=False)
    representation.weight.data.copy_(torch.tensor([[1.0, -0.5]]))
    counters = RuntimeCounters()
    loss = prototype_replay_consistency_loss(
        {0: torch.tensor([1.0, 2.0])}, {0: torch.zeros(2)},
        {0: torch.tensor(memory_mu)}, {0: torch.zeros(2)},
        representation, num_samples=3, device=torch.device("cpu"),
        runtime_counters=counters,
    )
    loss.backward()
    return representation.weight.grad.clone(), loss.item(), counters.snapshot()


class Phase0EffectTests(unittest.TestCase):
    def test_classification_prototype_std_is_reduced_in_fp32(self):
        # This magnitude is representable in fp16, but its squared deviation is
        # not.  It models the large autocast ResNet activations observed in the
        # classification client extraction path.
        activations = torch.tensor(
            [[400.0, -400.0], [-400.0, 400.0]], dtype=torch.float16,
        )
        # CPU's std kernel may internally promote half inputs, unlike the CUDA
        # path at issue.  Spell out the fp16 variance operation to reproduce the
        # underlying overflow on CPU.
        unsafe_variance = (
            activations - activations.mean(0)
        ).square().mean(0)
        safe_sigma = activations.float().std(0, unbiased=False)
        self.assertFalse(torch.isfinite(unsafe_variance).all())
        self.assertTrue(torch.isfinite(safe_sigma).all())

        record = PrototypeRecord(
            0, activations.float().mean(0), safe_sigma, 2, "client_0", 0,
        )
        features = build_reliability_features([record], record.mu)
        self.assertTrue(torch.isfinite(features).all())

        # CPU CI cannot execute CUDA autocast, so pin the production boundary
        # responsible for ensuring the tensors reaching the reduction are fp32.
        hierarchy = (
            Path(__file__).resolve().parents[1]
            / "classification" / "H-SFP" / "hierarchy.py"
        ).read_text()
        self.assertIn("out = model(data).float()", hierarchy)

    def test_memory_consumers_coerce_cpu_records_to_compute_device(self):
        # Resolve to an indexed device (e.g. cuda:0) so equality with a
        # tensor's .device holds on GPU hosts (torch.device('cuda') != 'cuda:0').
        target = torch.empty(0, device="cuda" if torch.cuda.is_available() else "cpu").device
        record = PrototypeRecord(
            0, torch.tensor([1.0, 2.0]), torch.tensor([0.2, 0.4]),
            3, "cpu-memory", 0,
        )
        center = torch.tensor([1.5, 2.5], device=target)

        features = build_reliability_features([record], center, device=target)
        self.assertEqual(features.device, target)

        memory = EpisodicPrototypeMemory(max_size=2, max_age=2)
        memory.add_from_proto_dicts(
            {0: record.mu}, {0: record.sigma}, "cpu-memory", round_idx=0,
        )
        memory_protos, memory_stds = memory.to_proto_dist_dicts()
        network = PrototypeReliabilityNetwork(hidden_dim=2).to(target)
        synthetic, labels = reliability_weighted_aggregate(
            {"memory": (memory_protos, memory_stds)},
            num_samples_per_class=2,
            device=target,
            reliability_net=network,
            memory=memory,
        )
        self.assertEqual(synthetic.device, target)
        self.assertEqual(labels.device, target)
        self.assertEqual(memory.get_recent()[0].mu.device.type, "cpu")
        self.assertEqual(memory.get_recent()[0].sigma.device.type, "cpu")

        mixed_protos, mixed_stds = mix_current_and_memory(
            {0: torch.tensor([2.0, 3.0])},
            {0: torch.tensor([0.3, 0.5])},
            memory,
            alpha=0.5,
            device=target,
        )
        self.assertEqual(mixed_protos[0].device, target)
        self.assertEqual(mixed_stds[0].device, target)

        representation = torch.nn.Linear(2, 2).to(target)
        prc = prototype_replay_consistency_loss(
            {0: torch.tensor([2.0, 3.0])},
            {0: torch.tensor([0.3, 0.5])},
            memory_protos,
            memory_stds,
            representation,
            num_samples=2,
            device=target,
        )
        self.assertEqual(prc.device, target)

        # CPU-only CI cannot exercise a CUDA transfer, so retain a structural
        # guard that each audited consumption boundary explicitly copies.
        self.assertIn(".to(device=device", inspect.getsource(
            reliability_module.build_reliability_features,
        ))
        self.assertIn(".to(device)", inspect.getsource(
            aggregation_module.reliability_weighted_aggregate,
        ))
        self.assertIn(".to(compute_device)", inspect.getsource(
            memory_module.mix_current_and_memory,
        ))
        self.assertIn(".to(compute_device)", inspect.getsource(
            losses_module.prototype_replay_consistency_loss,
        ))

    def test_memory_replay_changes_prototype(self):
        memory = EpisodicPrototypeMemory(max_size=4, max_age=2)
        memory.add(PrototypeRecord(
            0, torch.tensor([10.0, 12.0]), torch.ones(2), 4, "old", 0,
        ))
        counters = RuntimeCounters()
        current_p = {0: torch.tensor([0.0, 2.0])}
        current_d = {0: torch.ones(2)}
        mixed_p, _ = mix_current_and_memory(
            current_p, current_d, memory, alpha=0.5,
            runtime_counters=counters,
        )
        self.assertTrue(torch.equal(mixed_p[0], torch.tensor([5.0, 7.0])))
        self.assertEqual(counters.snapshot()["memory.changed_classes"], 1)

    def test_reliability_weights_are_nonuniform_for_unequal_inputs(self):
        net = PrototypeReliabilityNetwork(hidden_dim=2)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.net[0].weight[0, 0] = 1.0
            net.net[2].weight[0, 0] = 1.0
            net.net[4].weight[0, 0] = 1.0
        features = torch.zeros(2, 6)
        features[1, 0] = 1.0
        weights = net(features)
        self.assertGreater(weights[1], weights[0])
        self.assertFalse(torch.allclose(weights, weights[0].expand_as(weights)))

    def test_prc_changes_trainable_gradient_when_inputs_differ(self):
        same_grad, same_loss, _ = _prc_gradient([1.0, 2.0])
        changed_grad, changed_loss, counters = _prc_gradient([3.0, -1.0])
        self.assertEqual(same_loss, 0.0)
        self.assertTrue(torch.equal(same_grad, torch.zeros_like(same_grad)))
        self.assertGreater(changed_loss, 0.0)
        self.assertFalse(torch.equal(changed_grad, same_grad))
        self.assertEqual(counters["prc.shared_classes"], 1)
        self.assertEqual(counters["prc.nonzero_loss_calls"], 1)

    def test_prc_batches_classes_into_two_representation_forwards(self):
        class CountingRepresentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return self.linear(inputs)

        representation = CountingRepresentation()
        current = {class_id: torch.full((2,), float(class_id)) for class_id in range(7)}
        memory = {class_id: value + 1 for class_id, value in current.items()}
        stds = {class_id: torch.zeros(2) for class_id in current}
        loss = prototype_replay_consistency_loss(
            current, stds, memory, stds, representation, num_samples=3,
            device=torch.device("cpu"),
        )
        loss.backward()

        self.assertEqual(representation.calls, 2)
        self.assertEqual(representation.linear.weight.grad.shape, (2, 2))

    def test_prc_bounds_large_replay_graph_and_keeps_trainable_gradient(self):
        class RecordingRepresentation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.batch_sizes = []

            def forward(self, inputs):
                self.batch_sizes.append(inputs.shape[0])
                return self.linear(inputs)

        representation = RecordingRepresentation()
        current = {class_id: torch.ones(2) for class_id in range(100)}
        memory = {class_id: torch.zeros(2) for class_id in range(100)}
        stds = {class_id: torch.zeros(2) for class_id in current}
        loss = prototype_replay_consistency_loss(
            current, stds, memory, stds, representation, num_samples=16,
            device=torch.device("cpu"), max_replay_batch=128,
        )
        loss.backward()

        self.assertEqual(representation.batch_sizes, [100, 100])
        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(representation.linear.weight.grad.norm().item(), 0.0)

    def test_prc_remains_finite_and_differentiable_as_memory_grows(self):
        torch.manual_seed(23)
        representation = torch.nn.Linear(3, 2, bias=False)
        memory = EpisodicPrototypeMemory(max_size=50, max_age=30)
        gradient_norms = []

        for round_idx in range(20):
            current_protos = {
                cls_id: torch.tensor(
                    [0.1 * round_idx, float(cls_id), 1.0],
                    dtype=torch.float32,
                )
                for cls_id in range(2)
            }
            current_stds = {
                cls_id: torch.full((3,), 0.05) for cls_id in current_protos
            }
            memory.add_from_proto_dicts(
                current_protos,
                current_stds,
                source_id=f"round_{round_idx}",
                round_idx=round_idx,
            )
            memory_protos, memory_stds = memory.to_proto_dist_dicts()

            representation.zero_grad(set_to_none=True)
            loss = prototype_replay_consistency_loss(
                current_protos,
                current_stds,
                memory_protos,
                memory_stds,
                representation,
                num_samples=4,
                device=torch.device("cpu"),
            )
            self.assertTrue(loss.requires_grad)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grad_norm = representation.weight.grad.norm()
            self.assertTrue(torch.isfinite(grad_norm))
            gradient_norms.append(grad_norm.item())
            self.assertEqual(len(memory), 2 * (round_idx + 1))

        self.assertLess(max(gradient_norms), 10.0)

    def test_dropout_changes_active_client_ids_when_enabled(self):
        active = {client_id: object() for client_id in range(5)}
        dropout = PrototypeDropout(rate=1.0, seed=17)
        kept = dropout.apply_to_source_outputs(active)
        self.assertNotEqual(set(kept), set(active))
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropout.stats["changed_calls"], 1)
        self.assertEqual(dropout.stats["active_set_last"], 1)

    def test_runtime_counter_schema_has_explicit_inactive_values(self):
        counters = RuntimeCounters().snapshot()
        self.assertEqual(counters["memory.reads"], 0)
        self.assertEqual(counters["memory.writes"], 0)
        self.assertEqual(counters["memory.replays"], 0)
        self.assertEqual(counters["reliability.weight_variance"], 0.0)
        self.assertEqual(counters["prc.loss_mean"], 0.0)
        self.assertEqual(counters["dropout.active_set_last"], 0)

    def test_ablation_presets_have_distinct_configs_and_hashes(self):
        effective, hashes = {}, {}
        for name in ABLATION_PRESETS:
            cfg = {**EHSFP_DEFAULTS, "dataset": "cifar100", "ablation_mode": name}
            effective[name] = get_ehsfp_config(cfg)
            hashes[name] = resolved_config_hash(cfg, effective[name])
        serialized = {json.dumps(v, sort_keys=True) for v in effective.values()}
        self.assertEqual(len(serialized), len(effective))
        self.assertEqual(len(set(hashes.values())), len(hashes))
        self.assertIs(effective["full_e_hsfp"]["use_prc_loss"], True)
        self.assertIs(effective["baseline_hsfp"]["use_prc_loss"], False)

    def test_checkpoint_hash_mismatch_is_refused(self):
        validate_checkpoint_config_hash({"resolved_config_hash": "same"}, "same")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_checkpoint_config_hash({"resolved_config_hash": "old"}, "new")

    def test_result_directory_guard(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            first = prepare_run_dir(tmp_path, "legacy-split", "abc")
            second = prepare_run_dir(tmp_path, "new-split", "abc")
            self.assertNotEqual(first, second)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                prepare_run_dir(tmp_path, "legacy-split", "abc")


if __name__ == "__main__":
    unittest.main()
