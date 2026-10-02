import math
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ehsfp.communication import (
    COMMUNICATION_ROUTES,
    add_communication,
    assert_communication_total,
    mb_of,
    new_communication_tracker,
)


def _assert_schema_and_total(tracker):
    assert set(tracker) == set(COMMUNICATION_ROUTES) | {"total_comm_MB"}
    assert math.isclose(
        tracker["total_comm_MB"],
        sum(tracker[route] for route in COMMUNICATION_ROUTES),
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert_communication_total(tracker)


def _load_hsfp_hierarchy():
    path = Path(__file__).parents[1] / "classification" / "H-SFP" / "hierarchy.py"
    spec = importlib.util.spec_from_file_location("hsfp_hierarchy_for_comm_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packet_by_packet_one_round_reconciles_to_hand_calculated_total():
    tracker = new_communication_tracker()

    # Hand ledger, in MiB:
    #   two 1.25-MiB client prototype packets        = 2.50
    #   one 0.50-MiB edge prototype packet           = 0.50
    #   three 4-MiB client-model uploads/downloads   = 12 + 12
    #   two 6-MiB edge-model uploads/downloads       = 12 + 12
    #                                                --------
    #                                                  51.00
    packets = (
        ("client_to_edge_MB", 1.25, 2),
        ("edge_to_cloud_MB", 0.50, 1),
        ("client_to_edge_MB", 4.00, 3),
        ("edge_to_client_MB", 4.00, 3),
        ("edge_to_cloud_MB", 6.00, 2),
        ("cloud_to_edge_MB", 6.00, 2),
    )
    for route, payload_mb, copies in packets:
        add_communication(tracker, route, mb=payload_mb, copies=copies)

    assert tracker["client_to_edge_MB"] == 14.5
    assert tracker["edge_to_client_MB"] == 12.0
    assert tracker["edge_to_cloud_MB"] == 12.5
    assert tracker["cloud_to_edge_MB"] == 12.0
    assert tracker["client_to_server_MB"] == 0.0
    assert tracker["server_to_client_MB"] == 0.0
    assert tracker["total_comm_MB"] == 51.0
    _assert_schema_and_total(tracker)


def test_client_extraction_payload_is_independent_of_episodic_memory_flag():
    hierarchy = _load_hsfp_hierarchy()
    loader = DataLoader(
        TensorDataset(
            torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2),
            torch.tensor([0, 0, 1, 1]),
        ),
        batch_size=2,
        shuffle=False,
    )

    class FeatureModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.cat((inputs, inputs + 1), dim=1)

    outputs = []
    for memory_enabled in (False, True):
        # _client_ssl_extraction_phase does not read ecfg or either memory
        # object. Constructing only its actual dependencies makes that
        # independence executable without building the full hierarchy.
        owner = object.__new__(hierarchy.HierarchicalFL)
        owner.device = torch.device("cpu")
        owner.structure = {-1: {0: FeatureModel()}}
        owner.optimizers = {-1: {0: {"optimizer": None, "scaler": None}}}
        owner.ssl_transforms = torch.nn.Identity()
        owner.client_supervised_optimizers = {}
        owner.client_supervised_heads = {}
        owner.ecfg = {"use_episodic_memory": memory_enabled}
        owner.client_memory = object() if memory_enabled else None

        proto_dist, support = owner._client_ssl_extraction_phase(0, loader, ssl_epochs=0)
        outputs.append((proto_dist, support))

    for proto_dist, support in outputs:
        proto, dist = proto_dist
        assert set(proto) == {0, 1}
        assert set(dist) == {0, 1}
        assert support == {0: 2, 1: 2}
        assert all(tensor.shape == (2, 2, 2) for tensor in proto.values())

    assert outputs[0][1] == outputs[1][1]
    for left_dict, right_dict in zip(outputs[0][0], outputs[1][0]):
        assert left_dict.keys() == right_dict.keys()
        for class_id in left_dict:
            assert torch.equal(left_dict[class_id], right_dict[class_id])
    assert hierarchy.get_proto_dist_size_MB(outputs[0][0]) == hierarchy.get_proto_dist_size_MB(
        outputs[1][0]
    )


def test_archived_cifar_ablation_gap_matches_changed_feature_shape_not_memory():
    hierarchy = _load_hsfp_hierarchy()
    class_ids = range(7)
    old_boundary = (
        {class_id: torch.zeros(64, 32, 32) for class_id in class_ids},
        {class_id: torch.zeros(64, 32, 32) for class_id in class_ids},
    )
    pooled_boundary = (
        {class_id: torch.zeros(128, 1, 1) for class_id in class_ids},
        {class_id: torch.zeros(128, 1, 1) for class_id in class_ids},
    )

    # Commit cf6454c changed CIFAR's client boundary from [64,32,32] to
    # [128,1,1]. The class keys/call count can be identical while bytes fall
    # exactly 512x, explaining the cross-version archived result.
    assert hierarchy.get_proto_dist_size_MB(old_boundary) == (
        512 * hierarchy.get_proto_dist_size_MB(pooled_boundary)
    )


def test_flat_fedavg_synthetic_round_counts_selected_model_upload_and_download():
    model = {"weight": torch.zeros(8, dtype=torch.float32), "counter": torch.zeros(1, dtype=torch.int64)}
    tracker = new_communication_tracker()

    # Two selected clients receive and return one complete state_dict each.
    add_communication(tracker, "server_to_client_MB", payload=model, copies=2)
    add_communication(tracker, "client_to_server_MB", payload=model, copies=2)

    expected = 4 * mb_of(model)
    assert math.isclose(tracker["total_comm_MB"], expected)
    assert tracker["client_to_edge_MB"] == 0
    _assert_schema_and_total(tracker)


def test_hsfp_synthetic_round_counts_prototypes_and_periodic_model_aggregation():
    client_model = {"weight": torch.zeros(4, dtype=torch.float32)}
    edge_model = {"weight": torch.zeros(6, dtype=torch.float32)}
    proto_dist = (
        {0: torch.zeros(3, dtype=torch.float32)},
        {0: torch.zeros(3, dtype=torch.float32)},
    )
    tracker = new_communication_tracker()

    # Two clients send prototype/distribution pairs; one edge forwards its pair.
    add_communication(tracker, "client_to_edge_MB", payload=proto_dist, copies=2)
    add_communication(tracker, "edge_to_cloud_MB", payload=proto_dist)
    # A t1/t2 aggregation sends models in both directions at each boundary.
    for route in ("client_to_edge_MB", "edge_to_client_MB"):
        add_communication(tracker, route, payload=client_model, copies=2)
    for route in ("edge_to_cloud_MB", "cloud_to_edge_MB"):
        add_communication(tracker, route, payload=edge_model)

    assert tracker["client_to_server_MB"] == 0
    assert tracker["client_to_edge_MB"] > tracker["edge_to_client_MB"]
    _assert_schema_and_total(tracker)


def test_heterosfl_round_fixes_old_dead_download_and_activation_only_undercount():
    client_model = {"weight": torch.zeros(4, dtype=torch.float32)}
    activation = torch.zeros((2, 3), dtype=torch.float32)
    activation_grad = torch.zeros_like(activation)
    labels = torch.zeros(2, dtype=torch.int64)
    tracker = new_communication_tracker()

    add_communication(tracker, "server_to_client_MB", payload=client_model)
    add_communication(tracker, "client_to_server_MB", payload=(activation, labels))
    add_communication(tracker, "server_to_client_MB", payload=activation_grad)
    add_communication(tracker, "client_to_server_MB", payload=client_model)

    old_activation_only_total = mb_of(activation)
    assert tracker["server_to_client_MB"] > 0  # old download_MB stayed zero
    assert tracker["total_comm_MB"] > old_activation_only_total
    _assert_schema_and_total(tracker)
