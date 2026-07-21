import math

import torch

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
