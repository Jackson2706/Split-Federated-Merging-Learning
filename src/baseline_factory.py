from src.client.fedavg_client import FedAvgClient
from src.cloud.fedavg_cloud import FedAvgCloud
from src.client.hierfedavg_client import HierFedAvgClient
from src.edge.hierfedavg_edge import HierFedAvgEdge
from src.cloud.hierfedavg_cloud import HierFedAvgCloud
# later: from src.cloud.split_cloud import SplitCloud, etc.

BASELINES = {
    "fedavg": {
        "cloud": FedAvgCloud,
        "client": FedAvgClient
    },
    "hierfedavg":{
        "cloud": HierFedAvgCloud,
        "edge": HierFedAvgEdge,
        "client": HierFedAvgClient
    }
    # "split": { "cloud": SplitCloud, "client": SplitClient },
    # "my_method": { "cloud": MyCloud, "client": MyClient },
}

def get_baseline(baseline_name: str):
    baseline_name = baseline_name.lower()
    if baseline_name not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline_name}. Available: {list(BASELINES.keys())}")
    return BASELINES[baseline_name]