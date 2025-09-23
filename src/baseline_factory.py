from src.client.fedavg_client import FedAvgClient
from src.client.hierfedavg_client import HierFedAvgClient
from src.cloud.fedavg_cloud import FedAvgCloud
from src.cloud.hierfedavg_cloud import HierFedAvgCloud
from src.edge.hierfedavg_edge import HierFedAvgEdge
from src.cloud.hfedprotogencl_cloud import HFedProtoGenCLCloud
from src.edge.hfedprotogencl_edge import HFedProtoGenCLEdge
from src.client.hfedprotogencl_client import HFedProtoGenCLClient
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
    },
    "hfedprotogencl":{
        "cloud": HFedProtoGenCLCloud,
        "edge": HFedProtoGenCLEdge,
        "client": HFedProtoGenCLClient
    }
    # "split": { "cloud": SplitCloud, "client": SplitClient },
    # "my_Hmethod": { "cloud": MyCloud, "client": MyClient },
}

def get_baseline(baseline_name: str):
    baseline_name = baseline_name.lower()
    if baseline_name not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline_name}. Available: {list(BASELINES.keys())}")
    return BASELINES[baseline_name]