from .FedAvgClient import FedAvgClient
from .FedNovaClient import FedNovaClient
from .FedProxClient import FedProxClient
from .FedSGDClient import FedSGDClient

client_update_strategy_map = {
    "fedavg": FedAvgClient,
    "fednova": FedNovaClient,
    "fedprox": FedProxClient,
    "fedsgd": FedSGDClient,
}


def get_client_update_strategy(strategy_name: str):
    try:
        return client_update_strategy_map[strategy_name]
    except:
        exit("Error: unrecognized client update strategy")