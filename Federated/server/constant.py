from .FedAvgAggregator import FedAvgAggregator
from .FedNovaAggregator import FedNovaAggregator
from .FedProxAggregator import FedProxAggregator
from .FedSGDAggregator import FedSGDAggregator
strategies_map = {
    "fedavg": FedAvgAggregator,
    "fednova": FedNovaAggregator,
    "fedprox": FedProxAggregator,
    "fedsgd": FedSGDAggregator,
}

def get_strategy(strategy_name):
    try: 
        return strategies_map[strategy_name]
    except:
        exit("Error: unrecognized strategy")