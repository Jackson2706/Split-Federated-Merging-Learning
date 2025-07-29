from .FedAvgAggregator import FedAvgAggregator

strategies_map = {
    "Splitfedavg": FedAvgAggregator,
}


def get_strategy(strategy_name):
    try:
        return strategies_map[strategy_name]
    except:
        exit("Error: unrecognized strategy")
