from .fedavg import average_weights
strategies_map = {
    "fedavg": average_weights
}

def get_strategy(strategy_name):
    try: 
        return strategies_map[strategy_name]
    except:
        exit("Error: unrecognized strategy")