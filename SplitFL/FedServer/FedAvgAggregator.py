import copy

import torch

from .Aggregator import Aggregator


class FedAvgAggregator(Aggregator):
    def aggregate(self, _1, _2, client_updates):
        """
        client_updates: list of state_dicts
        Trả về trung bình weights
        """
        w_avg = copy.deepcopy(client_updates[0])
        for key in w_avg.keys():
            for i in range(1, len(client_updates)):
                w_avg[key] += client_updates[i][key]
            w_avg[key] = torch.div(w_avg[key], len(client_updates))
        return w_avg