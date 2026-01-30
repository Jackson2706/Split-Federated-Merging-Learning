import copy
import sys

import torch

from .Aggregator import Aggregator


class FedAvgAggregator(Aggregator):
    def aggregate(self, _1, _2, client_updates):
        """
        Parameters:
            client_updates: list of state_dicts
        Return:
            w_avg: the averaged weights of all clients
            server_comm: communication overhead
        """
        server_comm = sys.getsizeof(client_updates)
        w_avg = copy.deepcopy(client_updates[0])
        for key in w_avg.keys():
            for i in range(1, len(client_updates)):
                w_avg[key] += client_updates[i][key]
            w_avg[key] = torch.div(w_avg[key], len(client_updates))
        return w_avg, server_comm