import copy

import torch

from .Aggregator import Aggregator


class FedAvgAggregator(Aggregator):
    def aggregate(self, _1, _2, client_updates):
        """
        client_updates: list of state_dicts (each from a client)
        Return the average model weights (FedAvg)
        """
        w_avg = copy.deepcopy(client_updates[0])

        for key in w_avg.keys():
            # Convert to float for safe accumulation if needed
            if w_avg[key].dtype in [torch.float32, torch.float64, torch.float16]:
                for i in range(1, len(client_updates)):
                    w_avg[key] += client_updates[i][key].to(w_avg[key].device)
                w_avg[key] = w_avg[key] / len(client_updates)
            else:
                # # For non-float types (like Long), just pick the first client's version
                # for i in range(1, len(client_updates)):
                #     if not torch.equal(w_avg[key], client_updates[i][key]):
                #         print(f"[Warning] Skipping averaging non-float field: {key}")
                # Keep it unchanged (or majority vote if needed)
                # Optional: vote or assert consistency
                pass

        return w_avg
