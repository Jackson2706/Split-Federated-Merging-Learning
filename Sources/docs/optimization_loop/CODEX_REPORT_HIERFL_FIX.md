Fixed [classification/HierFL/servers/FedAvgAggregator.py:20](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/HierFL/servers/FedAvgAggregator.py:20):

```diff
- w_avg[key] += client_updates[i][key]
+ w_avg[key] += client_updates[i][key].to(w_avg[key].device)
```

Checked the complete `aggregate()` method; no other device-sensitive tensor operations require changes. Averaging math and weighting are unchanged. No training/GPU run performed, and all existing uncommitted changes were preserved.