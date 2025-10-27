from config import SystemConfig
from dataset_preparation import prepare_data
import yaml
import collections
import numpy as np

config = SystemConfig()


# Prepare dataset
## Load data config
try:
    with open(config.CONFIG_PATH, 'r') as f:
        loaded_config = yaml.safe_load(f)
        data_params = loaded_config['data_config']
except FileNotFoundError:
     print(f"FATAL: Config file not found at {config.CONFIG_PATH}.")
     raise
except Exception as e:
     print(f"FATAL: Error reading config file {config.CONFIG_PATH}: {e}")
     raise

## Load data loader
client_loaders, test_loader, class_names = prepare_data(
    data_path=config.DATA_PATH,
    batch_size=config.BATCH_SIZE,
    seed=config.SEED,
    **data_params
)



# Create hierarchy
entities_map = collections.defaultdict(list)
client_parent_map = {}
entity_childeren_map = collections.defaultdict(list)

## Distribute the clients, edge servers and clouds
total_number_clients = len(client_loaders) # Get the number of clients from data config
entities_map[0] = list(range(total_number_clients))

num_entities_edge_server = config.ENTITIES_PER_TIER[0] # Get the number of edge servers from system config
entities_map[1] = list(range(num_entities_edge_server))
client_indices_per_edge = np.array_split(
    ary=np.arange(total_number_clients),
    indices_or_sections=num_entities_edge_server
)

for edge_id, client_indices in enumerate(client_indices_per_edge):
    entity_childeren_map[(1, edge_id)] = list(client_indices)
    for client_id in client_indices:
        client_parent_map[client_id] = (1, edge_id)

if config.NUM_TIERS > 2:
    num_entities_cloud = config.ENTITIES_PER_TIER[1]
    entities_map[2] = list(range(num_entities_cloud))
    entity_childeren_map[(2, 0)] = entities_map[1]

print("\nHierarchical Structure:")
print(f"  - Tier 1 (Clients): {total_number_clients}")
print(f"  - Tier 2 (Edges): {num_entities_edge_server}")
if config.NUM_TIERS > 2: 
    print(f"  - Tier 3 (Cloud): {num_entities_cloud}")