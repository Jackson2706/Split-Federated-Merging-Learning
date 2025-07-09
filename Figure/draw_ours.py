import json

import matplotlib.pyplot as plt


def load_all_json(json_paths):
    data_list = {}
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            key = path.split('/')[-1].replace('.json', '')
            data_list[key] = data
    return data_list

# Define the paths
json_paths = [
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_ours_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_oursv2_output.json'
]

# Load the JSON files
all_data = load_all_json(json_paths)
feature_name = 'train_accuracy'

# Extract the lists
ours = all_data['cifar_ours_output'][feature_name]
oursv2 = all_data['cifar_oursv2_output'][feature_name]

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(oursv2, label='Ours V2', alpha=0.8)
# Plot OURS — only 5 points at [0, 50, 100, 150, 200]
ours_iters = [0, 50, 100, 150, 200]
plt.plot(ours_iters, ours, 'o--', label='Ours', linewidth=2.5, color='black', markersize=6)

plt.xlabel('Communication Round')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy Comparison Across Methods')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/figures/cifar_{feature_name}_comparison.png', dpi=300, bbox_inches='tight')