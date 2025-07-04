import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def load_all_json(json_paths):
    data_list = {}
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            key = path.split('/')[-1].replace('.json', '')
            data_list[key] = data
    return data_list

# Define paths
json_paths = [
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_fedavg_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_fednova_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_fedprox_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_fedsgd_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_hierfl_output.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/cifar_ours_output.json',
]

# Load the JSON files
all_data = load_all_json(json_paths)
feature_name = 'train_accuracy'

# Extract and convert to percent
fedavg = [v * 100 for v in all_data['cifar_fedavg_output'][feature_name]]
fednova = [v * 100 for v in all_data['cifar_fednova_output'][feature_name]]
fedprox = [v * 100 for v in all_data['cifar_fedprox_output'][feature_name]]
fedsgd = [v * 100 for v in all_data['cifar_fedsgd_output'][feature_name]]
hierfl = [v * 100 for v in all_data['cifar_hierfl_output'][feature_name]]
ours = [v * 100 for v in all_data['cifar_ours_output'][feature_name]]  # 5 points: 0, 50, 100, 150, 200

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(fedavg, label='FedAvg', alpha=0.8)
plt.plot(fednova, label='FedNova', alpha=0.8)
plt.plot(fedprox, label='FedProx', alpha=0.8)
plt.plot(fedsgd, label='FedSGD', alpha=0.8)
plt.plot(hierfl, label='HierFL', alpha=0.8)

# Smooth "Ours" using spline interpolation
ours_iters = np.array([0, 50, 100, 150, 200])
ours_values = np.array(ours)

spline = make_interp_spline(ours_iters, ours_values, k=2)
smooth_x = np.linspace(0, 200, 200)
smooth_y = spline(smooth_x)

plt.plot(smooth_x, smooth_y, label='Ours (Smooth)', color='black', linewidth=2.5, linestyle='--')

# Labeling
plt.xlabel('Communication Round')
plt.ylabel('Train F1 Score (%)')
plt.title('Train F1 Score Comparison Across Methods')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
output_dir = '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/figures/'
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, f'cifar_{feature_name}_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Plot saved to: {output_dir}cifar_{feature_name}_comparison.png")
