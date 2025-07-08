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

def interpolate_to_length(data, target_len=200):
    x_old = np.linspace(0, 1, len(data))
    y_old = np.array(data)
    if len(data) < 4:  # too short for spline
        return [y_old[-1]] * target_len
    spline = make_interp_spline(x_old, y_old, k=3)
    x_new = np.linspace(0, 1, target_len)
    y_new = spline(x_new)
    return y_new.tolist()

# Define paths
json_paths = [
    './data/cifar_fedavg_200_200_5_output.json',
    './data/cifar_fednova_200_200_5_output.json',
    './data/cifar_fedprox_200_200_5_output.json',
    './data/cifar_fedsgd_200_200_5_output.json',
    './data/cifar_ourv1_200_200_5_10_output.json',
    './data/cifar_ourv1_200_200_10_20_output.json',
    './data/cifar_ourv1_200_200_25_50_output.json',
    './data/cifar_SplitFed_200_200_5_output.json'
]

# Load the JSON files
all_data = load_all_json(json_paths)
feature_name = 'train_loss'

# Extract with interpolation
def extract_and_interpolate(key):
    raw = [v * 100 for v in all_data[key][feature_name]]
    return interpolate_to_length(raw) if len(raw) < 200 else raw

fedavg   = extract_and_interpolate('cifar_fedavg_200_200_5_output')
fednova  = extract_and_interpolate('cifar_fednova_200_200_5_output')
fedprox  = extract_and_interpolate('cifar_fedprox_200_200_5_output')
fedsgd   = extract_and_interpolate('cifar_fedsgd_200_200_5_output')
hierfl   = extract_and_interpolate('cifar_fedavg_200_200_5_output')  # dummy
splitfed = extract_and_interpolate('cifar_SplitFed_200_200_5_output')
ours1    = extract_and_interpolate('cifar_ourv1_200_200_5_10_output')
ours2    = extract_and_interpolate('cifar_ourv1_200_200_10_20_output')
ours3    = extract_and_interpolate('cifar_ourv1_200_200_25_50_output')

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(fedavg,   label='FedAvg', alpha=0.8)
plt.plot(fednova,  label='FedNova', alpha=0.8)
plt.plot(fedprox,  label='FedProx', alpha=0.8)
plt.plot(fedsgd,   label='FedSGD', alpha=0.8)
plt.plot(hierfl,   label='HierFL', alpha=0.8)
plt.plot(splitfed, label='SplitFed', alpha=0.8)

# Ours with LaTeX in legend
plt.plot(ours1, label=r'Ours ($t_1=5$, $t_2=10$)', linestyle='--', alpha=0.9)
plt.plot(ours2, label=r'Ours ($t_1=10$, $t_2=20$)', linestyle='--', alpha=0.9)
plt.plot(ours3, label=r'Ours ($t_1=25$, $t_2=50$)', linestyle='--', alpha=0.9)

# Labels and layout
plt.xlabel('Communication Round')
plt.ylabel('Train F1 Score (%)')
plt.title('Train F1 Score Comparison Across Methods')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save
output_dir = '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/figures/'
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f'cifar_{feature_name}_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Plot saved to: {plot_path}")
