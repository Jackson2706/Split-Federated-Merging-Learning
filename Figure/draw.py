import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# === Use LaTeX and Computer Modern font (with bold support) ===
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,lmodern}\boldmath"
})

# === Font sizes for clarity in 2-column paper ===
FONT_SIZE = 36       # Axis labels
LEGEND_SIZE = 19     # Legend text
TICK_SIZE = 28       # Tick labels
TITLE_SIZE = 32      # Title text
LINE_WIDTH = 3       # Line thickness
FIG_SIZE = (16, 9)   # Large figure
DPI = 600            # High resolution

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
    if len(data) < 4:
        return [y_old[-1]] * target_len
    spline = make_interp_spline(x_old, y_old, k=3)
    x_new = np.linspace(0, 1, target_len)
    y_new = spline(x_new)
    return y_new.tolist()

# === JSON paths ===
json_paths = [
    './data/cifar_fedavg_200_200_5_output.json',
    './data/cifar_fednova_200_200_5_output.json',
    './data/cifar_fedprox_200_200_5_output.json',
    './data/cifar_fedsgd_200_200_5_output.json',
    './data/cifar_ourv1_200_200_5_10_output.json',
    './data/cifar_ourv1_200_200_10_20_output.json',
    './data/cifar_ourv1_200_200_25_50_output.json',
    './data/cifar_SplitFed_200_200_5_output.json',
    './data/cifar_HierFL_200_200_5_output.json'
]

# === Load data ===
all_data = load_all_json(json_paths)
feature_name = 'train_accuracy'  # or 'train_loss'

def extract_and_interpolate(key):
    raw = [v * 100 for v in all_data[key][feature_name]]
    return interpolate_to_length(raw) if len(raw) < 200 else raw

fedavg   = extract_and_interpolate('cifar_fedavg_200_200_5_output')
fednova  = extract_and_interpolate('cifar_fednova_200_200_5_output')
fedprox  = extract_and_interpolate('cifar_fedprox_200_200_5_output')
fedsgd   = extract_and_interpolate('cifar_fedsgd_200_200_5_output')
hierfl   = extract_and_interpolate('cifar_HierFL_200_200_5_output')
splitfed = extract_and_interpolate('cifar_SplitFed_200_200_5_output')
ours1    = extract_and_interpolate('cifar_ourv1_200_200_5_10_output')
ours2    = extract_and_interpolate('cifar_ourv1_200_200_10_20_output')
ours3    = extract_and_interpolate('cifar_ourv1_200_200_25_50_output')

# === Plotting ===
plt.figure(figsize=FIG_SIZE)

plt.plot(fedavg,   label=r'\textbf{FedAvg}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fednova,  label=r'\textbf{FedNova}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fedprox,  label=r'\textbf{FedProx}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fedsgd,   label=r'\textbf{FedSGD}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(hierfl,   label=r'\textbf{HierFL}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(splitfed, label=r'\textbf{SplitFed}', linewidth=LINE_WIDTH, alpha=0.8)

# Ours with LaTeX-style labels
plt.plot(ours1, label=r'\textbf{Ours ($t_1{=}5$, $t_2{=}10$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)
plt.plot(ours2, label=r'\textbf{Ours ($t_1{=}10$, $t_2{=}20$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)
plt.plot(ours3, label=r'\textbf{Ours ($t_1{=}25$, $t_2{=}50$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)

# === Labels ===
plt.xlabel(r'\textbf{Communication Round}', fontsize=FONT_SIZE)
plt.ylabel(r'\textbf{Train F1 Score (\%)}', fontsize=FONT_SIZE)
plt.xticks(fontsize=TICK_SIZE, fontweight='bold')
plt.yticks(fontsize=TICK_SIZE, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=LEGEND_SIZE, loc='best')
plt.tight_layout()

# === Save Figure ===
output_dir = '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/figures/'
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f'cifar_{feature_name}_comparison.png')
plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"✅ High-res bold plot saved to: {plot_path}")
