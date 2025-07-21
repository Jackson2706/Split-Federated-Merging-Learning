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
FONT_SIZE = 55       # Axis labels
LEGEND_SIZE = 30     # Legend text
TICK_SIZE = 50       # Tick labels
TITLE_SIZE = 56      # Title text
LINE_WIDTH = 6      # Line thickness
FIG_SIZE = (25, 15)   # Large figure
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
    x_new = np.linspace(0, 1, target_len)
    y_new = np.interp(x_new, x_old, y_old)
    return y_new.tolist()



# === JSON paths ===
json_paths = [
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/fedavg_cifar10_iid:True_alexnet_25 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/fednova_cifar10_iid:True_alexnet_25 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/fedprox_cifar10_iid:True_alexnet_25 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/fedsgd_cifar10_iid:True_alexnet_25 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/HierFL_cifar10_iid:True_alexnet_20 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/SplitFL_cifar10_iid:True_alexnet_25 users.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/HSPL_cifar10_iid:True_alexnet_20 users_t1:5_t2:10.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/HSPL_cifar10_iid:True_alexnet_20 users_t1:10_t2:20.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/HSPL_cifar10_iid:True_alexnet_20 users_t1:25_t2:50.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/OursV1_cifar10_iid:True_alexnet_20 users_t1:5_t2:10.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/OursV1_cifar10_iid:True_alexnet_20 users_t1:10_t2:20.json',
    '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/OursV1_cifar10_iid:True_alexnet_20 users_t1:25_t2:50.json'
]

# === Load data ===
all_data = load_all_json(json_paths)
feature_name = 'train_accuracy'  # or 'train_loss'

def extract_and_interpolate(key):
    raw = [v * 100 for v in all_data[key][feature_name]]
    return interpolate_to_length(raw) if len(raw) < 200 else raw

fedavg   = extract_and_interpolate('fedavg_cifar10_iid:True_alexnet_25 users')
fednova  = extract_and_interpolate('fednova_cifar10_iid:True_alexnet_25 users')
fedprox  = extract_and_interpolate('fedprox_cifar10_iid:True_alexnet_25 users')
fedsgd   = extract_and_interpolate('fedsgd_cifar10_iid:True_alexnet_25 users')
hierfl   = extract_and_interpolate('HierFL_cifar10_iid:True_alexnet_20 users')
splitfed = extract_and_interpolate('SplitFL_cifar10_iid:True_alexnet_25 users')
hspl_5_10    = extract_and_interpolate('HSPL_cifar10_iid:True_alexnet_20 users_t1:5_t2:10')
hspl_10_20   = extract_and_interpolate('HSPL_cifar10_iid:True_alexnet_20 users_t1:10_t2:20')
hspl_25_50   = extract_and_interpolate('HSPL_cifar10_iid:True_alexnet_20 users_t1:25_t2:50')
ours1_5_10   = extract_and_interpolate('OursV1_cifar10_iid:True_alexnet_20 users_t1:5_t2:10')
ours1_10_20  = extract_and_interpolate('OursV1_cifar10_iid:True_alexnet_20 users_t1:10_t2:20')
ours1_25_50  = extract_and_interpolate('OursV1_cifar10_iid:True_alexnet_20 users_t1:25_t2:50')


# === Plotting ===
plt.figure(figsize=FIG_SIZE)

plt.plot(fedavg,   label=r'\textbf{FedAvg}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fednova,  label=r'\textbf{FedNova}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fedprox,  label=r'\textbf{FedProx}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(fedsgd,   label=r'\textbf{FedSGD}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(hierfl,   label=r'\textbf{HierFL}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(splitfed, label=r'\textbf{SplitFed}', linewidth=LINE_WIDTH, alpha=0.8)

# Ours with LaTeX-style labels
plt.plot(hspl_5_10, label=r'\textbf{HSPL ($t_1{=}5$, $t_2{=}10$)}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(hspl_10_20, label=r'\textbf{HSPL ($t_1{=}10$, $t_2{=}20$)}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(hspl_25_50, label=r'\textbf{HSPL ($t_1{=}25$, $t_2{=}50$)}', linewidth=LINE_WIDTH, alpha=0.8)
plt.plot(ours1_5_10, label=r'\textbf{Ours ($t_1{=}5$, $t_2{=}10$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)
plt.plot(ours1_10_20, label=r'\textbf{Ours ($t_1{=}10$, $t_2{=}20$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)
plt.plot(ours1_25_50, label=r'\textbf{Ours ($t_1{=}25$, $t_2{=}50$)}', linestyle='--', linewidth=LINE_WIDTH, alpha=0.9)

# === Labels ===
plt.xlabel(r'\textbf{Communication Round}', fontsize=FONT_SIZE)
plt.ylabel(r'\textbf{F1 Score (\%)}', fontsize=FONT_SIZE)
plt.xticks(fontsize=TICK_SIZE, fontweight='bold')
plt.yticks(fontsize=TICK_SIZE, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=LEGEND_SIZE, loc='lower right', frameon=False, ncol=3)
plt.tight_layout()

# === Save Figure ===
output_dir = '/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/figures/'
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f'cifar_{feature_name}_comparison.png')
plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"✅ High-res bold plot saved to: {plot_path}")
