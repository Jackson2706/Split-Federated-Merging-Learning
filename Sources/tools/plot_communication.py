"""Plot the communication-overhead comparison figure used in the paper.

Usage:
    python tools/plot_communication.py [output_path]

Numbers below are the verified values from the paper's communication-cost table.
LaTeX text rendering is used when a LaTeX toolchain is available, otherwise the
script falls back to mathtext so it runs anywhere.
"""
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HAS_LATEX = shutil.which("latex") is not None
plt.rcParams.update({
    "text.usetex": _HAS_LATEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "communication_overhead.png"

# --- Verified Data ---
# Left Plot: Total Overhead (Log Scale) using exact numbers from the table
methods_total = ['FedAvg', 'HierFL', 'SplitFed', 'FedProto', 'HSFL', 'Ours']
overhead_total = [3865.35, 4392.44, 228.51, 17.25, 115.97, 13.79]

# Right Plot: Breakdown (Stacked) using exact numbers from the table
methods_breakdown = ['SplitFed', 'HSFL (A)', 'Ours (A)']
smashed_data = np.array([6.87, 68.67, 9.96])
model_weights = np.array([217.91, 3.83, 3.83])
gradients = np.array([3.73, 43.47, 0.00])

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)

# --- Left Plot: Total Overhead (Log Scale) ---
bars1 = ax1.bar(methods_total, overhead_total, color='#2b3d63', edgecolor='black', width=0.6)
ax1.set_yscale('log')
ax1.set_ylabel(r'\textbf{Total Overhead (GB) [Log Scale]}')
ax1.set_title(r'Comparison of Total Communication Cost')
ax1.grid(axis='y', linestyle=':', alpha=0.6, which='both')

# Add precise values on top of bars
for bar in bars1:
    height = bar.get_height()
    # Formatting to 2 decimal places to match the table exactly, except for large numbers where it might get cluttered
    if height > 1000:
        label = f'${height:,.2f}$'
    else:
        label = f'${height:.2f}$'
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             label, ha='center', va='bottom', fontsize=8)

# --- Right Plot: Breakdown (Stacked) ---
bar_width = 0.5
p1 = ax2.bar(methods_breakdown, smashed_data, label='Smashed Data', 
             color='#70e0f0', edgecolor='black', width=bar_width)
p2 = ax2.bar(methods_breakdown, model_weights, bottom=smashed_data, label='Model Weights', 
             color='#ffb347', edgecolor='black', width=bar_width)
p3 = ax2.bar(methods_breakdown, gradients, bottom=smashed_data + model_weights, label='Gradients', 
             color='#b32424', edgecolor='black', width=bar_width)

ax2.set_ylabel(r'\textbf{Overhead Breakdown (GB)}')
ax2.set_title(r'Breakdown of Split Methods')
ax2.grid(axis='y', linestyle=':', alpha=0.6)

# Legend inside the figure (upper right)
ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

# Clean up layout
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f"saved {OUT_PATH}")