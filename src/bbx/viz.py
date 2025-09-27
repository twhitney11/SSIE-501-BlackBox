# viz.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_label_colors(json_path: str | Path) -> dict:
    return json.loads(Path(json_path).read_text())

def cmap_for_labels(labels, colors_dict):
    # labels is your index→label list
    return ListedColormap([colors_dict.get(lab, "#808080") for lab in labels], name="bbx")

def plot_state_colored(state_int, labels, colors_dict, title=None, show_colorbar=True):
    cmap = cmap_for_labels(labels, colors_dict)
    im = plt.imshow(state_int, cmap=cmap, interpolation="nearest", vmin=0, vmax=len(labels)-1)
    if show_colorbar:
        cbar = plt.colorbar(im, ticks=range(len(labels)))
        cbar.ax.set_yticklabels(labels)
    if title: plt.title(title)
    plt.axis("off"); plt.tight_layout(); plt.show()

def plot_sequence(states_int, labels, colors_dict, steps=None, cols=5, figsize=(12,8)):
    import numpy as np
    if steps is None:
        N = min(15, len(states_int))
        steps = np.linspace(0, len(states_int)-1, N, dtype=int).tolist()
    rows = -(-len(steps)//cols)  # ceil
    cmap = cmap_for_labels(labels, colors_dict)
    plt.figure(figsize=figsize)
    for k, t in enumerate(steps):
        ax = plt.subplot(rows, cols, k+1)
        ax.imshow(states_int[t], cmap=cmap, interpolation="nearest", vmin=0, vmax=len(labels)-1)
        ax.set_title(f"t={t}", fontsize=9)
        ax.axis("off")
    plt.tight_layout(); plt.show()
