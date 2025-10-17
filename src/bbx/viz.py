# viz.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

# Set bg color to gray because "mex" is white and invisible.
plt.rcParams['axes.facecolor']   = '#cccccc' 
plt.rcParams['figure.facecolor'] = '#cccccc'
plt.rcParams['savefig.facecolor'] = '#cccccc'

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

def plot_sequence(states_int, labels, colors_dict, steps=None, cols=5, figsize=(12,8),
                  save_path=None, show=True):
    if steps is None:
        N = min(15, len(states_int))
        steps = np.linspace(0, len(states_int)-1, N, dtype=int).tolist()
    rows = -(-len(steps)//cols)  # ceil
    cmap = cmap_for_labels(labels, colors_dict)
    fig = plt.figure(figsize=figsize)
    for k, t in enumerate(steps):
        ax = plt.subplot(rows, cols, k+1)
        ax.imshow(states_int[t], cmap=cmap, interpolation="nearest", vmin=0, vmax=len(labels)-1)
        ax.set_title(f"t={t}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def states_to_rgb_frames(states_int, labels, colors_dict, scale=1):
    cmap = cmap_for_labels(labels, colors_dict)
    lut = cmap(range(len(labels)))[:, :3]  # RGB floats
    frames = []
    for S in states_int:
        rgb = lut[np.clip(S, 0, len(labels)-1)]
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        frames.append((rgb * 255).astype(np.uint8))
    return frames


def save_gif(frames, path: Path, duration_ms=100, loop=0):
    if not frames:
        raise ValueError("No frames provided for GIF")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=loop)
    print(f"[saved] {path}")
