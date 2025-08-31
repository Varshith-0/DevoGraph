# MODELING NEURAL DEVELOPMENTAL PROGRAMS OF *C. elegans* USING GROWING HYPERGRAPH NEURAL NETWORKS


## Table of Contents
* [Updates](#updates)
* [NDP-HNN Environment Setup](#ndp-hnn-environment-setup)

  * [Steps](#steps)
* [Project Layout](#project-layout)
* [requirements.txt (and how to get it)](#requirementstxt-and-how-to-get-it)
* [Data Format](#data-format)
* [How It Works](#how-it-works)
* [Run It](#run-it)
* [Command-Line Options](#command-line-options)
* [Outputs](#outputs)
* [Optional Visualizations](#optional-visualizations)
* [Troubleshooting](#troubleshooting)

---
## Updates 

[Blog] Blog details will be provided soon. 

[Paper] Paper details will be provided soon. 


---

## NDP-HNN Environment Setup

This guide explains how to create a new conda environment named `ndphnn` and install the required Python packages using the provided `requirements.txt`.
It is expected to have conda/anaconda distribution already installed on your device or server.

---

## Steps

### 1. Create the conda environment with Python 3.9 (or another version of your choice):

```bash
conda create -n ndphnn python=3.9 -y
```

### 2. Activate the environment:

```bash
conda activate ndphnn
```


### 3. Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Verify the installation:

```bash
python -c "import torch, tqdm, torch_geometric, hypernetx, networkx, pandas, matplotlib, seaborn, umap, plotly"
echo "All packages imported successfully!"
```

---

## Project Layout

```
NDP-HNN/
├── main.py               # entry point: data → time-snapshots → train → save embeddings
├── config.py             # defaults / hyperparameters
├── data_io.py            # CSV loader + birth features + lineage graph
├── hyperedges.py         # per-time hyperedge builders (spatial-KNN + lineage siblings)
├── snapshots.py          # builds PyG Data snapshots for t = 0..T_max
├── model.py              # DynHNN and DynGrowingHNN
├── losses.py             # incidence_bce reconstruction loss
├── train.py              # training loop: MSE(next xyz) + incidence BCE
├── evaluate.py           # extract (T×N×D) embeddings and save to .npy
├── visualize.py          # optional UMAP & Plotly helpers
├── utils.py              # seeding, device, filesystem helpers
└── requirements.txt      # Python dependencies
```

---

## Data Format

Place your CSV in the working directory (or pass `--csv <path>`). The loader expects these columns and internally renames them:

| CSV column    | Used as | Type  | Notes                                 |
| ------------- | ------- | ----- | ------------------------------------- |
| `Parent Cell` | `cell`  | str   | Unique cell identifier at birth       |
| `Birth Time`  | `time`  | int   | Integer time when the cell is “alive” |
| `parent_x`    | `x`     | float | Birth position (x)                    |
| `parent_y`    | `y`     | float | Birth position (y)                    |
| `parent_z`    | `z`     | float | Birth position (z)                    |
| `Daughter 1`  | `d1`    | str   | Optional daughter cell ID             |
| `Daughter 2`  | `d2`    | str   | Optional daughter cell ID             |

**Node features at birth:** `[x, y, z, time/T_max]`
**Lineage graph:** directed edges `cell → d1/d2`

---

## Working

For each time `t`:

1. **Alive set**: cells with `birth_time ≤ t`
2. **Spatial hyperedges**: KNN neighborhoods among alive nodes (size ≥ 2), controlled by `--k` and `--radius`
3. **Lineage hyperedges**: siblings alive at `t` (size ≥ 2)
4. **Snapshot**: PyG `Data` with node features, bipartite incidence, and edge‐type tags (`0=spatial`, `1=lineage`)

**Model**: `DynGrowingHNN` (default)

* Message passing: `HypergraphConv` (or `GATConv` via `--conv gat`)
* Temporal state: `GRU` (default) or `LSTM` (`--rnn lstm`)
* Optional tiny Transformer on hidden states (`--tf`)
* Readout predicts next-step `(x, y, z)` per node

**Loss**:

* `MSE(pred_xyz, target_xyz)` for next-step regression
* `incidence_bce` to reconstruct observed incidence (skipped when no hyperedges)
* `total = mse + bce`

---

## Run It [CLI]


```bash
python main.py \
  --csv cells_birth_and_pos.csv \
  --epochs 30 \
  --lr 1e-3 \
  --k 5 \
  --radius 25.0 \
  --conv hgcn \
  --rnn gru \
  --hid 64 --out 64 \
  --save_dir outputs \
  --emb outputs/embeddings.npy
```

This will:

1. Load the CSV (`data_io.py`)
2. Build time snapshots (`snapshots.py` using `hyperedges.py`)
3. Train (`train.py` with `losses.py`)
4. Save `(T×N×D)` embeddings to `outputs/embeddings.npy` (`evaluate.py`)

---

## CLI Options

```
--csv        Path to the input CSV (default: config.py)
--epochs     Training epochs (default: 30)
--lr         Learning rate (default: 1e-3)
--k          KNN size for spatial hyperedges (default: 5)
--radius     Spatial radius (default: 25.0)
--conv       Message passing: hgcn | gat (default: hgcn)
--rnn        Temporal cell: gru | lstm (default: gru)
--tf         Use a tiny Transformer encoder (flag)
--hid        Hidden size (default: 64)
--out        Readout size (default: 64; xyz uses first 3 dims)
--save_dir   Output directory (default: outputs)
--emb        Path to save embeddings .npy (default: outputs/embeddings.npy)
--seed       Global seed (default: 42)
```

---

## Outputs

* `outputs/embeddings.npy` — NumPy array with shape `(T, N, D)` (hidden state per node over time)
* Console logs print average loss per epoch

---

## Optional Visualizations

```python
import numpy as np
from visualize import umap_2d, kmeans_labels, plot_2d_umap
from data_io import load_dataset

emb = np.load("outputs/embeddings.npy")  # (T, N, D)
final = emb[-1]                           # (N, D)

coords = umap_2d(final)
labels = kmeans_labels(final, k=8)

cells = load_dataset("cells_birth_and_pos.csv")["cells"]
fig = plot_2d_umap(coords, labels, cells)
fig.show()
```

> You can also use `umap_3d` + `plot_3d_umap` for interactive 3D plots.

---

## Troubleshooting

* **Env activation name**
  If you created `ndphnn`, activate it with `conda activate ndphnn` .

* **PyTorch / PyG wheels**
  If the generic `pip install -r requirements.txt` struggles on your system, install the correct
  PyTorch + PyG (CPU or CUDA) first, then re-run the requirements install.

* **Empty snapshots**
  Check CSV column names and values. The loader renames columns as documented above and expects non-empty rows.

* **Incidence BCE skipped**
  This is normal at time steps with no hyperedges.


