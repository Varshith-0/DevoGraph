# MODELING NEURAL DEVELOPMENTAL PROGRAMS OF *C. elegans* USING GROWING HYPERGRAPH NEURAL NETWORKS 

## NDP-HNN Environment Setup

This guide explains how to create a new conda environment named `ghnn` and install the required Python packages using the provided `requirements.txt`.  
It is expected to have conda/anaconda distribution already installed on your device or server. 

---

## Steps

### 1. Create the conda environment with Python 3.9 (or another version of your choice):
```bash
conda create -n ghnn python=3.9 -y
````

### 2. Activate the environment:

```bash
conda activate ghnn
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

## Static Hypergraph Visualization

We provide a script `static_hypergraph_visualization.py` to visualize a small subset of cells using a **star-expansion** hypergraph approach.

### **Dataset Required**

`CE_cell_graph_data.csv` containing columns:

```
cell, x, y, z
```

### **Run**

```bash
python static_hypergraph_visualization.py
```

### **What it Does**

* Reads the first 30 cells from the dataset.
* Divides cells into hyperedges based on x-coordinate quartiles.
* Adds **extra nodes (centroids)** for hyperedges and connects members to centroid.
* Generates random DNA-like sequences for each hyperedge for demonstration.
* Produces a **static 3D plot** using Matplotlib.

### **Output**

* A window with a static 3D plot showing:

  * Blue spheres for cell positions.
  * Triangle markers (`^`) for hyperedge centroids.
  * Lines connecting members to their centroid.
  * Hyperedge labels with randomly generated sequences.

---

## Dynamic Hypergraph Visualization

We also provide a script `dynamic_hypergraph_visualization.py` for **interactive visualization** using Plotly, which exports directly to **HTML**.

### **Dataset Required**

`cells_birth_and_pos.csv` containing columns:

```
Parent Cell, Daughter 1, Daughter 2, parent_x, parent_y, parent_z, Birth Time
```

---

### **Mode 1 – Full Star-Expansion Hypergraph**

Generates one HTML file with **all hyperedges**.

**Run:**

```bash
python -c "from dynamic_hypergraph_visualization import save_cell_division_hypergraph; \
save_cell_division_hypergraph('cells_birth_and_pos.csv', html_save='full')"
```

**Output:**

* `cells_division_hypergraph.html` – Open in any browser to view interactive 3D plot.

---

### **Mode 2 – Partial Time Window Hypergraph**

Generates **two subplots** side-by-side, showing different birth-time windows.

**Run:**

```bash
python -c "from dynamic_hypergraph_visualization import save_cell_division_hypergraph; \
save_cell_division_hypergraph('cells_birth_and_pos.csv', html_save='partial', timepoints=[0, 100])"
```

**Output:**

* `cells_division_windows_hypergraph.html` – Open in any browser for interactive exploration.

---

### **What It Does**

* Builds hyperedges for each cell division event (`Parent Cell → Daughter 1 & Daughter 2`).
* Creates centroid node for each division hyperedge.
* Links centroid to parent and daughters.
* Color codes each hyperedge.
* Supports:

  * **Full visualization** (all data)
  * **Partial window visualization** (split into two time ranges)

---

## Notes

* **Static Visualization** → Best for quick inspection of small subsets.
* **Dynamic Visualization** → Produces HTML for interactive zoom/pan/rotate and is shareable without Python.
* Place datasets in the same directory as the scripts or adjust file paths accordingly.

---

## Under Construction

Further details on **hypergraph neural network models** and **developmental program simulation** will be added soon.

