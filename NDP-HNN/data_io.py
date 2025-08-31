"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple

def load_dataset(csv_path: str) -> Dict[str, Any]:
    """
    Reads the CSV and constructs:
      df (standardized),
      cells (list[str]),
      idx (dict[cell->int]),
      T_max (int),
      birth_feat (np.ndarray [N,4]),
      G_lin (nx.DiGraph),
      birth_times (dict[cell->time])
    """
    df = pd.read_csv(csv_path)
    #--- make column names consistent with your single-file script
    df = df.rename(columns={
        'Parent Cell':'cell',
        'Birth Time':'time',
        'parent_x':'x', 'parent_y':'y', 'parent_z':'z',
        'Daughter 1':'d1', 'Daughter 2':'d2'
    })

    #--- ensure string cells
    for col in ['cell','d1','d2']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    #--- master cell list + index map
    cells = sorted(set(df['cell']) | set(df['d1']) | set(df['d2']))
    if 'nan' in cells:  # remove string "nan" artifacts
        cells.remove('nan')
    idx = {c:i for i,c in enumerate(cells)}

    T_max = int(df['time'].max())

    #--- birth features: (x,y,z, time/T_max), 
    #--- placed on parent at birth; 
    #--- daughters inherit parent birth pos
    birth_feat = np.zeros((len(cells), 4), dtype=np.float32)
    for _, r in df.iterrows():
        i = idx[r.cell]
        birth_feat[i, :3] = [r.x, r.y, r.z]
        birth_feat[i,  3] = r.time / T_max
        for child in (r.d1, r.d2):
            if pd.notna(child) and child != 'nan':
                j = idx[child]
                birth_feat[j, :3] = [r.x, r.y, r.z]
                birth_feat[j,  3] = r.time / T_max

    #--- directed lineage graph
    G_lin = nx.DiGraph()
    for _, r in df.iterrows():
        if pd.notna(r.d1) and r.d1 != 'nan':
            G_lin.add_edge(r.cell, r.d1)
        if pd.notna(r.d2) and r.d2 != 'nan':
            G_lin.add_edge(r.cell, r.d2)

    #--- birth times for alive masks
    birth_times = {}
    for _, r in df.iterrows():
        birth_times[r.cell] = int(r.time)
        for child in (r.d1, r.d2):
            if pd.notna(child) and child != 'nan':
                birth_times[child] = int(r.time)

    return dict(
        df=df, cells=cells, idx=idx, T_max=T_max,
        birth_feat=birth_feat, G_lin=G_lin, birth_times=birth_times
    )
