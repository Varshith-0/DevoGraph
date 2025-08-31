"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

def make_hyperedges_alive(t: int,
                          cells: list,
                          df,
                          idx: dict,
                          G_lin,
                          birth_feat: np.ndarray,
                          k: int = 5,
                          spatial_r: float = 25.0) -> Tuple[list, list, dict]:
    """
    Returns (alive_idx, incidence_list, he2type) at time t.

    - Spatial hyperedges: KNN neighborhoods (size >= 2) among alive nodes.
    - Lineage hyperedges: siblings alive at t (kids of same parent) with size >= 2.
    """
    #--- alive nodes by birth time
    alive = [c for c in cells if int(df.loc[df['cell'] == c, 'time'].values[0]) <= t]
    alive_idx = [idx[c] for c in alive]

    #--- spatial hyperedges
    spatial = []
    if len(alive_idx) >= 2:
        X = birth_feat[alive_idx, :3]
        nbrs = NearestNeighbors(n_neighbors=min(k+1, len(alive_idx)),
                                radius=spatial_r).fit(X)
        _, knn = nbrs.kneighbors(X)
        for row in knn:
            he = tuple(sorted({alive_idx[i] for i in row}))
            if len(he) > 1:
                spatial.append(he)

    #--- lineage hyperedges
    lineage = []
    for p in alive:
        kids = [child for child in G_lin.successors(p)
                if int(df.loc[df['cell'] == child, 'time'].values[0]) <= t]
        if len(kids) > 1:
            lineage.append(tuple(sorted(idx[k] for k in kids)))

    #--- tag by type and deduplicate
    he2type, all_incidence = {}, []
    for he in spatial:
        he2type[he] = 'spatial'; all_incidence.append(he)
    for he in lineage:
        he2type[he] = 'lineage'; all_incidence.append(he)

    #--- unique preserve order
    seen = set(); unique = []
    for he in all_incidence:
        if he not in seen:
            unique.append(he); seen.add(he)
    return alive_idx, unique, he2type
