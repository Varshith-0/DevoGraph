"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

def umap_2d(final_emb: np.ndarray, n_neighbors=30, min_dist=0.1, random_state=42):
    return UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=random_state).fit_transform(final_emb)

def umap_3d(final_emb: np.ndarray, n_neighbors=30, min_dist=0.1, random_state=42):
    return UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=random_state).fit_transform(final_emb)

def kmeans_labels(final_emb: np.ndarray, k: int = 8, seed: int = 42):
    return KMeans(n_clusters=k, n_init='auto', random_state=seed).fit_predict(final_emb)

def plot_2d_umap(coords2d: np.ndarray, labels: np.ndarray, cells: list):
    df = pd.DataFrame(dict(x=coords2d[:,0], y=coords2d[:,1], cluster=labels, cell_id=cells))
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_name='cell_id',
                     title='2D UMAP of last-t embeddings')
    fig.update_layout(template='plotly_white')
    return fig

def plot_3d_umap(coords3d: np.ndarray, labels: np.ndarray, cells: list):
    df = pd.DataFrame(dict(x=coords3d[:,0], y=coords3d[:,1], z=coords3d[:,2],
                           cluster=labels, cell_id=cells))
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', hover_name='cell_id',
                        title='3D UMAP of last-t embeddings')
    fig.update_layout(template='plotly_white')
    return fig
