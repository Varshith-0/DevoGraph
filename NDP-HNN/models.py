"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GATConv

class DynHNN(nn.Module):
    """
    Simpler dynamic HNN (your first variant).
    """
    def __init__(self, in_dim=4, hid_dim=64, out_dim=64, num_edge_types=2):
        super().__init__()
        self.hconvs = nn.ModuleList(
            [HypergraphConv(in_dim, hid_dim) for _ in range(num_edge_types)]
        )
        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)
        self.gru     = nn.GRUCell(hid_dim, hid_dim)
        self.readout = nn.Linear(hid_dim, out_dim)

    def forward(self, data, h_prev=None):
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei   = data.edge_index[:, mask] if mask.numel() > 0 else data.edge_index[:, :0]
            outs.append(conv(data.x, ei))
        h = torch.relu(self.lin_mix(torch.cat(outs, dim=1)))
        h_next = h if h_prev is None else self.gru(h, h_prev)
        pred_xyz = self.readout(h_next)[:, :3]
        return h_next, pred_xyz

class DynGrowingHNN(nn.Module):
    """
    Fully-configurable dynamic hypergraph model.
    """
    def __init__(self,
                 in_dim=4, hid_dim=64, out_dim=64,
                 num_edge_types=2,
                 conv_type="hgcn", conv_kwargs=None,
                 rnn_type="gru",   rnn_kwargs=None,
                 use_transformer=False, transformer_kwargs=None,
                 readout_dim=None):
        super().__init__()
        readout_dim = out_dim if readout_dim is None else readout_dim

        conv_kwargs = conv_kwargs or {}
        if isinstance(conv_type, str):
            ct = conv_type.lower()
            if ct == "hgcn":
                Conv = HypergraphConv
            elif ct == "gat":
                Conv = GATConv
                default_gat = {"heads": 4, "concat": False}
                default_gat.update(conv_kwargs)
                conv_kwargs = default_gat
            else:
                raise ValueError(f"unknown conv_type `{conv_type}`")
        else:
            Conv = conv_type

        self.hconvs = nn.ModuleList([
            Conv(in_dim, hid_dim, **conv_kwargs)
            for _ in range(num_edge_types)
        ])

        self.lin_mix = nn.Linear(num_edge_types * hid_dim, hid_dim)

        rnn_type = rnn_type.lower()
        rnn_kwargs = rnn_kwargs or {}
        if rnn_type == "lstm":
            self.rnn = nn.LSTMCell(hid_dim, hid_dim, **rnn_kwargs)
            self._is_lstm = True
        else:
            self.rnn = nn.GRUCell(hid_dim, hid_dim, **rnn_kwargs)
            self._is_lstm = False

        self.use_transformer = bool(use_transformer)
        if self.use_transformer:
            d_model = hid_dim
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
                num_layers=1
            )

        self.readout = nn.Linear(hid_dim, readout_dim)

    def forward(self, data, state_prev=None):
        outs = []
        for etype, conv in enumerate(self.hconvs):
            mask = (data.edge_attr == etype).nonzero(as_tuple=True)[0]
            ei   = data.edge_index[:, mask] if mask.numel() > 0 else data.edge_index[:, :0]
            outs.append(conv(data.x, ei))
        h = torch.relu(self.lin_mix(torch.cat(outs, dim=1)))

        if self.use_transformer:
            h_seq = h.unsqueeze(1)              # (N,1,H)
            h_seq = self.tf(h_seq)              # (N,1,H)
            h     = h_seq.squeeze(1)            # (N,H)

        if self._is_lstm:
            if state_prev is None:
                h_next, c_next = self.rnn(h)
            else:
                h_prev, c_prev = state_prev
                h_next, c_next = self.rnn(h, (h_prev, c_prev))
            state_next = (h_next, c_next)
        else:
            h_next     = self.rnn(h, state_prev) if state_prev is not None else self.rnn(h)
            state_next = h_next

        out = self.readout(h_next)[:, :3]
        return state_next, out
