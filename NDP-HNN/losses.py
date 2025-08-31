"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
import torch

def incidence_bce(data, device=None):
    """
    Reconstructs the incidence by setting known memberships to +10 logits and
    unknown to -10; BCE-with-logits against the implied binary targets.
    Skips if there are no hyperedges.
    """
    if data.edge_index.numel() == 0:
        return torch.tensor(0.0, device=device or data.x.device)
    E = int(data.edge_index[1].max().item()) + 1
    logits = torch.full((data.num_nodes, E), -10.0, device=device or data.x.device)
    logits[data.edge_index[0], data.edge_index[1]] = 10.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, (logits > 0).float()
    )
