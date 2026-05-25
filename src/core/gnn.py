import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


class GNNFeatureExtractor(nn.Module):
    """GNN-based feature extractor for O-RAN topology using PyTorch Geometric."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.gine1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=2,
        )
        self.gine2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=2,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_data: Data) -> torch.Tensor:
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr

        x = F.relu(self.node_proj(x))
        x = F.relu(self.gine1(x, edge_index, edge_attr))
        x = F.relu(self.gine2(x, edge_index, edge_attr))

        if hasattr(graph_data, "batch") and graph_data.batch is not None:
            batch = graph_data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embed = global_mean_pool(x, batch)
        return self.output_proj(graph_embed)


class ORANGraphBuilder:
    """Convert O-RAN state to PyTorch Geometric graph representation."""

    def __init__(self, num_rhs: int, num_ess: int, num_rcs: int):
        self.num_rhs = num_rhs
        self.num_ess = num_ess
        self.num_rcs = num_rcs
        self.num_nodes = num_rhs + num_ess + num_rcs

    def build_graph(
        self,
        rh_demands: np.ndarray,
        rh_latencies: np.ndarray,
        es_remaining: np.ndarray,
        rc_remaining: np.ndarray,
        adjacency: np.ndarray,
        edge_features_dict: dict = None,
    ) -> Data:
        node_features = []

        for i in range(self.num_rhs):
            node_features.append([
                1.0,
                0.0,
                0.0,
                0.0,
                rh_demands[i] / 300.0,
                rh_latencies[i] / 200.0,
            ])

        for i in range(self.num_ess):
            node_features.append([
                0.0,
                1.0,
                0.0,
                es_remaining[i] / 20.0,
                0.0,
                0.0,
            ])

        for i in range(self.num_rcs):
            node_features.append([
                0.0,
                0.0,
                1.0,
                rc_remaining[i] / 100.0,
                0.0,
                0.0,
            ])

        x = torch.FloatTensor(node_features)

        edge_index_list = []
        edge_attr_list = []

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if adjacency[i, j] > 0:
                    edge_index_list.append([i, j])
                    edge_index_list.append([j, i])

                    if edge_features_dict and (i, j) in edge_features_dict:
                        bandwidth = edge_features_dict[(i, j)]["bandwidth"] / 160.0
                        delay = edge_features_dict[(i, j)]["delay"] / 3.6
                    else:
                        bandwidth = 0.25
                        delay = 0.5

                    edge_attr_list.append([bandwidth, delay])
                    edge_attr_list.append([bandwidth, delay])

        edge_index = torch.LongTensor(edge_index_list).t().contiguous() if edge_index_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.FloatTensor(edge_attr_list) if edge_attr_list else torch.zeros((0, 2))
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
