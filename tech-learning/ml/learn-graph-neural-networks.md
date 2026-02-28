# Graph Neural Networks (GNNs): Complete Guide

## Table of Contents
1. [Introduction to Graph Neural Networks](#introduction-to-graph-neural-networks)
2. [Graph Fundamentals](#graph-fundamentals)
3. [Message Passing Paradigm](#message-passing-paradigm)
4. [GNN Architectures](#gnn-architectures)
5. [Graph Convolutional Networks (GCN)](#graph-convolutional-networks-gcn)
6. [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
7. [GraphSAGE and Inductive Learning](#graphsage-and-inductive-learning)
8. [Temporal and Dynamic Graphs](#temporal-and-dynamic-graphs)
9. [Heterogeneous Graphs](#heterogeneous-graphs)
10. [Practical Examples](#practical-examples)
11. [Advanced Topics](#advanced-topics)
12. [Best Practices](#best-practices)

---

## Introduction to Graph Neural Networks

Graph Neural Networks (GNNs) are deep learning architectures designed to operate on **graph-structured data**. Unlike CNNs (for grids) or RNNs (for sequences), GNNs capture relationships and dependencies between entities through graph connectivity.

### Why Graphs?

Many real-world problems are naturally graph-structured:

| Domain | Graph Structure | Application |
|--------|-----------------|-------------|
| **Social Networks** | Users as nodes, follows as edges | Recommendation, influence prediction |
| **Molecular Chemistry** | Atoms as nodes, bonds as edges | Drug discovery, property prediction |
| **Knowledge Graphs** | Entities as nodes, relations as edges | Question answering, link prediction |
| **Recommendation** | Users + items as nodes | Collaborative filtering |
| **Traffic Networks** | Intersections as nodes, roads as edges | Traffic prediction |
| **Code Analysis** | Code elements as nodes | Code understanding, vulnerability detection |

### Key GNN Concepts

- **Message Passing**: Nodes aggregate information from neighbors
- **Node Embeddings**: Learn vector representations capturing graph structure
- **Graph-level Tasks**: Node, edge, or whole-graph classification
- **Inductive vs Transductive**: Generalize to new nodes vs fixed graph

### GNN vs Other Architectures

```python
# CNN: Fixed grid (image pixels) - local receptive fields
# RNN: Sequential - processes tokens one at a time
# Transformer: Fully connected attention - O(n²) for sequence
# GNN: Sparse structure - only processes connected nodes (efficient for graphs)
```

---

## Graph Fundamentals

### Graph Representation

A graph G = (V, E) consists of:
- **V**: Set of nodes (vertices)
- **E**: Set of edges (connections)
- **Node features**: X ∈ ℝ^(n×d) (n nodes, d features each)
- **Adjacency matrix**: A ∈ ℝ^(n×n) where A[i,j] = 1 if edge exists

```python
import numpy as np
import torch
from torch_geometric.data import Data

# Simple graph: 4 nodes, 4 edges
# 0 -- 1
# |    |
# 2 -- 3

# Edge list representation (PyTorch Geometric format)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
    [1, 0, 3, 2, 3, 2, 0, 3]   # Target nodes
], dtype=torch.long)

# Node features (4 nodes, 3 features each)
x = torch.tensor([
    [1.0, 0.0, 0.0],  # Node 0
    [0.0, 1.0, 0.0],  # Node 1
    [0.0, 0.0, 1.0],  # Node 2
    [1.0, 1.0, 0.0],  # Node 3
], dtype=torch.float)

# Create PyG Data object
data = Data(x=x, edge_index=edge_index)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Node features shape: {data.x.shape}")
```

### Common Graph Types

```python
# 1. Undirected graph (symmetric adjacency)
# 2. Directed graph (e.g., Twitter follow)
# 3. Weighted graph (edge weights = connection strength)
# 4. Multiplex graph (multiple edge types)
# 5. Heterogeneous graph (multiple node/edge types)

# Example: Creating weighted graph
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
edge_weight = torch.tensor([0.5, 0.8, 1.0])  # Edge weights
```

### Graph Statistics

```python
def graph_statistics(edge_index, num_nodes):
    """Compute basic graph statistics"""
    in_degree = torch.zeros(num_nodes)
    out_degree = torch.zeros(num_nodes)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        out_degree[src] += 1
        in_degree[dst] += 1
    
    return {
        'avg_degree': (in_degree + out_degree).mean().item(),
        'density': edge_index.shape[1] / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    }
```

---

## Message Passing Paradigm

The core of GNNs: **aggregate** neighbor information and **update** node representations.

### Message Passing Equation

At layer k, for each node v:

1. **Message**: m_uv = MESSAGE(h_u^(k-1), h_v^(k-1), e_uv)
2. **Aggregate**: m_v = AGGREGATE({m_uv : u ∈ N(v)})
3. **Update**: h_v^(k) = UPDATE(h_v^(k-1), m_v)

Where h_u = node embedding, e_uv = edge features, N(v) = neighbors of v.

```python
import torch
import torch.nn as nn
from torch_scatter import scatter_add

class SimpleMessagePassing(nn.Module):
    """Basic message passing layer"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.message_net = nn.Linear(2 * in_dim, out_dim)
        self.update_net = nn.Linear(in_dim + out_dim, out_dim)
    
    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Create messages: concat source and target node features
        x_src = x[src]  # [num_edges, in_dim]
        x_dst = x[dst]  # [num_edges, in_dim]
        messages = self.message_net(torch.cat([x_src, x_dst], dim=-1))  # [num_edges, out_dim]
        
        # 2. Aggregate: sum messages by destination node
        aggregated = scatter_add(messages, dst, dim=0, dim_size=x.size(0))
        
        # 3. Update: combine old embedding with aggregated messages
        combined = torch.cat([x, aggregated], dim=-1)
        out = self.update_net(combined)
        
        return out

# Example usage
mp = SimpleMessagePassing(in_dim=16, out_dim=32)
x = torch.randn(100, 16)  # 100 nodes, 16-dim features
edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
out = mp(x, edge_index)  # [100, 32]
```

### Permutation Invariance

GNN layers must be **permutation invariant**—output shouldn't change if we reorder nodes. Sum, mean, and max aggregation satisfy this.

```python
# Valid aggregations (permutation invariant):
# - Sum: Σ m_uv
# - Mean: (1/|N(v)|) Σ m_uv  
# - Max: max(m_uv)

# Invalid: Concatenation (order-dependent)
```

---

## Graph Convolutional Networks (GCN)

GCN (Kipf & Welling, 2017) applies **spectral graph convolution** with a simplified filter.

### GCN Layer Formula

h_v^(k) = σ( Σ_{u∈N(v)∪{v}} (1/√(d_v·d_u)) · W · h_u^(k-1) )

Where d_v = degree of node v, W = learnable weight matrix.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Node classification example
model = GCN(in_dim=128, hidden_dim=64, out_dim=7)  # 7 classes
logits = model(node_features, edge_index)
loss = nn.functional.cross_entropy(logits[train_mask], labels[train_mask])
```

### GCN with Batch Normalization

```python
from torch_geometric.nn import GCNConv, BatchNorm

class GCNWithNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

---

## Graph Attention Networks (GAT)

GAT (Veličković et al., 2018) uses **attention** to learn importance of each neighbor.

### Attention Mechanism

α_uv = softmax_v( LeakyReLU(a^T [W·h_u ‖ W·h_v]) )

h_v' = σ( Σ_{u∈N(v)} α_uv · W · h_u )

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = torch.elu(x)
        x = nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Multi-head attention allows different importance for different relation types
model = GAT(in_dim=1433, hidden_dim=8, out_dim=7, heads=8)
```

### Why GAT over GCN?

- **Adaptive weighting**: Important neighbors get higher attention
- **Interpretability**: Can inspect attention weights
- **Handles varying degree**: No fixed normalization
- **Multi-head**: Captures different relationship aspects

---

## GraphSAGE and Inductive Learning

GraphSAGE (Hamilton et al., 2017) enables **inductive learning**—generalization to unseen nodes/graphs.

### Sampling and Aggregation

Instead of using full neighborhood, GraphSAGE:
1. **Samples** k neighbors at each layer
2. **Aggregates** with learnable functions (Mean, LSTM, Pool)
3. **Concatenates** own features with neighbor aggregate

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Inductive: Train on subgraph, test on new nodes
# Useful for: recommendation (new users), molecular (new compounds)
```

### Mini-batch Training for Large Graphs

```python
from torch_geometric.loader import NeighborLoader

# Create mini-batch loader (samples subgraphs)
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 2-hop: 25 then 10 neighbors
    batch_size=128,
    input_nodes=data.train_mask,
    num_workers=0
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()
```

---

## Temporal and Dynamic Graphs

Graphs that evolve over time (e.g., social networks, traffic).

### Temporal Graph Networks (TGN)

```python
# Key idea: Embeddings depend on time
# - Store memory for each node
# - Update memory when new edge observed
# - Use temporal encoding (e.g., Time2Vec)

# Simplified temporal message passing
def temporal_aggregate(node_states, edge_times, current_time):
    # Weight by recency: recent interactions matter more
    time_decay = torch.exp(-0.1 * (current_time - edge_times))
    return (node_states * time_decay.unsqueeze(-1)).sum(0) / time_decay.sum()
```

### Dynamic Graph Attention

```python
# PyTorch Geometric Temporal
# pip install torch-geometric-temporal

from torch_geometric_temporal import TGCN

# TGCN: Combines GCN with GRU for temporal dynamics
# Input: Sequence of graph snapshots
# Output: Node predictions at next timestep
```

---

## Heterogeneous Graphs

Graphs with **multiple node types** (users, items, categories) and **edge types** (bought, viewed, rated).

### Heterogeneous GNN (HeteroConv)

```python
from torch_geometric.nn import HGTConv, Linear
import torch_geometric
from torch_geometric.data import HeteroData

# Define heterogeneous graph
data = HeteroData()
data['user'].x = torch.randn(1000, 64)   # 1000 users
data['item'].x = torch.randn(5000, 64)    # 5000 items
data['user', 'buys', 'item'].edge_index = torch.randint(0, 1000, (2, 10000))
data['user', 'views', 'item'].edge_index = torch.randint(0, 1000, (2, 50000))

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, metadata):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=4))
        self.lin = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['user'])
```

### Knowledge Graph Embeddings

```python
# For link prediction in knowledge graphs: (head, relation, tail)
# Models: TransE, RotatE, CompGCN

# CompGCN: Composition-based GCN for KGs
# Learns relation-specific transformations
```

---

## Practical Examples

### Example 1: Node Classification (Cora Dataset)

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCNNodeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCNNodeClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
```

### Example 2: Graph Classification (Molecular Graphs)

```python
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # Graph-level: pool node embeddings
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# batch: assigns each node to its graph
# global_mean_pool: mean of node embeddings per graph
```

### Example 3: Link Prediction

```python
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(2 * hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index[0], edge_index[1]
        edge_emb = torch.cat([z[src], z[dst]], dim=-1)
        return (self.lin(edge_emb)).squeeze(-1).sigmoid()

model = LinkPredictor(in_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train: positive edges vs negative samples
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    pos_edge = data.edge_index
    neg_edge = negative_sampling(edge_index=pos_edge, num_nodes=data.num_nodes, num_neg_samples=pos_edge.size(1))
    pos_score = model.decode(z, pos_edge)
    neg_score = model.decode(z, neg_edge)
    loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score)) + \
           F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    loss.backward()
    optimizer.step()
```

---

## Advanced Topics

### Over-smoothing Problem

**Issue**: Deep GNNs cause node embeddings to become similar (over-smoothing).
**Solutions**:
- Residual connections: h^(k) = h^(k-1) + f(h^(k-1))
- PairNorm: Normalize to preserve total pairwise distance
- Jumping knowledge: Concatenate embeddings from all layers

```python
# Residual GCN
class ResGCN(torch.nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = h + self.conv2(h, edge_index).relu()  # Residual
        h = self.conv3(h, edge_index)
        return h
```

### Expressiveness: WL Test

GNNs are at most as expressive as the Weisfeiler-Lehman (WL) graph isomorphism test. **Graph Isomorphism Networks (GIN)** match WL expressiveness.

```python
from torch_geometric.nn import GINConv
import torch.nn as nn

# GIN: Sum aggregation + MLP
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = GINConv(self.mlp, train_eps=True)
```

### Scalability: GraphSAINT, Cluster-GCN

For billion-scale graphs:
- **GraphSAINT**: Sample subgraphs, train on them
- **Cluster-GCN**: Partition graph, train on clusters
- ** scalable-GNN**: Approximate aggregation

---

## Best Practices

1. **Normalize features**: Graph-level batch norm or LayerNorm
2. **Dropout**: Use dropout (0.5-0.6) for regularization
3. **Learning rate**: Lower LR (0.001-0.01) than vision/NLP
4. **Early stopping**: Monitor validation loss
5. **Edge sampling**: For large graphs, use neighbor sampling
6. **Feature preprocessing**: Consider Laplacian normalization
7. **Choose architecture**: GCN for homogeneity, GAT for varying importance, GraphSAGE for inductive

---

## Summary

| Architecture | Use Case | Key Feature |
|--------------|----------|-------------|
| GCN | Node classification, homogeneity | Spectral convolution |
| GAT | Varying neighbor importance | Attention weights |
| GraphSAGE | Inductive, large graphs | Neighbor sampling |
| GIN | Expressiveness | WL-equivalent |
| HGT | Heterogeneous graphs | Type-specific attention |

**Installation**: `pip install torch-geometric torch-scatter torch-sparse`
