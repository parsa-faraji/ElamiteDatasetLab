#!/usr/bin/env python3
"""
Elamite Network Visualization Script

Creates publication-quality static visualizations of the word network.
Requires: matplotlib, networkx (pip install matplotlib networkx)

Outputs:
- network_overview.png: Full network with hub nodes labeled
- network_hub_subgraph.png: Core hub words and their connections
- similarity_heatmap.png: Heatmap of top word similarities
- morphological_distribution.png: Suffix and determinative distributions
"""

import json
import csv
import sys

# Check dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    import numpy as np
    from collections import Counter, defaultdict
except ImportError as e:
    print("Missing required packages. Install with:")
    print("  pip install matplotlib networkx numpy")
    sys.exit(1)

print("=" * 60)
print("ELAMITE NETWORK VISUALIZATION GENERATOR")
print("=" * 60)

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#0f3460'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['xtick.color'] = '#aaa'
plt.rcParams['ytick.color'] = '#aaa'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['font.size'] = 10

# Load data
print("\n[1] Loading data...")
with open('elamite_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['word']: n for n in data['nodes']}
edges = data['edges']

print(f"    Loaded {len(nodes)} nodes, {len(edges)} edges")

# Build NetworkX graph
print("\n[2] Building network graph...")
G = nx.Graph()

for word, attrs in nodes.items():
    G.add_node(word, **attrs)

for edge in edges:
    G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Compute metrics
print("\n[3] Computing network metrics...")
degree_dict = dict(G.degree())
strength_dict = dict(G.degree(weight='weight'))
betweenness = nx.betweenness_centrality(G, weight='weight')
pagerank = nx.pagerank(G, weight='weight')

# Color scheme
det_colors = {
    'divine': '#e94560',
    'divine-personal': '#ff6b6b',
    'personal-male': '#4ecdc4',
    'personal-female': '#ffe66d',
    None: '#6c757d',
    '': '#6c757d'
}

suffix_colors = {
    'me': '#e94560',
    'ak': '#4ecdc4',
    'ik': '#ffe66d',
    'ra': '#95e1d3',
    'na': '#f38181',
    'ni': '#aa96da',
    'ka': '#fcbad3',
    'ir': '#a8d8ea',
    None: '#6c757d',
    '': '#6c757d'
}

# =============================================================================
# Visualization 1: Full Network Overview
# =============================================================================
print("\n[4] Creating network overview...")

fig, ax = plt.subplots(figsize=(16, 12))

# Use spring layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node colors by determinative
node_colors = [det_colors.get(nodes[n].get('determinative'), '#6c757d') for n in G.nodes()]

# Node sizes by degree
max_degree = max(degree_dict.values()) if degree_dict else 1
node_sizes = [100 + (degree_dict.get(n, 0) / max_degree) * 800 for n in G.nodes()]

# Edge widths by weight
edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]

# Draw network
nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, edge_color='#0f3460', ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)

# Label high-degree nodes
top_nodes = sorted(degree_dict.items(), key=lambda x: -x[1])[:20]
labels = {n: n for n, _ in top_nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', ax=ax)

# Legend
legend_patches = [
    mpatches.Patch(color='#e94560', label='Divine (d)'),
    mpatches.Patch(color='#ff6b6b', label='Divine-Personal (md)'),
    mpatches.Patch(color='#4ecdc4', label='Personal (m)'),
    mpatches.Patch(color='#6c757d', label='Other'),
]
ax.legend(handles=legend_patches, loc='upper left', framealpha=0.8)

ax.set_title('Elamite Word Network - Full Overview\n(Node size = degree, color = determinative type)',
             fontsize=14, fontweight='bold', color='#e94560')
ax.axis('off')

plt.tight_layout()
plt.savefig('network_overview.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: network_overview.png")

# =============================================================================
# Visualization 2: Hub Subgraph (Core Network)
# =============================================================================
print("\n[5] Creating hub subgraph...")

# Extract top 30 nodes and their connections
hub_nodes = [n for n, _ in sorted(degree_dict.items(), key=lambda x: -x[1])[:30]]
hub_subgraph = G.subgraph(hub_nodes).copy()

fig, ax = plt.subplots(figsize=(14, 10))

pos_hub = nx.spring_layout(hub_subgraph, k=3, iterations=100, seed=42)

# Colors by suffix
hub_colors = [suffix_colors.get(nodes[n].get('suffix'), '#6c757d') for n in hub_subgraph.nodes()]
hub_sizes = [200 + degree_dict.get(n, 0) * 30 for n in hub_subgraph.nodes()]

edge_weights_hub = [hub_subgraph[u][v]['weight'] * 4 for u, v in hub_subgraph.edges()]

nx.draw_networkx_edges(hub_subgraph, pos_hub, alpha=0.5, width=edge_weights_hub,
                       edge_color='#0f3460', ax=ax)
nx.draw_networkx_nodes(hub_subgraph, pos_hub, node_color=hub_colors,
                       node_size=hub_sizes, alpha=0.9, ax=ax)
nx.draw_networkx_labels(hub_subgraph, pos_hub, font_size=9, font_color='white', ax=ax)

# Add edge labels for strongest connections
edge_labels = {(u, v): f'{hub_subgraph[u][v]["weight"]:.2f}'
               for u, v in hub_subgraph.edges() if hub_subgraph[u][v]['weight'] > 0.5}
nx.draw_networkx_edge_labels(hub_subgraph, pos_hub, edge_labels,
                             font_size=7, font_color='#aaa', ax=ax)

# Legend
suffix_patches = [
    mpatches.Patch(color='#e94560', label='-me'),
    mpatches.Patch(color='#4ecdc4', label='-ak'),
    mpatches.Patch(color='#ffe66d', label='-ik'),
    mpatches.Patch(color='#95e1d3', label='-ra'),
    mpatches.Patch(color='#fcbad3', label='-ka'),
    mpatches.Patch(color='#6c757d', label='other'),
]
ax.legend(handles=suffix_patches, loc='upper left', framealpha=0.8, title='Suffix')

ax.set_title('Core Hub Words - Top 30 by Connectivity\n(Node size = degree, color = suffix pattern)',
             fontsize=14, fontweight='bold', color='#e94560')
ax.axis('off')

plt.tight_layout()
plt.savefig('network_hub_subgraph.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: network_hub_subgraph.png")

# =============================================================================
# Visualization 3: Similarity Heatmap
# =============================================================================
print("\n[6] Creating similarity heatmap...")

# Get top 25 words by degree
top25 = [n for n, _ in sorted(degree_dict.items(), key=lambda x: -x[1])[:25]]

# Build similarity matrix
sim_matrix = np.zeros((25, 25))
for i, w1 in enumerate(top25):
    for j, w2 in enumerate(top25):
        if G.has_edge(w1, w2):
            sim_matrix[i, j] = G[w1][w2]['weight']
        elif i == j:
            sim_matrix[i, j] = 1.0

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(sim_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)

ax.set_xticks(range(25))
ax.set_yticks(range(25))
ax.set_xticklabels(top25, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(top25, fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Cosine Similarity', color='#eee')

# Add value annotations for high similarities
for i in range(25):
    for j in range(25):
        if sim_matrix[i, j] > 0.4 and i != j:
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                   fontsize=7, color='white')

ax.set_title('Similarity Heatmap - Top 25 Hub Words\n(Higher values indicate stronger distributional similarity)',
             fontsize=14, fontweight='bold', color='#e94560', pad=20)

plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: similarity_heatmap.png")

# =============================================================================
# Visualization 4: Morphological Distribution
# =============================================================================
print("\n[7] Creating morphological distribution charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Suffix distribution
suffixes = [nodes[n].get('suffix', '') or 'none' for n in nodes]
suffix_counts = Counter(suffixes)
del suffix_counts['none']

ax = axes[0, 0]
bars = ax.barh(list(suffix_counts.keys())[:10],
               list(suffix_counts.values())[:10],
               color=[suffix_colors.get(s, '#6c757d') for s in list(suffix_counts.keys())[:10]])
ax.set_xlabel('Count')
ax.set_title('Suffix Distribution (Top 10)', fontsize=12, fontweight='bold', color='#e94560')
ax.invert_yaxis()

# Determinative distribution
dets = [nodes[n].get('determinative', '') or 'none' for n in nodes]
det_counts = Counter(dets)
del det_counts['none']

ax = axes[0, 1]
colors = [det_colors.get(d, '#6c757d') for d in det_counts.keys()]
ax.pie(det_counts.values(), labels=det_counts.keys(), colors=colors,
       autopct='%1.0f%%', textprops={'color': 'white'})
ax.set_title('Determinative Distribution', fontsize=12, fontweight='bold', color='#e94560')

# Degree distribution
ax = axes[1, 0]
degrees = list(degree_dict.values())
ax.hist(degrees, bins=30, color='#e94560', alpha=0.7, edgecolor='#0f3460')
ax.set_xlabel('Degree (Number of Connections)')
ax.set_ylabel('Number of Words')
ax.set_title('Degree Distribution', fontsize=12, fontweight='bold', color='#e94560')
ax.axvline(np.mean(degrees), color='#4ecdc4', linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
ax.legend()

# Strength vs Degree scatter
ax = axes[1, 1]
x = [degree_dict.get(n, 0) for n in nodes]
y = [strength_dict.get(n, 0) for n in nodes]
colors = [det_colors.get(nodes[n].get('determinative'), '#6c757d') for n in nodes]
ax.scatter(x, y, c=colors, alpha=0.6, s=30)
ax.set_xlabel('Degree (Connections)')
ax.set_ylabel('Strength (Weighted Degree)')
ax.set_title('Degree vs Strength', fontsize=12, fontweight='bold', color='#e94560')

# Label outliers
for n in nodes:
    if degree_dict.get(n, 0) > 35:
        ax.annotate(n, (degree_dict[n], strength_dict[n]),
                   fontsize=7, color='white', alpha=0.8)

plt.tight_layout()
plt.savefig('morphological_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: morphological_distribution.png")

# =============================================================================
# Visualization 5: Community Structure
# =============================================================================
print("\n[8] Creating community visualization...")

# Detect communities using Louvain-like algorithm (greedy modularity)
try:
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight='weight'))
except:
    # Fallback: use connected components
    communities = list(nx.connected_components(G))

print(f"    Detected {len(communities)} communities")

# Get largest communities
large_communities = sorted(communities, key=len, reverse=True)[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
community_colors = ['#e94560', '#4ecdc4', '#ffe66d', '#95e1d3', '#f38181', '#aa96da']

for idx, (ax, comm) in enumerate(zip(axes.flat, large_communities)):
    subgraph = G.subgraph(comm).copy()
    pos_comm = nx.spring_layout(subgraph, k=2, seed=42)

    # Node sizes by degree within community
    comm_degrees = dict(subgraph.degree())
    max_d = max(comm_degrees.values()) if comm_degrees else 1
    sizes = [100 + (comm_degrees.get(n, 0) / max_d) * 500 for n in subgraph.nodes()]

    nx.draw_networkx_edges(subgraph, pos_comm, alpha=0.4, width=1, ax=ax)
    nx.draw_networkx_nodes(subgraph, pos_comm, node_color=community_colors[idx],
                          node_size=sizes, alpha=0.8, ax=ax)

    # Label top 5 in community
    top_in_comm = sorted(comm_degrees.items(), key=lambda x: -x[1])[:5]
    labels = {n: n for n, _ in top_in_comm}
    nx.draw_networkx_labels(subgraph, pos_comm, labels, font_size=8, ax=ax)

    ax.set_title(f'Community {idx+1} ({len(comm)} words)', fontsize=11,
                fontweight='bold', color=community_colors[idx])
    ax.axis('off')

plt.suptitle('Largest Word Communities\n(Words grouped by similar usage patterns)',
            fontsize=14, fontweight='bold', color='#e94560', y=1.02)
plt.tight_layout()
plt.savefig('community_structure.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: community_structure.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("\nGenerated files:")
print("  - network_overview.png        (full network graph)")
print("  - network_hub_subgraph.png    (core hub words)")
print("  - similarity_heatmap.png      (word similarity matrix)")
print("  - morphological_distribution.png (suffix/determinative stats)")
print("  - community_structure.png     (word communities)")
print("=" * 60)
