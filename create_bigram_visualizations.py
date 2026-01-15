#!/usr/bin/env python3
"""
Bi-gram + Similarity Network Visualizations

Creates publication-quality static visualizations for the combined
syntagmatic + paradigmatic network analysis.
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
from collections import Counter, defaultdict

print("=" * 60)
print("BI-GRAM + SIMILARITY VISUALIZATION GENERATOR")
print("=" * 60)

# Set dark style
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
with open('bigram_similarity_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data['nodes']}
edges = data['edges']

# Load centrality data
centrality = {}
with open('nodes_centrality.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        centrality[row['word']] = {
            'degree': int(row['degree']),
            'strength': float(row['strength']),
            'eigenvector': float(row['eigenvector']),
            'bridging': float(row['bridging']),
            'pc1': float(row['pc1']),
            'pc2': float(row['pc2'])
        }

print(f"    Loaded {len(nodes)} nodes, {len(edges)} edges")

# Categorize edges
bigram_only = [e for e in edges if e['type'] == 'bigram']
similarity_only = [e for e in edges if e['type'] == 'similarity']
reinforced = [e for e in edges if e['type'] == 'both']

print(f"    Bi-gram only: {len(bigram_only)}")
print(f"    Similarity only: {len(similarity_only)}")
print(f"    Reinforced: {len(reinforced)}")

# =============================================================================
# Visualization 1: Reinforced Edges Network (Fixed Constructions)
# =============================================================================
print("\n[2] Creating reinforced edges visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Build graph with only reinforced edges
G_reinforced = nx.Graph()
for e in reinforced:
    G_reinforced.add_edge(e['source'], e['target'], weight=e['total'],
                          bigram=e['bigram'], similarity=e['similarity'])

# Add isolated high-centrality nodes for context
for node, cent in sorted(centrality.items(), key=lambda x: -x[1]['eigenvector'])[:30]:
    if node not in G_reinforced:
        G_reinforced.add_node(node)

pos = nx.spring_layout(G_reinforced, k=3, iterations=100, seed=42)

# Node sizes by eigenvector
max_eigen = max(centrality[n]['eigenvector'] for n in G_reinforced.nodes() if n in centrality)
node_sizes = [100 + (centrality.get(n, {}).get('eigenvector', 0) / max_eigen) * 800
              for n in G_reinforced.nodes()]

# Node colors: reinforced nodes in gold, others in gray
node_colors = ['#ffe66d' if G_reinforced.degree(n) > 0 else '#6c757d'
               for n in G_reinforced.nodes()]

# Edge widths
edge_weights = [G_reinforced[u][v]['weight'] * 3 for u, v in G_reinforced.edges()]

# Draw
nx.draw_networkx_edges(G_reinforced, pos, alpha=0.8, width=edge_weights,
                       edge_color='#ffe66d', ax=ax)
nx.draw_networkx_nodes(G_reinforced, pos, node_color=node_colors,
                       node_size=node_sizes, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_reinforced, pos, font_size=8, font_color='white', ax=ax)

# Add edge labels for top pairs
edge_labels = {(u, v): f'{G_reinforced[u][v]["similarity"]:.2f}'
               for u, v in G_reinforced.edges() if G_reinforced[u][v]['similarity'] > 0.7}
nx.draw_networkx_edge_labels(G_reinforced, pos, edge_labels,
                             font_size=7, font_color='#ffe66d', ax=ax)

ax.set_title('Fixed Constructions: Reinforced Edges\n(Words that are BOTH adjacent AND distributionally similar)',
             fontsize=14, fontweight='bold', color='#ffe66d', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('reinforced_constructions.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: reinforced_constructions.png")

# =============================================================================
# Visualization 2: Centrality Comparison
# =============================================================================
print("\n[3] Creating centrality comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Get top words by each centrality
top_eigen = sorted(centrality.items(), key=lambda x: -x[1]['eigenvector'])[:20]
top_bridge = sorted(centrality.items(), key=lambda x: -x[1]['bridging'])[:20]

# Eigenvector centrality
ax = axes[0]
words = [w for w, _ in top_eigen]
values = [c['eigenvector'] for _, c in top_eigen]
colors = ['#e94560' if centrality[w]['bridging'] > 30 else '#4ecdc4' for w in words]

bars = ax.barh(range(len(words)), values, color=colors)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Eigenvector Centrality')
ax.set_title('Top 20 by Eigenvector Centrality\n(Connected to important words)',
             fontsize=12, fontweight='bold', color='#e94560')

# Bridging centrality
ax = axes[1]
words = [w for w, _ in top_bridge]
values = [c['bridging'] for _, c in top_bridge]
colors = ['#4ecdc4' if centrality[w]['eigenvector'] > 0.15 else '#6c757d' for w in words]

bars = ax.barh(range(len(words)), values, color=colors)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Bridging Centrality')
ax.set_title('Top 20 by Bridging Centrality\n(Connects different clusters - clause boundaries?)',
             fontsize=12, fontweight='bold', color='#4ecdc4')

# Legend
legend_elements = [
    mpatches.Patch(color='#e94560', label='High bridging (>30)'),
    mpatches.Patch(color='#4ecdc4', label='High eigenvector (>0.15)'),
    mpatches.Patch(color='#6c757d', label='Other')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('centrality_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: centrality_comparison.png")

# =============================================================================
# Visualization 3: Edge Type Distribution
# =============================================================================
print("\n[4] Creating edge type analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Pie chart of edge types
ax = axes[0, 0]
sizes = [len(bigram_only), len(similarity_only), len(reinforced)]
labels = ['Bi-gram Only\n(Sequential)', 'Similarity Only\n(Distributional)', 'Reinforced\n(Both)']
colors = ['#4ecdc4', '#e94560', '#ffe66d']
explode = (0, 0, 0.1)

ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'color': 'white', 'fontsize': 10})
ax.set_title('Edge Type Distribution', fontsize=12, fontweight='bold', color='#e94560')

# Top bi-grams
ax = axes[0, 1]
top_bigrams = sorted([(e['source'], e['target'], e['bigram'])
                      for e in edges if e['bigram'] > 0],
                     key=lambda x: -x[2])[:15]

labels = [f"{s} — {t}" for s, t, _ in top_bigrams]
values = [v for _, _, v in top_bigrams]

ax.barh(range(len(labels)), values, color='#4ecdc4')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Bi-gram Weight (frequency × 0.15)')
ax.set_title('Top Sequential Pairs (Bi-grams)', fontsize=12, fontweight='bold', color='#4ecdc4')

# Top similarity edges
ax = axes[1, 0]
top_sim = sorted([(e['source'], e['target'], e['similarity'])
                  for e in edges if e['similarity'] > 0],
                 key=lambda x: -x[2])[:15]

labels = [f"{s} — {t}" for s, t, _ in top_sim]
values = [v for _, _, v in top_sim]

ax.barh(range(len(labels)), values, color='#e94560')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Similarity Weight (cosine × 1.5)')
ax.set_title('Top Distributional Pairs (Similarity)', fontsize=12, fontweight='bold', color='#e94560')

# Top reinforced edges
ax = axes[1, 1]
top_reinforced = sorted([(e['source'], e['target'], e['total'], e['similarity'])
                         for e in reinforced],
                        key=lambda x: -x[2])[:15]

labels = [f"{s} — {t}" for s, t, _, _ in top_reinforced]
values = [v for _, _, v, _ in top_reinforced]
sims = [s for _, _, _, s in top_reinforced]

bars = ax.barh(range(len(labels)), values, color='#ffe66d')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Total Weight (bi-gram + similarity)')
ax.set_title('Top Fixed Constructions (Reinforced)', fontsize=12, fontweight='bold', color='#ffe66d')

# Add similarity scores as text
for i, (bar, sim) in enumerate(zip(bars, sims)):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'sim={sim:.2f}', va='center', fontsize=7, color='#aaa')

plt.tight_layout()
plt.savefig('edge_type_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: edge_type_analysis.png")

# =============================================================================
# Visualization 4: PCA Document Space
# =============================================================================
print("\n[5] Creating PCA document visualization...")

# Load document PCA
doc_pca = []
with open('document_pca.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_pca.append({
            'doc_id': row['doc_id'],
            'word_count': int(row['word_count']),
            'pc1': float(row['pc1']),
            'pc2': float(row['pc2'])
        })

fig, ax = plt.subplots(figsize=(12, 10))

x = [d['pc1'] for d in doc_pca]
y = [d['pc2'] for d in doc_pca]
sizes = [d['word_count'] * 3 for d in doc_pca]

# Color by PC1 (text preservation quality)
colors = ['#e94560' if d['pc1'] > 0 else '#4ecdc4' for d in doc_pca]

scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidths=0.5)

# Label some documents
for d in doc_pca:
    if abs(d['pc1']) > 2 or abs(d['pc2']) > 2:
        ax.annotate(d['doc_id'].replace('UntN ', ''), (d['pc1'], d['pc2']),
                   fontsize=7, color='white', alpha=0.8)

ax.axhline(y=0, color='#0f3460', linestyle='--', linewidth=1)
ax.axvline(x=0, color='#0f3460', linestyle='--', linewidth=1)

ax.set_xlabel('PC1 (Fragmentary ← → Complete formulaic)', fontsize=11)
ax.set_ylabel('PC2 (Standard titulary ← → Rare vocabulary)', fontsize=11)
ax.set_title('Document Stylometry: PCA Space\n(Size = word count, Color = PC1 sign)',
             fontsize=14, fontweight='bold', color='#e94560', pad=20)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ecdc4',
           markersize=10, label='Negative PC1 (Core formulaic)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e94560',
           markersize=10, label='Positive PC1 (Fragmentary)')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('document_pca_space.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: document_pca_space.png")

# =============================================================================
# Visualization 5: Full Combined Network
# =============================================================================
print("\n[6] Creating full combined network...")

fig, ax = plt.subplots(figsize=(18, 14))

# Build full graph
G = nx.Graph()
for e in edges:
    if e['total'] > 0.4:  # Threshold for visibility
        G.add_edge(e['source'], e['target'], weight=e['total'], type=e['type'])

print(f"    Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

# Draw edges by type
for edge_type, color, alpha in [('bigram', '#4ecdc4', 0.3),
                                 ('similarity', '#e94560', 0.4),
                                 ('both', '#ffe66d', 0.8)]:
    edge_list = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == edge_type]
    if edge_list:
        weights = [G[u][v]['weight'] * 2 for u, v in edge_list]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=weights,
                               edge_color=color, alpha=alpha, ax=ax)

# Node sizes by eigenvector
node_sizes = [50 + centrality.get(n, {}).get('eigenvector', 0) * 2000 for n in G.nodes()]

# Node colors by bridging
max_bridge = max(centrality.get(n, {}).get('bridging', 0) for n in G.nodes())
node_colors = [centrality.get(n, {}).get('bridging', 0) / max_bridge if max_bridge > 0 else 0
               for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       cmap=plt.cm.YlOrRd, alpha=0.8, ax=ax)

# Label high-centrality nodes
labels = {n: n for n in G.nodes() if centrality.get(n, {}).get('eigenvector', 0) > 0.12}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', ax=ax)

# Legend
legend_elements = [
    Line2D([0], [0], color='#4ecdc4', linewidth=3, label='Bi-gram (sequential)'),
    Line2D([0], [0], color='#e94560', linewidth=3, label='Similarity (distributional)'),
    Line2D([0], [0], color='#ffe66d', linewidth=3, label='Reinforced (both)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.set_title('Combined Bi-gram + Similarity Network\n(Node size = eigenvector centrality, Color = bridging centrality)',
             fontsize=14, fontweight='bold', color='#e94560', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('bigram_similarity_network.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("    Saved: bigram_similarity_network.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)
print("\nGenerated files:")
print("  - reinforced_constructions.png    (fixed construction pairs)")
print("  - centrality_comparison.png       (eigenvector vs bridging)")
print("  - edge_type_analysis.png          (bi-gram vs similarity)")
print("  - document_pca_space.png          (stylometry)")
print("  - bigram_similarity_network.png   (full combined network)")
print("=" * 60)
