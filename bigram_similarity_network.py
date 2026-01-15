#!/usr/bin/env python3
"""
Bi-gram + Similarity Network Analysis for Elamite Texts

Methodology:
1. Build bi-gram edges (sequential word pairs) with LOW weight
   - Captures syntagmatic relationships (what words appear together)
2. Add Word2Vec similarity edges with HIGH weight
   - Captures paradigmatic relationships (what words are interchangeable)
3. Compute advanced network statistics:
   - Bridging centrality: words connecting different clusters
   - Eigenvector centrality: words connected to other important words
4. Analyze per-document patterns vs corpus-wide patterns
5. Apply stylometry/PCA for document-level word attributes

The hypothesis: words in similar syntactic positions (like "ball" and "cat"
as objects, or "threw" and "gave" as verbs) will cluster together.
"""

import csv
import json
import numpy as np
from collections import defaultdict, Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("BI-GRAM + SIMILARITY NETWORK ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Load data
# =============================================================================
print("\n[1] Loading data...")

# Load Word2Vec model
w2v = Word2Vec.load('elamite_word2vec.model')
vocab = set(w2v.wv.index_to_key)
print(f"    Word2Vec vocabulary: {len(vocab)} words")

# Load original CSV to get word sequences per document
documents = defaultdict(list)
with open('UntN-Nasu texts Word-level.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_id = row['id_text']
        word = row['Text']
        documents[doc_id].append(word)

print(f"    Documents: {len(documents)}")
print(f"    Total tokens: {sum(len(words) for words in documents.values())}")

# =============================================================================
# 2. Build bi-gram edges (sequential neighbors)
# =============================================================================
print("\n[2] Building bi-gram edges (sequential word pairs)...")

BIGRAM_WEIGHT = 0.15  # Low weight for sequential relationships

bigram_edges = defaultdict(float)
bigram_counts = Counter()

for doc_id, words in documents.items():
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        # Create undirected edge
        edge = tuple(sorted([w1, w2]))
        bigram_edges[edge] += BIGRAM_WEIGHT
        bigram_counts[edge] += 1

print(f"    Unique bi-gram pairs: {len(bigram_edges)}")
print(f"    Total bi-gram occurrences: {sum(bigram_counts.values())}")

# Show most common bi-grams
print("\n    Top 15 most frequent bi-grams:")
for (w1, w2), count in bigram_counts.most_common(15):
    print(f"      {w1:25} -- {w2:25} : {count}")

# =============================================================================
# 3. Build similarity edges (Word2Vec cosine similarity)
# =============================================================================
print("\n[3] Building similarity edges (Word2Vec)...")

SIMILARITY_THRESHOLD = 0.40  # Only add edges above this threshold
SIMILARITY_WEIGHT_MULTIPLIER = 1.5  # Scale similarity for edge weight

similarity_edges = {}
all_words = list(vocab)

for i, w1 in enumerate(all_words):
    # Get top similar words
    try:
        similar = w2v.wv.most_similar(w1, topn=10)
        for w2, sim in similar:
            if sim >= SIMILARITY_THRESHOLD:
                edge = tuple(sorted([w1, w2]))
                if edge not in similarity_edges:
                    similarity_edges[edge] = sim * SIMILARITY_WEIGHT_MULTIPLIER
    except KeyError:
        continue

print(f"    Similarity edges (>= {SIMILARITY_THRESHOLD}): {len(similarity_edges)}")

# =============================================================================
# 4. Combine into unified network
# =============================================================================
print("\n[4] Combining bi-gram and similarity networks...")

# Combine edges: sum weights where both exist
combined_edges = defaultdict(lambda: {'bigram': 0, 'similarity': 0, 'total': 0})

for edge, weight in bigram_edges.items():
    combined_edges[edge]['bigram'] = weight
    combined_edges[edge]['total'] += weight

for edge, weight in similarity_edges.items():
    combined_edges[edge]['similarity'] = weight
    combined_edges[edge]['total'] += weight

# Get all unique nodes
all_nodes = set()
for (w1, w2) in combined_edges.keys():
    all_nodes.add(w1)
    all_nodes.add(w2)

print(f"    Combined network: {len(all_nodes)} nodes, {len(combined_edges)} edges")

# Categorize edges
only_bigram = sum(1 for e in combined_edges.values() if e['bigram'] > 0 and e['similarity'] == 0)
only_similarity = sum(1 for e in combined_edges.values() if e['bigram'] == 0 and e['similarity'] > 0)
both = sum(1 for e in combined_edges.values() if e['bigram'] > 0 and e['similarity'] > 0)

print(f"\n    Edge breakdown:")
print(f"      Bi-gram only:     {only_bigram}")
print(f"      Similarity only:  {only_similarity}")
print(f"      Both (reinforced): {both}")

# =============================================================================
# 5. Compute network statistics
# =============================================================================
print("\n[5] Computing network statistics...")

# Build adjacency for centrality calculations
adjacency = defaultdict(list)
for (w1, w2), weights in combined_edges.items():
    adjacency[w1].append((w2, weights['total']))
    adjacency[w2].append((w1, weights['total']))

# Degree centrality
degree = {node: len(neighbors) for node, neighbors in adjacency.items()}

# Weighted degree (strength)
strength = {node: sum(w for _, w in neighbors) for node, neighbors in adjacency.items()}

# Eigenvector centrality (power iteration method)
print("    Computing eigenvector centrality...")
n = len(all_nodes)
node_list = list(all_nodes)
node_idx = {node: i for i, node in enumerate(node_list)}

# Initialize with uniform values
eigenvector = {node: 1.0 / n for node in all_nodes}

# Power iteration
for iteration in range(50):
    new_eigenvector = {}
    for node in all_nodes:
        score = 0
        for neighbor, weight in adjacency[node]:
            score += eigenvector[neighbor] * weight
        new_eigenvector[node] = score

    # Normalize
    norm = np.sqrt(sum(v**2 for v in new_eigenvector.values()))
    if norm > 0:
        eigenvector = {k: v/norm for k, v in new_eigenvector.items()}

# Bridging centrality approximation
# (nodes that connect different parts of the network)
print("    Computing bridging centrality...")

# Simple bridging: nodes whose neighbors have low connectivity to each other
bridging = {}
for node in all_nodes:
    neighbors = [n for n, _ in adjacency[node]]
    if len(neighbors) < 2:
        bridging[node] = 0
        continue

    # Count edges between neighbors
    neighbor_set = set(neighbors)
    internal_edges = 0
    for neighbor in neighbors:
        for nn, _ in adjacency[neighbor]:
            if nn in neighbor_set and nn != neighbor:
                internal_edges += 1

    # Bridging = degree / neighbor density
    max_possible = len(neighbors) * (len(neighbors) - 1)
    if max_possible > 0:
        neighbor_density = internal_edges / max_possible
        bridging[node] = degree[node] * (1 - neighbor_density)
    else:
        bridging[node] = degree[node]

# Sort and display top nodes by each metric
print("\n    Top 15 by Eigenvector Centrality (connected to important words):")
for word, score in sorted(eigenvector.items(), key=lambda x: -x[1])[:15]:
    print(f"      {word:30} : {score:.6f}")

print("\n    Top 15 by Bridging Centrality (connects different clusters):")
for word, score in sorted(bridging.items(), key=lambda x: -x[1])[:15]:
    print(f"      {word:30} : {score:.4f}")

# =============================================================================
# 6. Per-document analysis
# =============================================================================
print("\n[6] Analyzing per-document patterns...")

# For each document, find words that appear and their local clustering
doc_patterns = {}

for doc_id, words in documents.items():
    unique_words = set(words)

    # Count bi-grams within this document
    doc_bigrams = Counter()
    for i in range(len(words) - 1):
        edge = tuple(sorted([words[i], words[i + 1]]))
        doc_bigrams[edge] += 1

    # Find words with both high local (bi-gram) and global (similarity) connections
    word_scores = {}
    for word in unique_words:
        if word not in adjacency:
            continue

        local_score = sum(1 for (w1, w2), c in doc_bigrams.items() if word in (w1, w2))
        global_score = eigenvector.get(word, 0)
        word_scores[word] = {
            'local': local_score,
            'global': global_score,
            'combined': local_score * 0.3 + global_score * 100
        }

    doc_patterns[doc_id] = {
        'word_count': len(words),
        'unique_words': len(unique_words),
        'bigram_types': len(doc_bigrams),
        'top_words': sorted(word_scores.items(), key=lambda x: -x[1]['combined'])[:5]
    }

# Show patterns for a few documents
print("\n    Sample document patterns:")
for doc_id in list(documents.keys())[:5]:
    pattern = doc_patterns[doc_id]
    print(f"\n    {doc_id}:")
    print(f"      Words: {pattern['word_count']}, Unique: {pattern['unique_words']}")
    print(f"      Key words: {', '.join(w for w, _ in pattern['top_words'])}")

# =============================================================================
# 7. Stylometry / PCA analysis
# =============================================================================
print("\n[7] Performing stylometry / PCA analysis...")

# Build document-term matrix for stylometry
# Each document as a row, each word as a column (frequency)
all_words_list = sorted(list(all_nodes))
word_to_col = {w: i for i, w in enumerate(all_words_list)}

doc_term_matrix = []
doc_ids = []

for doc_id, words in documents.items():
    row = [0] * len(all_words_list)
    word_counts = Counter(words)
    for word, count in word_counts.items():
        if word in word_to_col:
            row[word_to_col[word]] = count

    # Normalize by document length
    total = sum(row)
    if total > 0:
        row = [x / total for x in row]

    doc_term_matrix.append(row)
    doc_ids.append(doc_id)

doc_term_matrix = np.array(doc_term_matrix)

# PCA on documents
print("    Running PCA on document-term matrix...")
scaler = StandardScaler()
doc_term_scaled = scaler.fit_transform(doc_term_matrix)

pca = PCA(n_components=min(10, len(doc_ids) - 1))
doc_pca = pca.fit_transform(doc_term_scaled)

print(f"    Explained variance by component:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"      PC{i+1}: {var:.3f} ({var*100:.1f}%)")

# Find words that load heavily on each component
print("\n    Top words loading on each principal component:")
for pc_idx in range(min(3, pca.n_components_)):
    loadings = pca.components_[pc_idx]
    top_positive = sorted(range(len(loadings)), key=lambda i: -loadings[i])[:5]
    top_negative = sorted(range(len(loadings)), key=lambda i: loadings[i])[:5]

    print(f"\n    PC{pc_idx + 1}:")
    print(f"      Positive: {', '.join(all_words_list[i] for i in top_positive)}")
    print(f"      Negative: {', '.join(all_words_list[i] for i in top_negative)}")

# Add PCA-derived attributes to words
word_pc_scores = {}
for word in all_words_list:
    col_idx = word_to_col[word]
    word_pc_scores[word] = {
        f'pc{i+1}': float(pca.components_[i][col_idx])
        for i in range(min(3, pca.n_components_))
    }

# =============================================================================
# 8. Export results
# =============================================================================
print("\n[8] Exporting results...")

# Export combined edge list
with open('bigram_similarity_edges.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['source', 'target', 'bigram_weight', 'similarity_weight', 'total_weight', 'edge_type'])

    for (w1, w2), weights in sorted(combined_edges.items(), key=lambda x: -x[1]['total']):
        edge_type = 'both' if weights['bigram'] > 0 and weights['similarity'] > 0 else \
                    'bigram' if weights['bigram'] > 0 else 'similarity'
        writer.writerow([w1, w2,
                        round(weights['bigram'], 4),
                        round(weights['similarity'], 4),
                        round(weights['total'], 4),
                        edge_type])

print(f"    Saved: bigram_similarity_edges.csv")

# Export node attributes with centrality scores
with open('nodes_centrality.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'degree', 'strength', 'eigenvector', 'bridging', 'pc1', 'pc2', 'pc3'])

    for word in sorted(all_nodes):
        pc_scores = word_pc_scores.get(word, {'pc1': 0, 'pc2': 0, 'pc3': 0})
        writer.writerow([
            word,
            degree.get(word, 0),
            round(strength.get(word, 0), 4),
            round(eigenvector.get(word, 0), 6),
            round(bridging.get(word, 0), 4),
            round(pc_scores.get('pc1', 0), 4),
            round(pc_scores.get('pc2', 0), 4),
            round(pc_scores.get('pc3', 0), 4)
        ])

print(f"    Saved: nodes_centrality.csv")

# Export document PCA coordinates
with open('document_pca.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    headers = ['doc_id', 'word_count'] + [f'pc{i+1}' for i in range(min(5, pca.n_components_))]
    writer.writerow(headers)

    for i, doc_id in enumerate(doc_ids):
        row = [doc_id, len(documents[doc_id])]
        row.extend([round(doc_pca[i, j], 4) for j in range(min(5, pca.n_components_))])
        writer.writerow(row)

print(f"    Saved: document_pca.csv")

# Export JSON for visualization
graph_data = {
    'nodes': [
        {
            'id': word,
            'degree': degree.get(word, 0),
            'eigenvector': round(eigenvector.get(word, 0), 6),
            'bridging': round(bridging.get(word, 0), 4),
            'pc1': round(word_pc_scores.get(word, {}).get('pc1', 0), 4)
        }
        for word in all_nodes
    ],
    'edges': [
        {
            'source': w1,
            'target': w2,
            'bigram': round(weights['bigram'], 4),
            'similarity': round(weights['similarity'], 4),
            'total': round(weights['total'], 4),
            'type': 'both' if weights['bigram'] > 0 and weights['similarity'] > 0 else \
                    'bigram' if weights['bigram'] > 0 else 'similarity'
        }
        for (w1, w2), weights in combined_edges.items()
    ]
}

with open('bigram_similarity_graph.json', 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2, ensure_ascii=False)

print(f"    Saved: bigram_similarity_graph.json")

# =============================================================================
# 9. Generate insights report
# =============================================================================
print("\n[9] Generating insights report...")

with open('bigram_similarity_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("BI-GRAM + SIMILARITY NETWORK ANALYSIS REPORT\n")
    f.write("=" * 75 + "\n\n")

    f.write("METHODOLOGY\n")
    f.write("-" * 75 + "\n")
    f.write("""
This analysis combines two types of linguistic relationships:

1. SYNTAGMATIC (Bi-gram edges, low weight):
   - Words that appear next to each other in the text
   - Captures phrase structure and word order patterns
   - Weight: 0.15 per occurrence

2. PARADIGMATIC (Similarity edges, high weight):
   - Words with similar distributional profiles (Word2Vec)
   - Captures words that can substitute for each other
   - Weight: cosine similarity * 1.5 (threshold >= 0.40)

The hypothesis: Words in similar syntactic positions will cluster together.
For example, objects of verbs, subjects, or verbal complements should form
clusters even if they never appear adjacent in the corpus.
\n\n""")

    f.write("NETWORK STATISTICS\n")
    f.write("-" * 75 + "\n")
    f.write(f"Total nodes: {len(all_nodes)}\n")
    f.write(f"Total edges: {len(combined_edges)}\n")
    f.write(f"  - Bi-gram only: {only_bigram}\n")
    f.write(f"  - Similarity only: {only_similarity}\n")
    f.write(f"  - Reinforced (both): {both}\n\n")

    f.write("EIGENVECTOR CENTRALITY (Connected to Important Words)\n")
    f.write("-" * 75 + "\n")
    f.write("High eigenvector centrality indicates words connected to other\n")
    f.write("highly-connected words - likely core grammatical elements.\n\n")
    for word, score in sorted(eigenvector.items(), key=lambda x: -x[1])[:20]:
        f.write(f"  {word:35} : {score:.6f}\n")

    f.write("\n\nBRIDGING CENTRALITY (Connects Different Clusters)\n")
    f.write("-" * 75 + "\n")
    f.write("High bridging centrality indicates words that connect otherwise\n")
    f.write("separate parts of the network - possibly polysemous words or\n")
    f.write("words that appear in multiple construction types.\n\n")
    for word, score in sorted(bridging.items(), key=lambda x: -x[1])[:20]:
        f.write(f"  {word:35} : {score:.4f}\n")

    f.write("\n\nREINFORCED EDGES (Both Bi-gram and Similarity)\n")
    f.write("-" * 75 + "\n")
    f.write("These word pairs appear adjacent AND have high distributional\n")
    f.write("similarity - strong candidates for fixed constructions.\n\n")

    reinforced = [(e, w) for e, w in combined_edges.items()
                  if w['bigram'] > 0 and w['similarity'] > 0]
    reinforced.sort(key=lambda x: -x[1]['total'])

    for (w1, w2), weights in reinforced[:25]:
        f.write(f"  {w1:25} -- {w2:25} (bigram: {weights['bigram']:.2f}, sim: {weights['similarity']:.2f})\n")

    f.write("\n\nMOST FREQUENT BI-GRAMS\n")
    f.write("-" * 75 + "\n")
    f.write("Sequential word pairs - potential phrase boundaries occur where\n")
    f.write("bi-gram frequency drops significantly.\n\n")
    for (w1, w2), count in bigram_counts.most_common(25):
        f.write(f"  {w1:25} -- {w2:25} : {count}\n")

    f.write("\n\nPCA ANALYSIS - DOCUMENT VARIATION\n")
    f.write("-" * 75 + "\n")
    f.write("Principal components capture systematic variation in word usage\n")
    f.write("across documents.\n\n")

    for i in range(min(3, pca.n_components_)):
        f.write(f"PC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.1f}% variance):\n")
        loadings = pca.components_[i]
        top_pos = sorted(range(len(loadings)), key=lambda j: -loadings[j])[:8]
        top_neg = sorted(range(len(loadings)), key=lambda j: loadings[j])[:8]
        f.write(f"  Positive: {', '.join(all_words_list[j] for j in top_pos)}\n")
        f.write(f"  Negative: {', '.join(all_words_list[j] for j in top_neg)}\n\n")

    f.write("\n\nLINGUISTIC HYPOTHESES\n")
    f.write("=" * 75 + "\n\n")

    f.write("""1. PHRASE BOUNDARY DETECTION
   - Look for positions where bi-gram frequency drops
   - High-bridging words may mark clause boundaries
   - Words with high eigenvector + low bridging are phrase-internal

2. CONSTRUCTION IDENTIFICATION
   - Reinforced edges (both bi-gram and similarity) indicate fixed phrases
   - Similar eigenvector scores suggest similar syntactic roles
   - Clusters of high-similarity words = paradigmatic classes (noun, verb, etc.)

3. DOCUMENT VARIATION
   - PC1 may capture genre or text type differences
   - Words loading on same PC appear in similar document contexts
   - Can identify text-specific vs. corpus-wide patterns

4. LATENT SYNTACTIC CLASSES
   - Words with similar bridging AND eigenvector profiles likely share
     syntactic category (e.g., all verbs, all nouns)
   - The "ball/cat" and "threw/gave" pattern should emerge in clusters
""")

print(f"    Saved: bigram_similarity_report.txt")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - bigram_similarity_edges.csv    (combined edge list)")
print("  - nodes_centrality.csv           (centrality metrics per word)")
print("  - document_pca.csv               (document coordinates in PCA space)")
print("  - bigram_similarity_graph.json   (full graph for visualization)")
print("  - bigram_similarity_report.txt   (detailed analysis report)")
print("=" * 70)
