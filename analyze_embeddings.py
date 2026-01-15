#!/usr/bin/env python3
"""
Analyze Elamite Word2Vec embeddings and export insights.
"""

from gensim.models import Word2Vec
import numpy as np
import csv
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("Elamite Word Embedding Analysis")
print("=" * 60)

# Load the model
w2v = Word2Vec.load('elamite_word2vec.model')
vocab = list(w2v.wv.index_to_key)
print(f"\nVocabulary size: {len(vocab)} unique words")

# =============================================================================
# 1. Export word similarities
# =============================================================================
print("\n[1] Exporting word similarities...")

similarity_data = []
for word in vocab:
    similar = w2v.wv.most_similar(word, topn=5)
    row = {
        'word': word,
        'similar_1': similar[0][0],
        'score_1': round(similar[0][1], 4),
        'similar_2': similar[1][0],
        'score_2': round(similar[1][1], 4),
        'similar_3': similar[2][0],
        'score_3': round(similar[2][1], 4),
        'similar_4': similar[3][0],
        'score_4': round(similar[3][1], 4),
        'similar_5': similar[4][0],
        'score_5': round(similar[4][1], 4),
    }
    similarity_data.append(row)

# Save to CSV
with open('word_similarities.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['word', 'similar_1', 'score_1', 'similar_2', 'score_2',
                  'similar_3', 'score_3', 'similar_4', 'score_4', 'similar_5', 'score_5']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(similarity_data)
print("    Saved: word_similarities.csv")

# =============================================================================
# 2. Cluster analysis
# =============================================================================
print("\n[2] Performing cluster analysis...")

# Get all vectors
vectors = np.array([w2v.wv[word] for word in vocab])

# Try different cluster sizes
n_clusters = 15  # Reasonable for 649 words
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(vectors)

# Group words by cluster
cluster_groups = defaultdict(list)
for word, cluster_id in zip(vocab, clusters):
    cluster_groups[cluster_id].append(word)

# Save clusters to file
with open('word_clusters.txt', 'w', encoding='utf-8') as f:
    f.write("ELAMITE WORD CLUSTERS (K-Means, k=15)\n")
    f.write("=" * 60 + "\n\n")
    for cluster_id in sorted(cluster_groups.keys()):
        words = cluster_groups[cluster_id]
        f.write(f"CLUSTER {cluster_id} ({len(words)} words):\n")
        f.write("-" * 40 + "\n")
        # Sort by frequency in original corpus (approximated by position in vocab)
        for word in words:
            f.write(f"  {word}\n")
        f.write("\n")
print("    Saved: word_clusters.txt")

# =============================================================================
# 3. Identify patterns - words with specific markers
# =============================================================================
print("\n[3] Analyzing morphological patterns...")

patterns = {
    'divine_determinative_(d)': [w for w in vocab if w.startswith('(d)')],
    'personal_determinative_(m)': [w for w in vocab if w.startswith('(m)')],
    'divine_determinative_(md)': [w for w in vocab if w.startswith('(md)')],
    'feminine_(f)': [w for w in vocab if w.startswith('(f)')],
    'UPPERCASE_sumerograms': [w for w in vocab if w.isupper() and len(w) > 1],
    'ends_with_-ra': [w for w in vocab if w.endswith('-ra')],
    'ends_with_-ir': [w for w in vocab if w.endswith('-ir')],
    'ends_with_-ik': [w for w in vocab if w.endswith('-ik')],
    'ends_with_-ak': [w for w in vocab if w.endswith('-ak')],
    'ends_with_-na': [w for w in vocab if w.endswith('-na')],
    'ends_with_-ni': [w for w in vocab if w.endswith('-ni')],
    'ends_with_-me': [w for w in vocab if w.endswith('-me')],
    'contains_dingir': [w for w in vocab if 'dingir' in w.lower()],
}

# =============================================================================
# 4. Analyze within-pattern similarities
# =============================================================================
print("\n[4] Computing within-pattern similarities...")

pattern_analysis = []
for pattern_name, pattern_words in patterns.items():
    if len(pattern_words) >= 2:
        # Compute pairwise similarities within pattern
        pattern_vectors = [w2v.wv[w] for w in pattern_words]
        if len(pattern_vectors) >= 2:
            sim_matrix = cosine_similarity(pattern_vectors)
            # Get average similarity (excluding self-similarity)
            n = len(pattern_vectors)
            avg_sim = (sim_matrix.sum() - n) / (n * n - n) if n > 1 else 0
            pattern_analysis.append({
                'pattern': pattern_name,
                'count': len(pattern_words),
                'avg_internal_similarity': round(avg_sim, 4),
                'words': pattern_words
            })

# Sort by internal similarity
pattern_analysis.sort(key=lambda x: x['avg_internal_similarity'], reverse=True)

# =============================================================================
# 5. Generate insights report
# =============================================================================
print("\n[5] Generating insights report...")

with open('embedding_insights.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("ELAMITE WORD2VEC EMBEDDING ANALYSIS - INSIGHTS REPORT\n")
    f.write("=" * 70 + "\n\n")

    # Overview
    f.write("OVERVIEW\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total unique words: {len(vocab)}\n")
    f.write(f"Vector dimensions: 100\n")
    f.write(f"Training corpus: 85 documents, 2,582 tokens\n\n")

    # Pattern analysis
    f.write("MORPHOLOGICAL PATTERN ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write("Words grouped by morphological markers and their embedding coherence:\n\n")

    for pa in pattern_analysis:
        f.write(f"Pattern: {pa['pattern']}\n")
        f.write(f"  Count: {pa['count']} words\n")
        f.write(f"  Avg internal similarity: {pa['avg_internal_similarity']}\n")
        f.write(f"  Words: {', '.join(pa['words'][:10])}")
        if len(pa['words']) > 10:
            f.write(f" ... (+{len(pa['words'])-10} more)")
        f.write("\n\n")

    # High-similarity pairs
    f.write("\nHIGHEST SIMILARITY WORD PAIRS\n")
    f.write("-" * 70 + "\n")

    # Find top pairs
    top_pairs = []
    for i, word in enumerate(vocab):
        similar = w2v.wv.most_similar(word, topn=1)
        top_pairs.append((word, similar[0][0], similar[0][1]))
    top_pairs.sort(key=lambda x: x[2], reverse=True)

    f.write("These word pairs have the highest cosine similarity, suggesting\n")
    f.write("they appear in very similar contexts and may share grammatical roles:\n\n")

    seen = set()
    count = 0
    for w1, w2, score in top_pairs:
        pair_key = tuple(sorted([w1, w2]))
        if pair_key not in seen:
            seen.add(pair_key)
            f.write(f"  {w1:30} <-> {w2:30} : {score:.4f}\n")
            count += 1
            if count >= 25:
                break

    # Cluster insights
    f.write("\n\nCLUSTER ANALYSIS HIGHLIGHTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Words grouped into {n_clusters} clusters based on embedding similarity.\n")
    f.write("Clusters may represent:\n")
    f.write("  - Words with similar syntactic roles\n")
    f.write("  - Words appearing in similar constructions\n")
    f.write("  - Semantically related terms\n\n")

    # Show interesting clusters
    for cluster_id in sorted(cluster_groups.keys()):
        words = cluster_groups[cluster_id]
        if 3 <= len(words) <= 30:  # Medium-sized clusters are often most informative
            f.write(f"Cluster {cluster_id} ({len(words)} words):\n")
            f.write(f"  {', '.join(words[:15])}")
            if len(words) > 15:
                f.write(f" ...")
            f.write("\n\n")

    # Key observations
    f.write("\nKEY OBSERVATIONS\n")
    f.write("-" * 70 + "\n")

    observations = []

    # Check if determinatives cluster together
    divine_words = patterns['divine_determinative_(d)']
    if len(divine_words) >= 2:
        divine_vectors = [w2v.wv[w] for w in divine_words]
        sim_matrix = cosine_similarity(divine_vectors)
        n = len(divine_vectors)
        avg_sim = (sim_matrix.sum() - n) / (n * n - n) if n > 1 else 0
        observations.append(f"1. Divine determinative words (d)* show internal similarity of {avg_sim:.3f}")

    # Check suffix patterns
    suffix_ik = patterns['ends_with_-ik']
    suffix_ak = patterns['ends_with_-ak']
    if suffix_ik and suffix_ak:
        observations.append(f"2. Words ending in -ik ({len(suffix_ik)}) and -ak ({len(suffix_ak)}) may represent related morphological forms")

    # Check for construction patterns
    observations.append("3. High-similarity pairs often share syllable patterns, suggesting")
    observations.append("   the model captures phonological/morphological regularities")

    observations.append("4. Clusters group words that appear in similar sentence positions,")
    observations.append("   potentially revealing Elamite syntactic patterns")

    for obs in observations:
        f.write(f"{obs}\n")

    f.write("\n\nNEXT STEPS FOR ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write("1. Review word_clusters.txt to identify semantically coherent groups\n")
    f.write("2. Use word_similarities.csv to find potential lemma candidates\n")
    f.write("3. Compare clusters with known Elamite grammatical categories\n")
    f.write("4. Consider training on lemmatized text for refined embeddings\n")

print("    Saved: embedding_insights.txt")

# =============================================================================
# 6. Print summary to console
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY OF KEY FINDINGS")
print("=" * 60)

print("\nTop 10 highest-similarity word pairs:")
seen = set()
count = 0
for w1, w2, score in top_pairs:
    pair_key = tuple(sorted([w1, w2]))
    if pair_key not in seen:
        seen.add(pair_key)
        print(f"  {w1:25} <-> {w2:25} : {score:.4f}")
        count += 1
        if count >= 10:
            break

print("\nPattern coherence (higher = words in pattern are more similar):")
for pa in pattern_analysis[:5]:
    print(f"  {pa['pattern']:35} : {pa['avg_internal_similarity']:.4f} ({pa['count']} words)")

print("\n" + "=" * 60)
print("Files exported:")
print("  - word_similarities.csv    (each word + top 5 similar words)")
print("  - word_clusters.txt        (words grouped by embedding clusters)")
print("  - embedding_insights.txt   (detailed analysis report)")
print("=" * 60)
