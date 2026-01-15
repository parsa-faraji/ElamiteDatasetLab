#!/usr/bin/env python3
"""
Build Word2Vec model from Elamite texts and compute cosine similarity scores.
"""

from gensim.models import Word2Vec
import os
import csv
import numpy as np

print("=" * 50)
print("Elamite Word2Vec Pipeline")
print("=" * 50)

# 1. Load all text files
print("\n[1/5] Loading text files...")
texts_dir = 'texts'
txts = []

for file in sorted(os.listdir(texts_dir)):
    if file.endswith('.txt'):
        with open(os.path.join(texts_dir, file), 'r', encoding='utf-8') as f:
            txts.append(f.read())

print(f"    Loaded {len(txts)} documents")

# 2. Tokenize texts
print("\n[2/5] Tokenizing texts...")
def tokenize_elamite(texts):
    """Tokenize Elamite texts by whitespace, preserving special characters."""
    tokenized = []
    for text in texts:
        words = text.strip().split()
        if words:
            tokenized.append(words)
    return tokenized

words = tokenize_elamite(txts)
total_tokens = sum(len(doc) for doc in words)
print(f"    Tokenized {len(words)} documents")
print(f"    Total tokens: {total_tokens}")
print(f"    Sample tokens: {words[0][:5]}")

# 3. Build Word2Vec model
print("\n[3/5] Building Word2Vec model...")
w2v = Word2Vec(
    sentences=words,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)
print(f"    Vocabulary size: {len(w2v.wv)}")

# 4. Test the model
print("\n[4/5] Testing model with sample words...")
test_words = ['su-un-ki-ik', 'dingir-gal', 'a-ak', 'u2']
for test_word in test_words:
    if test_word in w2v.wv:
        similar = w2v.wv.most_similar(test_word, topn=3)
        similar_str = ', '.join([f"{w}({s:.3f})" for w, s in similar])
        print(f"    '{test_word}' -> {similar_str}")

# 5. Compute cosine similarity and update CSV
print("\n[5/5] Computing cosine similarity and updating CSV...")

def compute_avg_similarity(word, model, topn=5):
    """Compute average cosine similarity of a word to its top-n neighbors."""
    try:
        similar = model.wv.most_similar(word, topn=topn)
        avg_sim = np.mean([score for _, score in similar])
        return round(avg_sim, 4)
    except KeyError:
        return None

input_csv = 'UntN-Nasu texts Word-level.csv'
output_csv = 'UntN-Nasu texts Word-level with similarity.csv'

rows = []
with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['Cosine_similarity']

    for row in reader:
        word = row['Text']
        similarity = compute_avg_similarity(word, w2v)
        row['Cosine_similarity'] = similarity if similarity else ''
        rows.append(row)

with open(output_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"    Updated CSV saved to: {output_csv}")
print(f"    Total rows: {len(rows)}")

# Save the model
model_path = 'elamite_word2vec.model'
w2v.save(model_path)
print(f"    Model saved to: {model_path}")

# Summary stats
similarities = [row['Cosine_similarity'] for row in rows if row['Cosine_similarity']]
if similarities:
    sims = [float(s) for s in similarities]
    print(f"\n    Similarity stats:")
    print(f"      Min: {min(sims):.4f}")
    print(f"      Max: {max(sims):.4f}")
    print(f"      Mean: {np.mean(sims):.4f}")

print("\n" + "=" * 50)
print("Done!")
print("=" * 50)
