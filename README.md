# Elamite Dataset Lab

Word2Vec analysis of Elamite texts from the UntN-Nasu collection.

## Overview

This project applies Word2Vec word embeddings to a corpus of Elamite texts to explore how words form constructions and identify patterns that may assist with lemmatization.

**Corpus Statistics:**
- 85 documents
- 2,582 tokens
- 649 unique words

## Key Findings

- High-similarity word pairs (e.g., `a-ak` ↔ `tu4-ur`, `u2-me` ↔ `in`) suggest shared syntactic roles
- Morphological suffixes (-ak, -ik, -me) show measurable coherence in the embedding space
- Word clusters reveal potential groupings for linguistic analysis

See `Elamite_Word2Vec_Report.md` for the full analysis.

## Project Structure

```
├── texts/                          # 85 document text files
├── word_similarities.csv           # Each word + top 5 similar words
├── word_clusters.txt               # Words grouped into 15 clusters
├── embedding_insights.txt          # Detailed analysis
├── elamite_word2vec.model          # Trained Word2Vec model
├── Elamite_Word2Vec_Report.md      # Full report
├── Elamite_Word2Vec.ipynb          # Interactive notebook
├── generate_txt_files.py           # Script: CSV → text files
├── run_word2vec.py                 # Script: Build model
└── analyze_embeddings.py           # Script: Generate analysis
```

## Usage

```bash
# Generate text files from CSV
python3 generate_txt_files.py

# Build Word2Vec model
python3 run_word2vec.py

# Run clustering and analysis
python3 analyze_embeddings.py
```

**Requirements:** Python 3.9+, gensim, scikit-learn

## Team

- Parsa Faraji
- Adam Anderson

## License

For academic use.
