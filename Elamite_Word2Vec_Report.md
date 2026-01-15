# Elamite Word2Vec Analysis Report

**Project:** TokenWorks - Elamite Dataset Lab
**Date:** January 14, 2025
**Prepared by:** Parsa Faraji
**Supervisor:** Adam Anderson

---

## 1. Executive Summary

This report presents the results of applying Word2Vec word embeddings to a corpus of Elamite texts (UntN-Nasu collection). The goal was to explore whether vector representations of Elamite words can reveal patterns in how these strings form constructions, as a preliminary step before lemmatization.

**Key Findings:**
- The model successfully captured contextual relationships between Elamite words
- High-similarity word pairs suggest shared syntactic/grammatical roles
- Morphological patterns (suffixes like -ak, -ik, -me) show measurable coherence in the embedding space
- Word clusters reveal potential groupings for linguistic analysis

---

## 2. Methodology

### 2.1 Data Preparation
- **Source:** UntN-Nasu texts (Google Sheets export)
- **Format conversion:** Document-level to word-level (one word per row)
- **Text files:** 85 individual .txt files created for each document

### 2.2 Word2Vec Configuration
| Parameter | Value |
|-----------|-------|
| Vector dimensions | 100 |
| Context window | 5 |
| Minimum word count | 1 |
| Algorithm | Skip-gram (default) |

### 2.3 Corpus Statistics
| Metric | Value |
|--------|-------|
| Total documents | 85 |
| Total tokens | 2,582 |
| Unique words (vocabulary) | 649 |

---

## 3. Results

### 3.1 Highest Similarity Word Pairs

These word pairs have the highest cosine similarity, suggesting they appear in very similar contexts and may share grammatical roles:

| Word 1 | Word 2 | Cosine Similarity |
|--------|--------|-------------------|
| a-ak | tu4-ur | 0.638 |
| u2-me | in | 0.622 |
| u2-pa-at | u2-me | 0.618 |
| (d)in-šu-uš-na-ak | a-ak | 0.595 |
| ku-ši-ih | u2-me | 0.588 |
| hu-us-si-ip-me | in | 0.571 |
| si-it-me | in | 0.548 |
| si-ia-an | in | 0.546 |
| si-ia-an-ku-uk-ra | a-gi | 0.545 |
| dingir-gal | u2-me | 0.542 |

**Interpretation:** Words like `a-ak`, `u2-me`, `in`, and `tu4-ur` appear to function as connective or grammatical elements, frequently co-occurring in similar sentence positions.

### 3.2 Morphological Pattern Analysis

Words were grouped by morphological markers to measure their embedding coherence (average internal cosine similarity):

| Pattern | Count | Avg. Internal Similarity |
|---------|-------|-------------------------|
| Feminine (f)* | 2 | 0.053 |
| Suffix -ik | 16 | 0.045 |
| Suffix -ak | 23 | 0.039 |
| Personal determinative (m)* | 7 | 0.036 |
| Suffix -ir | 10 | 0.033 |
| Contains "dingir" | 11 | 0.029 |
| Suffix -me | 53 | 0.020 |
| Suffix -na | 15 | 0.018 |
| Suffix -ra | 33 | 0.017 |
| Divine determinative (d)* | 86 | 0.013 |

**Interpretation:**
- The `-ik` and `-ak` suffixes show the highest coherence, suggesting these morphological endings mark words with similar grammatical functions
- Divine determinative words `(d)*` have low internal similarity, indicating semantic diversity among divine names despite shared markup
- The `-me` suffix is highly frequent (53 words) but appears in varied contexts

### 3.3 Cluster Analysis

K-means clustering (k=15) grouped words by embedding similarity. Notable clusters include:

**Cluster 11 - Core Grammatical Vocabulary (14 words):**
```
u2-me, ku-ši-ih, a-ak, šu-šu-un-ka, tu4-ur, dingir-gal,
(d)in-šu-uš-na-ak, in, u2-pa-at, hu-us-si-ip-me,
si-ia-an-ku-uk-ra, […], si-it-me, a-ha-ar
```
These appear to be high-frequency words with core syntactic roles.

**Cluster 3 - Verbal/Action Forms (28 words):**
```
la-ha-ak-ir-ra, te-ip-ti, a-ha, la-an-si-ti-ir-ra, ša-ra-ra,
um-me, (d)ša-la, ta-ah, su-un-ki-ir, hu-ut-ta-ak...
```

**Cluster 6 - Mixed Divine/Numerical (17 words):**
```
(d)30, zu-un-ki-ip, si-ma-aš2, DINGIR-GAL, 10,
(d)ki-ri-ri-ša-me, (d)ha-te-ri-iš-nu-me...
```

### 3.4 Spelling Variants Detection

The embeddings help identify potential spelling variants of the same word:

| Variant 1 | Variant 2 | Notes |
|-----------|-----------|-------|
| (f)na-pír-a-su | (f)na-pir2-a-su | Same name, different transliteration |
| su-un-ki-ik | zu-un-ki-ik | Possible s/z alternation |
| (d)in-šu-uš-na-ak | (d)in-šu-ši-na-ak | Variant spellings |
| hu-ut-tak-ha-li-ik | hu-ut-ta-ak-ha-li-ik | Syllable boundary variation |

---

## 4. Key Observations

1. **Contextual Relationships Captured:** The Word2Vec model successfully learned contextual relationships despite the small corpus size. Words appearing in similar syntactic positions cluster together.

2. **Morphological Patterns:** Suffix-based groupings (-ak, -ik, -ir) show measurable coherence, suggesting these represent productive morphological categories in Elamite.

3. **Core Vocabulary Identification:** Cluster 11 contains what appear to be high-frequency grammatical words that form the structural backbone of Elamite sentences.

4. **Lemmatization Candidates:** The similarity scores can help identify spelling variants that may represent the same lemma, useful for the upcoming lemmatization work.

5. **Limitations:**
   - Small corpus (2,582 tokens) limits the model's ability to learn rare word patterns
   - Transliteration conventions introduce artificial variation
   - No lemmatization means inflected forms are treated as separate words

---

## 5. Files Delivered

| File | Description |
|------|-------------|
| `word_similarities.csv` | Each word with its top 5 most similar words and scores |
| `word_clusters.txt` | All 649 words grouped into 15 clusters |
| `embedding_insights.txt` | Detailed analysis report |
| `elamite_word2vec.model` | Trained model (reusable for future queries) |
| `texts/` | 85 individual document text files |
| `Elamite_Word2Vec.ipynb` | Interactive notebook for exploration |

---

## 6. Next Steps

1. **Review Clusters:** Examine `word_clusters.txt` with Elamite linguistic knowledge to validate/refine groupings

2. **Lemmatization Integration:** Use `word_similarities.csv` to identify potential lemma candidates before manual lemmatization

3. **Expanded Corpus:** Training on a larger corpus would improve embedding quality

4. **Post-Lemmatization Retraining:** After lemmatization, retrain the model for cleaner semantic groupings

5. **Comparative Analysis:** Compare clusters with known Elamite grammatical categories from existing scholarship

---

## 7. Technical Notes

**Environment:**
- Python 3.9
- Gensim 4.4.0
- scikit-learn (for clustering)

**Reproducibility:**
All scripts are included in the project folder. To regenerate results:
```bash
cd ElamiteDatasetLab
python3 generate_txt_files.py  # Create text files
python3 run_word2vec.py        # Build model
python3 analyze_embeddings.py  # Generate analysis
```

---

*Report generated as part of the TokenWorks Elamite Dataset Lab project.*
