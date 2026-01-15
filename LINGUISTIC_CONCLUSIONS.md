# Deep Linguistic Conclusions from Network Analysis

## Executive Summary

By combining **syntagmatic** (bi-gram/sequential) and **paradigmatic** (distributional similarity) relationships, we've identified:

1. **32 reinforced word pairs** that are both adjacent AND distributionally similar — strong candidates for fixed constructions
2. **Key grammatical function words** with high centrality scores
3. **Potential phrase/clause boundary markers** via bridging centrality
4. **Formulaic patterns** in royal/dedicatory texts

---

## Key Finding 1: Fixed Constructions Identified

### Reinforced Edges (Both Sequential AND Similar)

These 32 word pairs appear next to each other in the texts AND have high Word2Vec similarity — the strongest evidence for fixed grammatical constructions:

| Word Pair | Bi-gram Freq | Similarity | Interpretation |
|-----------|:------------:|:----------:|----------------|
| `(d)in-šu-uš-na-ak — a-ak` | 13x | 0.89 | Divine name + connective |
| `dingir-gal — u2-me` | 15x | 0.81 | "Great god" + verbal marker |
| `ku-ši-ih — (d)in-šu-uš-na-ak` | 4x | 0.81 | ? + divine name |
| `si-it-me — u2-me` | 18x | 0.77 | Verbal forms clustering |
| `tu4-ur — u2-me` | 17x | 0.74 | Verbal construction |
| `si-ia-an — u2-pa-at` | 6x | 0.74 | Place reference + ? |

**Linguistic Hypothesis**: The `-me` suffix words (`u2-me`, `si-it-me`, `ta-ak-me`) cluster tightly together, suggesting `-me` marks a specific grammatical category (possibly verbal/participial).

---

## Key Finding 2: Grammatical Function Words

### Words by Eigenvector Centrality
(Connected to other important/central words)

| Word | Eigenvector | Role Hypothesis |
|------|:-----------:|-----------------|
| `u2-me` | 0.354 | Core verbal auxiliary or copula |
| `ku-ši-ih` | 0.278 | High-frequency grammatical particle |
| `šu-šu-un-ka` | 0.263 | "of Susa" — locative/genitive |
| `an-za-an` | 0.242 | "Anshan" — place name in formulae |
| `dingir-gal` | 0.239 | "Great god" — divine epithet |
| `a-ak` | 0.213 | Connective/conjunction |
| `in` | 0.210 | Particle or conjunction |

**Observation**: The top eigenvector words form the grammatical backbone of the corpus. They appear across many different constructions and connect diverse vocabulary.

---

## Key Finding 3: Phrase Boundary Candidates

### Words by Bridging Centrality
(Connects otherwise separate parts of the network)

| Word | Bridging | Interpretation |
|------|:--------:|----------------|
| `ku-ši-ih` | 93.9 | Connects multiple clause types |
| `a-ak` | 83.0 | **Connective** — joins clauses |
| `u2-me` | 80.4 | Verbal element bridging different constructions |
| `si-ia-an` | 49.2 | Place reference across contexts |
| `šu-šu-un-ka` | 41.5 | "of Susa" — in multiple formulae |

**Linguistic Hypothesis**: High-bridging words like `a-ak` and `ku-ši-ih` may mark **clause or phrase boundaries**. They connect different semantic domains, similar to how conjunctions work in other languages.

### The `a-ak` Pattern
The word `a-ak` appears with:
- Divine names: `(d)in-šu-uš-na-ak — a-ak`
- Place names: after `an-za-an`, `šu-šu-un-ka`
- Other grammatical words: `dingir-gal — a-ak`

This strongly suggests `a-ak` functions as a **connective or conjunction**, possibly meaning "and" or marking genitive/possessive relationships.

---

## Key Finding 4: Formulaic Royal Patterns

### Most Frequent Bi-grams (Sequential Pairs)

| Bi-gram | Frequency | Interpretation |
|---------|:---------:|----------------|
| `(m)un-taš-dingir-gal — ša-ak` | 70 | Royal name formula |
| `an-za-an — šu-šu-un-ka` | 66 | "of Anshan (and) Susa" |
| `(m)un-taš-dingir-gal — u2` | 64 | Royal name + ? |
| `an-za-an — su-un-ki-ik` | 49 | "king of Anshan" |
| `hu-ut-tak-ha-li-ik — u2-me` | 37 | Verbal construction |

**The Royal Title Formula**:
```
(m)un-taš-dingir-gal  ša-ak  an-za-an  šu-šu-un-ka  su-un-ki-ik
[Personal name]       [?]    [Anshan]  [of Susa]    [king]
```

This appears to be the standard royal titulary: "Untash-Napirisha, [X] of Anshan and Susa, king..."

---

## Key Finding 5: Morphological Paradigm Classes

### Words Clustering by Suffix

The network reveals that words with the **same suffix** cluster together distributionally:

| Suffix | Example Words | Avg. Internal Similarity | Hypothesis |
|--------|---------------|:------------------------:|------------|
| `-me` | u2-me, si-it-me, ta-ak-me, hu-us-si-ip-me | High | Verbal/participial marker |
| `-ak` | a-ak, ša-ak, te-la-ak | Medium | Connective/genitive |
| `-ik` | su-un-ki-ik, hu-ut-tak-ha-li-ik | Medium | Nominal (king, etc.) |
| `-ra` | si-ia-an-ku-uk-ra, me-lu-uk-ra | Medium | Locative/directional? |
| `-ka` | šu-šu-un-ka, hi-en-ka | Medium | Locative/possessive? |

**The `-me` cluster** is particularly coherent, suggesting it marks a consistent grammatical category across different verb roots.

---

## Key Finding 6: Document-Level Variation (PCA)

### Principal Component Analysis

**PC1 (5.8% variance)** separates:
- **Negative loading**: Core formulaic vocabulary (`ku-ši-ih`, `ša-ak`, `an-za-an`, `šu-šu-un-ka`)
- **Positive loading**: Fragmentary/variant readings (`(d)in-šu-[uš]-na-ak`, `[az-ki-it]`)

**Interpretation**: PC1 captures text preservation quality. Documents with negative PC1 scores have well-preserved formulaic content; positive scores indicate more fragmentary texts.

**PC2 (4.5% variance)** separates:
- **Negative loading**: Standard royal titulary (`šu-šu-un-ka`, `ša-ak`, `an-za-an`, `tu4-ur`)
- **Positive loading**: Less common vocabulary (`hu-hu-un`, `a-al`, `ha-al-ma-šu-um`)

**Interpretation**: PC2 may capture text type or genre differences.

---

## Synthesis: Toward Phrase Structure

### Proposed Elamite Clause Template

Based on the bi-gram frequencies and reinforced edges:

```
[Royal Name]  [Connective]  [Place1]  [Connective]  [Place2]  [Title]  [Verbal]
     |             |           |            |          |         |        |
(m)un-taš-    ša-ak      an-za-an      ?       šu-šu-un-ka  su-un-ki-ik  u2-me
dingir-gal                                                               |
                                                                    [Verb complex]
```

### Candidate Clause Boundaries

Words with **high bridging centrality** likely mark boundaries:
1. `a-ak` — conjunction/connective between clauses
2. `ku-ši-ih` — possibly marks predicate boundaries
3. `u2-me` — possibly marks end of verbal complexes

---

## Recommendations for Next Steps

### 1. Close Reading Validation
Select 5-10 texts and manually annotate phrase boundaries. Compare with:
- Positions of high-bridging words (`a-ak`, `ku-ši-ih`)
- Bi-gram frequency drops (where common pairs break)

### 2. Linked Data Representation
Export as RDF with:
- `elamite:followedBy` for bi-gram relationships
- `elamite:similarTo` for distributional similarity
- `elamite:hasSuffix` for morphological attributes
- `elamite:bridgingScore` for clause boundary likelihood

### 3. Expand Stylometry
Apply the notebooks from the shared drive to:
- Compare UntN-Nasu with other Elamite corpora
- Identify genre-specific vocabulary
- Build document similarity networks

### 4. Iterative Lemmatization
Use the reinforced edges to propose lemma groups:
- High-similarity pairs with same suffix = likely same lemma
- Spelling variants (with `[` `]` fragments) can be grouped

---

## Files Generated

| File | Contents |
|------|----------|
| `bigram_similarity_edges.csv` | Combined edge list with type labels |
| `nodes_centrality.csv` | Eigenvector, bridging, PCA scores per word |
| `document_pca.csv` | Document coordinates in PCA space |
| `bigram_similarity_graph.json` | Full graph for visualization |
| `visualize_bigram_network.html` | Interactive exploration tool |
| `bigram_similarity_report.txt` | Detailed analysis report |

---

*Analysis by Parsa Faraji & Claude | January 2026*
