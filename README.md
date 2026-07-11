# NLP and Sentiment Analysis
### DecodeLabs Industrial Training Kit | Batch 2026

**Author:** Anubhav Singh  
**Track:** Optional Mastery Phase — Natural Language Processing & Unstructured Data Handling

---

## Overview

This project builds an end-to-end NLP pipeline that programs a machine to read and mathematically categorize unstructured human text (movie reviews) as **Positive** or **Negative** sentiment.

80% of all enterprise data exists in unstructured, qualitative formats — emails, reviews, support tickets. This project crosses the engineering threshold by converting that qualitative language into a quantitative spatial representation to predict emotional polarity.

**Dataset:** NLTK Movie Reviews Corpus — 2,000 labeled IMDB reviews (1,000 Positive / 1,000 Negative)

---

## Pipeline Architecture

```
Raw Text  →  Pre-Process  →  Vectorize (TF-IDF)  →  Model (Naive Bayes)  →  Prediction
  ↓               ↓                  ↓                       ↓
String    →  Cleaned Tokens  →  Sparse Matrix  →  Probability Distribution
```

### Stage 1 — Text Pre-Processing
- HTML tag removal and character normalization
- Lowercasing
- Word tokenization
- **Custom stopword removal with negations preserved** — default NLTK removes "not", turning "I am not happy" → predicts POSITIVE (fatal flaw). Fix: explicitly remove negations from the stopword set using set-based operations.
- **POS-guided WordNet Lemmatization** — Penn Treebank POS tags mapped to WordNet tags before lemmatizing. Without POS context, "went" stays "went" instead of reducing to "go".

### Stage 2 — TF-IDF Vectorization
- `ngram_range=(1,2)` — unigrams + bigrams to capture negated phrases like "not good"
- `max_features=10,000` — caps vocabulary to prevent dimensionality explosion
- `min_df=2` — drops rare typos appearing in only one review
- `sublinear_tf=True` — log-scaling prevents high-frequency words from dominating
- Output stored as **SciPy CSR sparse matrix** — saves ~99% memory vs dense arrays

### Stage 3 — Classification
- **Multinomial Naive Bayes** with Laplace smoothing (`alpha=1.0`)
- MultinomialNB chosen over ComplementNB because the dataset is perfectly balanced (50/50)
- Laplace smoothing prevents zero-frequency probability collapse on unseen words

---

## Results

| Metric    | Score     |
|-----------|-----------|
| Accuracy  | ~83-85%   |
| Precision | ~83-85%   |
| Recall    | ~83-85%   |
| F1-Score  | ~83-85%   |

*Run the notebook to see exact scores on your machine.*

---

## Project Structure

```
NLP_and_Sentiment-Analysis/
├── NLP_Sentiment_Analysis.ipynb   # Main notebook
├── requirements.txt                                       # Dependencies
└── README.md                                              # This file
```

---

## Setup & Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/Anubhavsingh311/NLP_and_Sentiment-Analysis.git
cd NLP_and_Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook NLP_Sentiment_Analysis.ipynb
```

The notebook will automatically download all required NLTK corpora on first run (movie_reviews, stopwords, wordnet, averaged_perceptron_tagger).

---

## Dependencies

See `requirements.txt`. Core libraries:

| Library | Purpose |
|---------|---------|
| `nltk` | Corpus, tokenization, POS tagging, lemmatization |
| `scikit-learn` | TF-IDF vectorizer, Naive Bayes, evaluation metrics |
| `scipy` | CSR sparse matrix storage |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `wordcloud` | Word cloud generation |

---

## Key Engineering Rules Applied (DecodeLabs Blueprint)

1. **Negation preservation** — explicitly remove negations from NLTK stopword list via `set` subtraction
2. **POS-guided lemmatization** — map Treebank tags to WordNet tags before calling `WordNetLemmatizer`
3. **Bigrams in TF-IDF** — `ngram_range=(1,2)` captures negated sentiment pairs
4. **CSR sparse storage** — prevents memory crashes on high-dimensional text matrices
5. **Laplace smoothing** — `alpha=1.0` prevents zero-probability collapse on unseen vocabulary
6. **Train/test split before vectorization** — fit vectorizer on training data only to prevent data leakage

---

## Other Projects in This Series

| Project | Title | Link |
|---------|-------|------|
| Project 1 | Data Cleaning & EDA | [GitHub](https://github.com/Anubhavsingh311/EDA_and_Feature-Engineering) |
| Project 2 | Fraud Detection Pipeline | [GitHub](https://github.com/Anubhavsingh311/Fraud-Detection-Pipeline) |
| Project 3 | Customer Segmentation — Unsupervised ML | [GitHub](https://github.com/Anubhavsingh311/Customer-Segmentation-Unsupervised-ML) |
| Project 4 | NLP & Sentiment Analysis | [GitHub](https://github.com/Anubhavsingh311/NLP_and_Sentiment-Analysis) |

---

## Known Limitations

- NLTK Movie Reviews corpus is domain-specific (film criticism). Accuracy will vary on other text domains (product reviews, tweets).
- Bag-of-words approach via TF-IDF loses word order. "The movie was not good" and "The movie was good" differ only in "not" — preserved here via negation fix but order-sensitive semantics are still lost.
- Naive Bayes assumes feature independence (words are independent). This is linguistically naive but computationally efficient.

## Future Improvements

- Replace TF-IDF with transformer embeddings (BERT, DistilBERT) for context-aware representations
- Extend to multi-class sentiment (1-5 star ratings)
- Build a REST API wrapper for live inference
- Add LIME/SHAP explainability to highlight which words drove each prediction
