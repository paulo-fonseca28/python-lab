# TF-IDF Information Retrieval Engine in Python

## Project Overview

This project implements a modular information retrieval (IR) engine in Python built on top of TF-IDF weighting, boolean retrieval, cosine similarity, and phrase queries. It loads a text collection from a JSON file, builds an inverted index with positional information, and offers a simple CLI for experimentation with IR concepts on a small document set.

## Features

- Load a document collection from JSON (`collection_tfidf_ir_engine.json`) with `name` (ID) and `content` (text) fields. 
- Incremental or bulk insertion and removal of documents from the active collection.  
- Text normalization with lowercasing, accent removal, and regex-based tokenization.  
- Vocabulary construction and positional inverted index: term → {doc_id → [positions]}.  
- TF-IDF computation using log-based term frequency and inverse document frequency.  
- Boolean retrieval with `AND`, `OR` and `NOT` over normalized query terms.  
- Cosine similarity ranking using TF-IDF vectors for free-text queries.  
- Phrase querying using positional information to detect exact term sequences.

## Project Structure

- `tfidf_ir_engine.py` – Core implementation of the IR system (class, indexing, TF-IDF, queries, CLI menu).  
- `collection_tfidf_ir_engine.json` – Example document collection with `name` and `content` fields used as the corpus.

## Collection and Text Processing

The system expects the collection file `collection_tfidf_ir_engine.json` to be a JSON list where each element contains at least:

```json
{
"name": "D1",
"content": "Texto completo do documento..."
}
```


When the engine is initialized:

- The JSON is loaded, and raw documents are stored in an internal list.
- Documents can be added one by one or all at once to the active collection.
- Each added document is normalized:
  - Convert to lowercase.
  - Remove accents via Unicode normalization.
  - Replace non-alphanumeric characters with spaces using regular expressions.
  - Split into whitespace-separated tokens.

These tokens are used to build both the vocabulary and the inverted index.

## Indexing and TF-IDF Computation

### Vocabulary and inverted index

- The vocabulary is the sorted set of all distinct tokens that appear in the active collection.

- The positional inverted index has the form:

    term -> { doc_id: [pos1, pos2, ...] }


- Positions are zero-based indices in the token sequence of each document, enabling phrase queries later.

### TF, IDF, and TF-IDF

For each document:

- Term frequency (TF) uses log weighting:
  - If `freq > 0`: `tf = 1 + log2(freq)`
  - If `freq = 0`: `tf = 0.0`  
- Document frequency `df(term)` is the number of documents where the term has `tf > 0`.
- Inverse document frequency (IDF) is:

```python
idf(term) = log2(N / df(term)) if df(term) > 0
idf(term) = 0.0 otherwise
```

- TF-IDF for each document is then:

```python
tfidf_doc[term] = tf_doc[term] * idf[term]
```


The engine can display the non-zero TF-IDF weights per document, summarizing which terms are most relevant in each text.

## Query Capabilities

### Boolean queries

The engine supports simple boolean queries over normalized terms with the operators `AND`, `OR`, and `NOT`:

- Evaluation is left-to-right (no precedence or parentheses).
- Terms are normalized with the same pipeline used for documents.
- For each term, the engine retrieves the set of document IDs from the inverted index.
- `NOT` uses the set of all document IDs to compute complements.

Examples (as strings):

- `brasil AND liberdade`  
- `estrutura AND NOT linear`  
- `hash OR arvore`

The result is the set of document IDs that satisfy the boolean expression.

### Cosine similarity (TF-IDF ranking)

For a free-text query:

- The query text is normalized and tokenized like documents.
- A query TF-IDF vector is built using the same vocabulary and IDF values.
- Cosine similarity is computed between the query vector and each document TF-IDF vector:

```python
cosine(q, d) = dot(q, d) / (||q|| * ||d||)
```


- Documents are ranked by similarity (descending), and the top-k results are displayed.

### Phrase queries (positional)

For an exact phrase query:

- The phrase is normalized and tokenized.
- Candidate documents are those that contain all terms in the phrase.
- For each candidate document, the engine checks positional sequences:
  - It looks for positions `p` where the first term appears and tests whether
    `p + 1`, `p + 2`, ..., `p + (n - 1)` also appear for the subsequent terms.  
- Documents and the number of exact occurrences of the phrase are reported, sorted by frequency.

## Example Usage

Typical workflow from the CLI:

1. Start the engine pointing to the collection file:

```bash
python tfidf_ir_engine.py
```


2. Use the menu options to:
- Add a single document from the JSON collection.
- Add all remaining documents at once.
- Remove a document by ID (e.g., `D1`).
- Display the vocabulary and its size.
- Display the non-zero TF-IDF weights per document.
- Display the full positional inverted index.
- Run boolean queries with `AND`, `OR`, `NOT`.
- Run similarity (cosine) ranking for a text query.
- Run phrase queries over the indexed collection.

## Learning Outcomes

- Practical understanding of building an inverted index with positions in Python.  
- Hands-on experience with TF-IDF weighting (TF log scaling, IDF, and full document vectors).  
- Implementation of boolean, cosine similarity, and phrase retrieval on the same indexed collection.
- Exposure to IR engine design, normalization pipelines, and simple CLI interaction for experimentation.