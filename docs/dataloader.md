# Data Loading API

The Dataloader component loads datasets from the `ir_datasets` library.

## Overview

Dataloader provides a unified interface to load benchmark datasets like:

- **BEIR datasets**: Financial QA (FIQA), Scientific Fact (SciFact), Nutrition Facts (NFCorpus)
- **TREC datasets**: TREC-COVID, TREC-19, TREC-20

Each dataset includes:

- **Corpus**: Collection of documents
- **Queries**: Set of search queries
- **Relevance Judgments (qrels)**: Ground-truth relevance ratings

---

## Dataloader Class

Loads and processes datasets from `ir_datasets`.

```python
from rankers import Dataloader
```

### Parameters

- `dataset_name` (str, required): Dataset identifier in `ir_datasets` format.
  - Format: `"collection/name/split"`
  - Examples:
    - `"beir/fiqa/test"` - BEIR FIQA test set
    - `"beir/scifact/test"` - BEIR SciFact test set
    - `"beir/nfcorpus/test"` - BEIR NFCorpus test set
    - `"trec-covid"` - TREC-COVID
    - `"trec-19"` - TREC-19 (TREC 2019)
    - `"trec-20"` - TREC-20 (TREC 2020)

### Methods

**`load()`**: Load and process the dataset.

```python
dataset = dataloader.load()
```

Returns a `Dataset` object with attributes:

- `corpus` (dict[str, dict[str, str]]): Document collection.
  - Structure: `{doc_id: {"text": "document content"}}`
- `queries` (dict[str, str]): Query set.
  - Structure: `{query_id: "query text"}`
- `relevance_judgments` (dict[str, dict[str, int]]): Ground-truth relevance.
  - Structure: `{query_id: {doc_id: relevance_score}}`
  - Relevance scores: typically 0 (not relevant) or 1 (relevant), but may vary by dataset.

### Usage Example

```python
from rankers import Dataloader

# Load FIQA dataset
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

# Access components
corpus = dataset.corpus
queries = dataset.queries
relevance_judgments = dataset.relevance_judgments

# Iterate through data
for query_id, query_text in list(queries.items())[:5]:
    print(f"Query {query_id}: {query_text}")
    relevant_docs = relevance_judgments.get(query_id, {})
    print(f"  Relevant docs: {list(relevant_docs.keys())[:3]}")

# Get document content
doc_id = list(corpus.keys())[0]
doc_content = corpus[doc_id]["text"]
print(f"Sample document: {doc_content[:100]}...")
```

---

## Supported Datasets

### BEIR Datasets

BEIR (Benchmark for Information Retrieval) provides diverse IR tasks:

- **`beir/fiqa/test`** - Financial QA
  - Domain: Finance/Economics
  - Documents: ~600k
  - Queries: ~648
  - Relevance: 0-1 scale

- **`beir/scifact/test`** - Scientific Fact Verification
  - Domain: Academic/Science
  - Documents: ~5k
  - Queries: ~300
  - Relevance: 0-1 scale

- **`beir/nfcorpus/test`** - Nutrition Facts Corpus
  - Domain: Nutrition/Health
  - Documents: ~3.6k
  - Queries: ~322
  - Relevance: 0-1 scale

- **`beir/trec-covid/test`** - COVID-19 Research Papers
  - Domain: Medical/Research
  - Documents: ~171k
  - Queries: ~50
  - Relevance: 0-2 scale

### TREC Datasets

TREC (Text REtrieval Conference) benchmark datasets:

- **`trec-covid`** - TREC-COVID
  - Documents: ~171k
  - Queries: ~50
  - Relevance: Multi-level (0-2)

- **`trec-19`** - TREC 2019
  - Documents: ~8.8m
  - Queries: ~200k+
  - Relevance: Binary (0-1)

- **`trec-20`** - TREC 2020
  - Documents: ~8.8m
  - Queries: ~200k+
  - Relevance: Binary (0-1)

### Dataset Comparison

| Dataset | Docs | Queries | Domain | Relevance |
|---------|------|---------|--------|-----------|
| FIQA | 600k | 648 | Finance | Binary |
| SciFact | 5k | 300 | Science | Binary |
| NFCorpus | 3.6k | 322 | Nutrition | Binary |
| TREC-COVID | 171k | 50 | Medical | 0-2 scale |
| TREC-19 | 8.8m | 200k+ | General | Binary |
| TREC-20 | 8.8m | 200k+ | General | Binary |

---

## Dataset Structure

### Corpus

Document collection with content indexed by document ID.

```python
corpus = {
    "doc_id_1": {"text": "Document content for ID 1"},
    "doc_id_2": {"text": "Document content for ID 2"},
    # ... more documents
}

# Access a document
doc_content = corpus["doc_id_1"]["text"]
```

**Structure:**

- Outer dict key: Document ID (string)
- Inner dict: Always contains `"text"` key with document content
- May contain other metadata keys depending on dataset

### Queries

Query set indexed by query ID.

```python
queries = {
    "q1": "What are machine learning algorithms?",
    "q2": "How does deep learning work?",
    # ... more queries
}

# Access a query
query_text = queries["q1"]
```

**Structure:**

- Dict key: Query ID (string)
- Value: Query text (string)

### Relevance Judgments

Ground-truth relevance ratings for query-document pairs.

```python
relevance_judgments = {
    "q1": {
        "doc_id_1": 1,  # relevant
        "doc_id_3": 1,  # relevant
        "doc_id_7": 0,  # not relevant
        # ... more docs for q1
    },
    "q2": {
        "doc_id_2": 1,  # relevant
        # ... more docs for q2
    },
    # ... more queries
}

# Access relevance for a query-document pair
relevance = relevance_judgments["q1"]["doc_id_1"]  # 1 or 0
```

**Structure:**

- Outer dict key: Query ID (string)
- Inner dict key: Document ID (string)
- Inner dict value: Relevance score (int, typically 0 or 1)

---

## Usage Examples

### Basic Loading

```python
from rankers import Dataloader

# Load dataset
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

print(f"Corpus size: {len(dataset.corpus)}")
print(f"Num queries: {len(dataset.queries)}")
```

### Explore Dataset

```python
# Print first 3 queries
for query_id, query_text in list(dataset.queries.items())[:3]:
    print(f"\nQuery {query_id}: {query_text}")
    
    # Find relevant documents for this query
    relevant_docs = dataset.relevance_judgments.get(query_id, {})
    relevant_doc_ids = [doc_id for doc_id, rel in relevant_docs.items() if rel > 0]
    
    print(f"  Relevant docs: {relevant_doc_ids[:5]}")
    
    # Print content of first relevant doc
    if relevant_doc_ids:
        doc_id = relevant_doc_ids[0]
        doc_content = dataset.corpus[doc_id]["text"]
        print(f"  Sample relevant doc: {doc_content[:100]}...")
```

### Convert to Haystack Documents

```python
from haystack import Document

# Convert corpus to Document objects
documents = [
    Document(id=doc_id, content=dataset.corpus[doc_id]["text"])
    for doc_id in dataset.corpus.keys()
]

print(f"Created {len(documents)} Document objects")
```

### Prepare for Ranking

```python
from rankers import PairwiseLLMRanker

# Load data
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

# Create ranker
ranker = PairwiseLLMRanker(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    method="heapsort",
    top_k=10,
    device="cuda"
)

# Process queries
from haystack import Document

for query_id, query_text in list(dataset.queries.items())[:5]:
    # Convert corpus to Documents
    docs = [
        Document(id=doc_id, content=dataset.corpus[doc_id]["text"])
        for doc_id in dataset.corpus.keys()
    ]
    
    # Rank documents
    result = ranker.run(documents=docs, query=query_text)
    
    # Process ranked results
    ranked_docs = result["documents"]
    print(f"Top-3 for query {query_id}:")
    for i, doc in enumerate(ranked_docs[:3], 1):
        print(f"  {i}. {doc.content[:60]}...")
```

---

## Working with Custom Datasets

Custom datasets must match the standard structure:

```python
# Required attributes
dataset.corpus = {"doc_id": {"text": "content"}, ...}
dataset.queries = {"query_id": "query text", ...}
dataset.relevance_judgments = {"query_id": {"doc_id": relevance_score, ...}, ...}
```

Relevance scores are typically integers (0 = not relevant, 1 = relevant).

---

## Complete Example

```python
from rankers import Dataloader, PairwiseLLMRanker, Evaluator, EvaluatorParams
from haystack import Document

# Load dataset
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

print(f"Loaded dataset:")
print(f"  Corpus: {len(dataset.corpus)} documents")
print(f"  Queries: {len(dataset.queries)} queries")
print(f"  Relevance judgments: {sum(len(v) for v in dataset.relevance_judgments.values())} pairs")

# Create ranker
ranker = PairwiseLLMRanker(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    method="heapsort",
    top_k=10,
    device="cuda"
)

# Rank all queries
run_results = {}
for query_id, query_text in dataset.queries.items():
    # Convert corpus to Documents
    docs = [
        Document(id=doc_id, content=dataset.corpus[doc_id]["text"])
        for doc_id in dataset.corpus.keys()
    ]
    
    # Rank
    result = ranker.run(documents=docs, query=query_text)
    
    # Store results
    run_results[query_id] = {
        doc.id: i for i, doc in enumerate(result["documents"])
    }

# Evaluate
evaluator = Evaluator(
    relevance_judgments=dataset.relevance_judgments,
    run_results=run_results,
    config=EvaluatorParams(cutoff_values=(1, 5, 10))
)

metrics = evaluator.evaluate()
print(f"NDCG@10: {metrics.ndcg['NDCG@10']}")
print(f"MAP@10: {metrics.map['MAP@10']}")
```

---

## See Also

- [Evaluation API](./evaluation.md) - How to evaluate rankings
- [Ranker APIs](./rankers.md) - How to run rankers on loaded data (detailed parameter descriptions)
- [Configuration](./configuration.md) - How to configure pipelines with dataset settings
- [ir_datasets Documentation](https://ir-datasets.com/) - Full list of available datasets
- [Main Documentation](./README.md) - Data loading workflow overview
