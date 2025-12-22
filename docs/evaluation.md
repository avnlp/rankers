# Evaluation API

The Evaluator component measures ranker quality by comparing ranked results against ground-truth relevance judgments using standard Information Retrieval metrics.

## Overview

Evaluation workflow:

1. **Load dataset** with ground-truth relevance judgments (qrels)
2. **Run ranker** on queries to get ranked results
3. **Create Evaluator** with qrels and run results
4. **Compute metrics** (NDCG, MAP, Recall, Precision) at specified cutoffs
5. **Analyze results** to understand ranker performance

---

## EvaluatorParams

Configuration for evaluation behavior.

```python
from rankers import EvaluatorParams
```

### Parameters

- `cutoff_values` (tuple[int, ...], default: `(1, 3, 5, 10)`): Cutoff points for metrics.
  - Metrics are computed at each cutoff level (e.g., NDCG@5, NDCG@10).
  - Example: `(1, 3, 5, 10)` computes metrics at ranks 1, 3, 5, and 10.
  - Must contain positive integers.

- `ignore_identical_ids` (bool, default: `True`): Whether to exclude query-document pairs where query ID matches document ID.
  - Used to prevent accidental test set leakage in evaluation.
  - Set to `False` if your dataset intentionally includes such pairs.

- `decimal_precision` (int, default: `4`): Number of decimal places for rounding metric values.
  - Range: 0-6.
  - Example: `4` rounds to 4 decimal places (e.g., 0.7234).

- `metrics_to_compute` (tuple[str, ...], default: `("ndcg", "map", "recall", "precision")`): Metrics to compute.
  - `"ndcg"`: Normalized Discounted Cumulative Gain (best for ranking quality).
  - `"map"`: Mean Average Precision (balances precision and recall).
  - `"recall"`: Recall@k (fraction of relevant documents retrieved).
  - `"precision"`: Precision@k (fraction of retrieved documents that are relevant).

### Customization Examples

```python
# Only compute NDCG at top-5 and top-10
config = EvaluatorParams(
    cutoff_values=(5, 10),
    metrics_to_compute=("ndcg",),
    decimal_precision=4
)

# Include identical query-doc pairs
config = EvaluatorParams(
    ignore_identical_ids=False,
    metrics_to_compute=("ndcg", "map", "recall", "precision")
)

# High precision output
config = EvaluatorParams(
    cutoff_values=(1, 3, 5, 10, 20),
    decimal_precision=6
)
```

---

## Evaluator Class

Evaluates ranked documents against ground-truth relevance judgments.

```python
from rankers import Evaluator, EvaluatorParams
```

### Constructor

```python
evaluator = Evaluator(
    relevance_judgments=qrels,  # {query_id: {doc_id: relevance_score}}
    run_results=results,        # {query_id: {doc_id: ranking_score}}
    config=EvaluatorParams(...)
)
```

**Parameters:**

- `relevance_judgments` (dict[str, dict[str, int]]): Ground-truth relevance judgments.
  - Structure: `{query_id: {doc_id: relevance_score}}`
  - Relevance scores typically 0 (not relevant) or 1 (relevant), but can vary by dataset

- `run_results` (dict[str, dict[str, float]]): Ranked results from your ranker.
  - Structure: `{query_id: {doc_id: ranking_score}}`
  - Ranking score can be position (float) or confidence score

- `config` (EvaluatorParams, optional): Evaluation configuration. Defaults to standard settings.

### Methods

**`evaluate()`**: Compute evaluation metrics.

```python
metrics = evaluator.evaluate()
# Returns EvaluationMetrics object
```

**`evaluate_metrics` (property)**: Access the last computed metrics.

```python
metrics = evaluator.evaluate_metrics
print(metrics.ndcg)  # NDCG scores
```

### Output

Returns an `EvaluationMetrics` object with averaged scores across all queries:

```python
{
    "ndcg": {
        "NDCG@1": 0.5234,
        "NDCG@5": 0.6123,
        "NDCG@10": 0.6890,
    },
    "map": {
        "MAP@1": 0.4500,
        "MAP@5": 0.5234,
        "MAP@10": 0.5890,
    },
    "recall": {
        "RECALL@1": 0.1200,
        "RECALL@5": 0.3400,
        "RECALL@10": 0.5600,
    },
    "precision": {
        "PRECISION@1": 0.5200,
        "PRECISION@5": 0.3400,
        "PRECISION@10": 0.2100,
    }
}
```

Scores are **averages across all queries** in the evaluation set.

---

## Usage Example

Complete end-to-end evaluation:

```python
from rankers import PairwiseLLMRanker, Evaluator, EvaluatorParams, Dataloader
from haystack import Document

# Load dataset
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

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
     # Convert corpus to Document objects
     docs = [
         Document(id=doc_id, content=dataset.corpus[doc_id]["text"])
         for doc_id in dataset.corpus.keys()
     ]
     
     # Rank documents
     result = ranker.run(documents=docs, query=query_text)
     
     # Store ranking: {doc_id: score}
     run_results[query_id] = {
         doc.id: i for i, doc in enumerate(result["documents"])
     }

# Create evaluator
evaluator = Evaluator(
     relevance_judgments=dataset.relevance_judgments,
     run_results=run_results,
     config=EvaluatorParams(
         cutoff_values=(1, 3, 5, 10),
         metrics_to_compute=("ndcg", "map", "recall", "precision"),
         decimal_precision=4
     )
 )

# Evaluate
metrics = evaluator.evaluate()

# Access metrics
print(f"NDCG@10: {metrics.ndcg['NDCG@10']}")
print(f"MAP@5: {metrics.map['MAP@5']}")
print(f"Recall@10: {metrics.recall['RECALL@10']}")

# Print all metrics
for metric_name, scores in metrics.items():
    print(f"\n{metric_name.upper()}:")
    for cutoff, score in scores.items():
        print(f"  {cutoff}: {score}")
```

---

## Metrics Reference

### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality by discounting relevance scores based on position. Higher-ranked relevant documents contribute more to the score.

**Description:**

- Range: 0-1 (1 = perfect ranking)
- Best for: Evaluating search result quality
- Interpretation: How much better is the ranking compared to random? Higher is better

**Formula Interpretation:**

- Relevance is discounted by log position (lower positions valued more)
- Normalized by ideal ranking to handle varying numbers of relevant documents
- NDCG@k considers only top-k documents

**When to use:**

- Standard choice for ranking evaluation
- Works with graded relevance (0, 1, 2, 3, etc.)
- Preferred when document order matters

### MAP (Mean Average Precision)

Average precision across all relevant documents. Balances precision and recall.

**Description:**

- Range: 0-1 (1 = all relevant docs ranked first)
- Best for: Binary relevance scenarios
- Interpretation: How early are relevant documents in the ranking?

**Formula Interpretation:**

- Precision is computed at each relevant document position
- Averaged across all relevant documents
- Penalizes non-relevant documents above relevant ones

**When to use:**

- Binary relevance (relevant/not-relevant only)
- When you care about overall recall of relevant documents
- Information retrieval benchmarks

### Recall

Fraction of relevant documents retrieved in top-k.

**Description:**

- Range: 0-1 (1 = all relevant documents retrieved)
- Formula: Relevant documents in top-k / Total relevant documents
- Interpretation: What fraction of relevant documents did we find?

**When to use:**

- When missing relevant documents is costly
- Evaluating coverage of relevant information
- Combined with precision for F1-like analysis

### Precision

Fraction of retrieved documents that are relevant in top-k.

**Description:**

- Range: 0-1 (1 = all retrieved documents are relevant)
- Formula: Relevant documents in top-k / k
- Interpretation: How many of the top-k results are actually relevant?

**When to use:**

- When false positives are costly
- Evaluating result quality for users
- Paired with recall for comprehensive analysis

---

## Metric Selection Guide

| Use Case | Best Metric | Why |
|----------|------------|-----|
| Overall ranking quality | NDCG | Considers position and works with graded relevance |
| Binary relevance | MAP | Designed for relevant/not-relevant judgments |
| Coverage of relevant docs | Recall | Ensures we find all relevant information |
| Result quality for users | Precision | Measures how many results are actually useful |
| Balanced evaluation | NDCG@10 + MAP + Recall@10 | Covers quality, coverage, and position |

### Interpreting Scores

- **NDCG@10 = 0.75**: Ranking achieves 75% of ideal ranking quality
- **MAP@10 = 0.65**: On average, relevant documents appear at position 3-4
- **RECALL@10 = 0.82**: Found 82% of all relevant documents
- **PRECISION@10 = 0.55**: 5-6 of top 10 results are relevant

---

## Data Structures

### Relevance Judgments Format

```python
relevance_judgments = {
    "q1": {
        "doc_id_1": 1,  # relevant
        "doc_id_3": 1,  # relevant
        "doc_id_7": 0,  # not relevant
    },
    "q2": {
        "doc_id_2": 1,  # relevant
        "doc_id_5": 0,
    },
    # ... more queries
}
```

**Structure:**

- Outer dict key: query ID
- Inner dict key: document ID
- Inner dict value: relevance score (usually 0 or 1)

### Run Results Format

```python
run_results = {
    "q1": {
        "doc_id_1": 0.95,  # ranking score (higher = better)
        "doc_id_2": 0.87,
        "doc_id_3": 0.65,
    },
    "q2": {
        "doc_id_2": 0.98,
        "doc_id_4": 0.72,
    },
    # ... more queries
}
```

**Structure:**

- Outer dict key: query ID
- Inner dict key: document ID
- Inner dict value: ranking score or position

**From Ranker Output:**

```python
result = ranker.run(documents=docs, query=query)
run_results[query_id] = {
    doc.id: i for i, doc in enumerate(result["documents"])
}
# Converts ranked document list to scoring dict
```

### EvaluationMetrics Output Format

```python
metrics.ndcg    # {"NDCG@1": 0.5, "NDCG@5": 0.6, "NDCG@10": 0.7}
metrics.map     # {"MAP@1": 0.4, "MAP@5": 0.5, "MAP@10": 0.55}
metrics.recall  # {"RECALL@1": 0.1, "RECALL@5": 0.3, "RECALL@10": 0.5}
metrics.precision  # {"PRECISION@1": 0.5, "PRECISION@5": 0.3, "PRECISION@10": 0.2}

# Access specific cutoff
ndcg_at_10 = metrics.ndcg["NDCG@10"]
map_at_5 = metrics.map["MAP@5"]
```

---

## Complete Example

```python
from rankers import Evaluator, EvaluatorParams

# Example data
relevance_judgments = {
    "q1": {"doc_a": 1, "doc_b": 1, "doc_c": 0, "doc_d": 0},
    "q2": {"doc_a": 0, "doc_b": 1, "doc_c": 1, "doc_d": 1},
}

run_results = {
    "q1": {"doc_a": 0.9, "doc_c": 0.8, "doc_b": 0.7, "doc_d": 0.6},
    "q2": {"doc_b": 0.95, "doc_d": 0.85, "doc_c": 0.75, "doc_a": 0.65},
}

# Create evaluator
evaluator = Evaluator(
     relevance_judgments=relevance_judgments,
     run_results=run_results,
     config=EvaluatorParams(
         cutoff_values=(1, 3, 5, 10),
         metrics_to_compute=("ndcg", "map", "recall", "precision"),
         decimal_precision=4
     )
 )

# Evaluate
metrics = evaluator.evaluate()

# Print results
print(f"NDCG@1: {metrics.ndcg['NDCG@1']}")    # 1.0 (doc_a is relevant and ranked 1st)
print(f"NDCG@5: {metrics.ndcg['NDCG@5']}")    # Good (relevant docs near top)
print(f"MAP: {metrics.map['MAP@5']}")
print(f"Recall@5: {metrics.recall['RECALL@5']}")
print(f"Precision@5: {metrics.precision['PRECISION@5']}")
```

---

## Error Handling

**ValueError: Invalid evaluation metrics**

```python
# Wrong
config = EvaluatorParams(metrics_to_compute=("invalid_metric",))

# Correct
config = EvaluatorParams(metrics_to_compute=("ndcg", "map"))
```

**ValueError: Empty relevance judgments**

```python
# Wrong
evaluator = Evaluator(relevance_judgments={}, run_results=results)

# Correct
evaluator = Evaluator(relevance_judgments=qrels, run_results=results)
```

**ValueError: Mismatched query IDs**

```python
# Wrong (run_results has queries not in qrels)
evaluator = Evaluator(
    relevance_judgments={"q1": {...}},
    run_results={"q1": {...}, "q2": {...}}  # q2 not in qrels
)

# Correct (all query IDs match)
evaluator = Evaluator(
    relevance_judgments={"q1": {...}, "q2": {...}},
    run_results={"q1": {...}, "q2": {...}}
)
```

---

## See Also

- [Ranker APIs](./rankers.md) - How to run rankers
- [Data Loading](./dataloader.md) - How to load datasets with qrels
- [Configuration](./configuration.md) - How to configure evaluation settings
- [Main Documentation](./README.md) - Evaluation workflow overview
