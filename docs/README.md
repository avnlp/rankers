# Rankers Documentation

- **[API Reference for Rankers](./rankers.md)** - Complete API documentation for PairwiseLLMRanker, SetwiseLLMRanker, and ListwiseLLMRanker
- **[Evaluation Metrics](./evaluation.md)** - Evaluator API, metrics definitions (NDCG, MAP, Recall, Precision), and usage examples
- **[Data Loading](./dataloader.md)** - Dataloader class, supported datasets, and dataset structure
- **[Configuration Management](./configuration.md)** - Configuration schemas, YAML examples, and loading patterns

## Core Concepts

### Ranker Interface

All rankers implement a Haystack component interface:

```python
result = ranker.run(documents=documents, query=query)
ranked_docs = result["documents"]
```

Output is always a dictionary with key `"documents"` containing a list of ranked `Document` objects, ordered by relevance (most relevant first).

### Evaluation Workflow

Evaluate ranker quality using standard IR metrics:

1. Load a dataset (FIQA, SciFact, NFCorpus, TREC-19/20)
2. Run ranker on queries
3. Compare results to ground-truth relevance judgments
4. Compute metrics (NDCG, MAP, Recall, Precision)

See [Evaluation Metrics](./evaluation.md) for detailed metric definitions.

### Configuration Approach

For production pipelines, use YAML configuration files that manage:

- Dataset selection
- Embedding models
- Retrieval settings
- Ranker parameters
- Evaluation metrics

See [Configuration Management](./configuration.md) for schemas and examples.

## Common Patterns

### Basic Pipeline (In-Memory)

```python
from rankers import PairwiseLLMRanker
from haystack import Document

# Initialize ranker
ranker = PairwiseLLMRanker(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    method="heapsort",
    top_k=10,
    device="cuda"
)

# Rank documents
documents = [Document(content="..."), ...]
result = ranker.run(documents=documents, query="your query")
ranked_docs = result["documents"]
```

### Configuration-Driven Pipeline

```python
from pathlib import Path
from rankers.config import PairwiseRankingConfig, load_config
from rankers import PairwiseLLMRanker

# Load config
config = load_config(Path("config.yaml"), PairwiseRankingConfig)

# Create ranker from config
ranker = PairwiseLLMRanker(
    model_name=config.llm.model_name,
    method=config.llm.method,
    top_k=config.llm.top_k,
    device=config.llm.device,
)
```

### Evaluation Pipeline

```python
from rankers import Dataloader, Evaluator, EvaluatorParams

# Load dataset
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

# Create evaluator
evaluator = Evaluator(
    relevance_judgments=dataset.relevance_judgments,
    run_results=run_results,
    config=EvaluatorParams(cutoff_values=(1, 5, 10))
)

# Evaluate
metrics = evaluator.evaluate()
print(f"NDCG@10: {metrics.ndcg['NDCG@10']}")
```
