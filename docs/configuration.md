# Configuration API

Configuration management for ranking pipelines using YAML files and Pydantic schemas. This enables reproducible, version-controlled setups for complex evaluation workflows.

---

## PairwiseRankingConfig

Configuration schema for Pairwise LLM Ranker pipelines.

```python
from rankers.config import PairwiseRankingConfig, load_config
from pathlib import Path
```

### Configuration Fields

- `dataset` (DatasetConfig): Dataset selection
  - `name` (str): ir_datasets identifier (e.g., `"beir/fiqa/test"`)

- `llm` (LLMConfig): Pairwise ranker settings
  - `model_name` (str): Hugging Face model identifier
  - `method` (str): One of `"heapsort"`, `"bubblesort"`, `"allpair"` (see [Ranker APIs](./rankers.md) for detailed descriptions)
  - `top_k` (int): Number of top documents to return
  - `device` (str or None): `"cuda"`, `"cpu"`, or `None` for auto-detect
  - `model_kwargs` (dict): Additional model initialization parameters
  - `tokenizer_kwargs` (dict): Additional tokenizer parameters

- `embedding` (EmbeddingConfig): Embedding model for retrieval
  - `model` (str): Embedding model identifier
  - `model_kwargs` (dict): Additional embedding model parameters

- `milvus` (MilvusConfig): Vector database settings
  - `connection_uri` (str): Milvus server URI
  - `connection_token` (str): Authentication token
  - `document_store_kwargs` (dict): Additional Milvus parameters

- `retrieval` (RetrievalConfig): Retrieval pipeline settings
  - `documents_to_retrieve` (int): Number of documents to retrieve per query
  - `filters` (dict): Metadata filters for filtering documents

- `evaluation` (EvaluationConfig): Evaluation metrics configuration
  - `cutoff_values` (list[int]): Metric cutoffs (e.g., `[1, 5, 10]`)
  - `metrics_to_compute` (list[str]): Metrics to compute
  - `ignore_identical_ids` (bool): Filter query-doc pairs with same ID
  - `decimal_precision` (int): Decimal places for rounding

### YAML Example

```yaml
# pairwise_ranking_config.yaml

dataset:
  name: "beir/fiqa/test"

llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  method: "heapsort"
  top_k: 10
  device: "cuda"
  model_kwargs: {}
  tokenizer_kwargs: {}

embedding:
  model: "hkunlp/instructor-xl"
  model_kwargs:
    device: "cuda"

milvus:
  connection_uri: "http://localhost:19530"
  connection_token: "milvus_token"
  document_store_kwargs:
    collection_name: "fiqa"

retrieval:
  documents_to_retrieve: 50
  filters: {}

evaluation:
  cutoff_values: [1, 3, 5, 10]
  metrics_to_compute: ["ndcg", "map", "recall", "precision"]
  ignore_identical_ids: true
  decimal_precision: 4
```

### Loading Configuration

```python
from pathlib import Path
from rankers.config import PairwiseRankingConfig, load_config
from rankers import PairwiseLLMRanker

# Load config from YAML
config = load_config(
    Path("pairwise_ranking_config.yaml"),
    PairwiseRankingConfig
)

# Create ranker from config
ranker = PairwiseLLMRanker(
    model_name=config.llm.model_name,
    method=config.llm.method,
    top_k=config.llm.top_k,
    device=config.llm.device,
)

# Use other config values
print(f"Dataset: {config.dataset.name}")
print(f"Embedding: {config.embedding.model}")
print(f"Cutoffs: {config.evaluation.cutoff_values}")
```

### Python Example

```python
from rankers.config import PairwiseRankingConfig, DatasetConfig, LLMConfig
from rankers import PairwiseLLMRanker

# Create config programmatically
config = PairwiseRankingConfig(
    dataset=DatasetConfig(name="beir/fiqa/test"),
    llm=LLMConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        method="heapsort",
        top_k=10,
        device="cuda",
    ),
    # ... other configs
)

# Use config
ranker = PairwiseLLMRanker(**config.llm.dict())
```

---

## SetwiseRankingConfig

Configuration schema for Setwise LLM Ranker pipelines.

Similar structure to `PairwiseRankingConfig` with Setwise-specific parameters in `llm` section.

### Configuration Fields (LLM-specific)

- `llm.method`: One of `"heapsort"`, `"bubblesort"` (see [Ranker APIs](./rankers.md) for detailed descriptions)
- `llm.num_child` (int): Number of documents to compare per set
- `llm.num_permutation` (int): Number of permutations (for backward compatibility)

All other fields (`dataset`, `embedding`, `milvus`, `retrieval`, `evaluation`) are identical to Pairwise.

### YAML Example

```yaml
# setwise_ranking_config.yaml

dataset:
  name: "beir/scifact/test"

llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  method: "heapsort"
  num_child: 3
  num_permutation: 1
  top_k: 10
  device: "cuda"
  model_kwargs: {}
  tokenizer_kwargs: {}

embedding:
  model: "hkunlp/instructor-xl"
  model_kwargs:
    device: "cuda"

milvus:
  connection_uri: "http://localhost:19530"
  connection_token: "milvus_token"
  document_store_kwargs:
    collection_name: "scifact"

retrieval:
  documents_to_retrieve: 50
  filters: {}

evaluation:
  cutoff_values: [1, 3, 5, 10]
  metrics_to_compute: ["ndcg", "map", "recall", "precision"]
  ignore_identical_ids: true
  decimal_precision: 4
```

### Python Example

```python
from rankers.config import SetwiseRankingConfig, load_config
from rankers import SetwiseLLMRanker

config = load_config(Path("setwise_ranking_config.yaml"), SetwiseRankingConfig)

ranker = SetwiseLLMRanker(
    model_name=config.llm.model_name,
    method=config.llm.method,
    num_child=config.llm.num_child,
    top_k=config.llm.top_k,
    device=config.llm.device,
)
```

---

## ListwiseRankingConfig

Configuration schema for Listwise LLM Ranker pipelines.

Listwise config uses `ranker` instead of `llm` and supports two variants: Hugging Face models and OpenAI API.

### Configuration Fields

- `dataset` (DatasetConfig): Dataset configuration
  - `name` (str): ir_datasets identifier

- `ranker` (ListwiseHFConfig): Listwise ranker configuration
  - `model_path` (str): Model path for RankZephyr, RankVicuna, or RankGPT
  - `ranker_type` (str): One of `"zephyr"`, `"vicuna"`, `"rank_gpt"`
  - `context_size` (int): Maximum context size in tokens
  - `num_few_shot_examples` (int): Number of few-shot examples
  - `device` (str): Device for inference (`"cuda"`, `"cpu"`)
  - `num_gpus` (int): Number of GPUs for distributed inference
  - `variable_passages` (bool): Allow variable number of passages
  - `sliding_window_size` (int): Documents per window
  - `sliding_window_step` (int): Window movement step size
  - `system_message` (str): System prompt
  - `top_k` (int, optional): Final number of documents to return (can be overridden at runtime)
  - `openai` (ListwiseOpenAIConfig or None): OpenAI configuration (for RankGPT)
  - `api_keys` (list[str] or None): OpenAI API keys
  - `key_start_id` (int or None): Key rotation start index
  - `proxy` (str or None): Proxy URL
  - `api_type` (str or None): API type (`"azure"` or `None`)
  - `api_base` (str or None): API base URL (Azure)
  - `api_version` (str or None): API version (Azure)

### YAML Example (RankZephyr)

```yaml
# listwise_ranking_config.yaml

dataset:
  name: "beir/nfcorpus/test"

ranker:
  model_path: "castorini/rank_zephyr_7b_v1_full"
  ranker_type: "zephyr"
  context_size: 4096
  num_few_shot_examples: 0
  device: "cuda"
  num_gpus: 1
  variable_passages: false
  sliding_window_size: 20
  sliding_window_step: 10
  top_k: 10
  system_message: "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."
  openai: null

# Optional: embedding, milvus, retrieval, evaluation (same as Pairwise)
```

### YAML Example (RankGPT)

```yaml
# rankgpt_ranking_config.yaml

dataset:
  name: "beir/fiqa/test"

ranker:
  model_path: "gpt-4"
  ranker_type: "rank_gpt"
  context_size: 4096
  num_few_shot_examples: 0
  sliding_window_size: 20
  sliding_window_step: 10
  top_k: 10
  system_message: "Rank the following passages by relevance to the query."
  openai:
    api_keys:
      - "sk-..."
      - "sk-..."
    key_start_id: 0
    proxy: null
    api_type: null
    api_base: null
    api_version: null

evaluation:
  cutoff_values: [1, 5, 10]
  metrics_to_compute: ["ndcg", "map"]
  decimal_precision: 4
```

### Python Example

```python
from rankers.config import ListwiseRankingConfig, load_config
from rankers import ListwiseLLMRanker

# Load YAML config and instantiate ranker
config = load_config(Path("listwise_config.yaml"), ListwiseRankingConfig)

ranker = ListwiseLLMRanker(
    model_path=config.ranker.model_path,
    ranker_type=config.ranker.ranker_type,
    sliding_window_size=config.ranker.sliding_window_size,
    sliding_window_step=config.ranker.sliding_window_step,
    top_k=config.ranker.top_k,
    device=config.ranker.device,
    # For RankGPT, add: openai_api_keys=config.ranker.openai.api_keys, ...
)
```

---

## Shared Base Configs

### DatasetConfig

Configuration for dataset loading.

```python
class DatasetConfig:
    name: str  # ir_datasets identifier
```

### EmbeddingConfig

Configuration for embedding model.

```python
class EmbeddingConfig:
    model: str  # Model identifier
    model_kwargs: dict = {}  # Additional parameters
```

### MilvusConfig

Configuration for vector database.

```python
class MilvusConfig:
    connection_uri: str  # Server URI
    connection_token: str  # Authentication token
    document_store_kwargs: dict = {}  # Additional parameters
```

### RetrievalConfig

Configuration for retrieval pipeline.

```python
class RetrievalConfig:
    documents_to_retrieve: int = 50  # Default 50
    filters: dict = {}  # Optional metadata filters
```

### EvaluationConfig

Configuration for evaluation metrics.

```python
class EvaluationConfig:
    cutoff_values: list[int] = [1, 3, 5, 10]
    metrics_to_compute: list[str] = ["ndcg", "map", "recall", "precision"]
    ignore_identical_ids: bool = True
    decimal_precision: int = 4
```

---

## Loading & Using Configs

### load_config() Function

```python
from rankers.config import load_config, PairwiseRankingConfig
from pathlib import Path

config = load_config(
    Path("path/to/config.yaml"),
    PairwiseRankingConfig
)
```

**Parameters:**

- `config_path` (Path): Path to YAML configuration file
- `config_class` (type): Configuration class to load into (e.g., `PairwiseRankingConfig`)

**Returns:**

- Initialized configuration object with all fields populated

### Python Example

```python
from pathlib import Path
from rankers.config import PairwiseRankingConfig, load_config
from rankers import PairwiseLLMRanker, Dataloader, Evaluator, EvaluatorParams

# Load configuration
config = load_config(Path("config.yaml"), PairwiseRankingConfig)

# Load dataset
loader = Dataloader(config.dataset.name)
dataset = loader.load()

# Create ranker
ranker = PairwiseLLMRanker(
    model_name=config.llm.model_name,
    method=config.llm.method,
    top_k=config.llm.top_k,
    device=config.llm.device,
)

# Run evaluation
run_results = {}
for query_id, query_text in dataset.queries.items():
    # ... rank documents ...
    pass

# Evaluate using config
evaluator = Evaluator(
    relevance_judgments=dataset.relevance_judgments,
    run_results=run_results,
    config=EvaluatorParams(
        cutoff_values=tuple(config.evaluation.cutoff_values),
        metrics_to_compute=tuple(config.evaluation.metrics_to_compute),
        decimal_precision=config.evaluation.decimal_precision,
    )
)

metrics = evaluator.evaluate()
```

### Environment Variables

```python
import os
from pathlib import Path
from rankers.config import load_config, ListwiseRankingConfig

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Load config and override OpenAI keys
config = load_config(Path("config.yaml"), ListwiseRankingConfig)
if api_key:
    config.ranker.openai.api_keys = [api_key]
```

### Config Validation Errors

```python
from rankers.config import load_config, PairwiseRankingConfig

try:
    config = load_config(Path("config.yaml"), PairwiseRankingConfig)
except ValueError as e:
    # Invalid YAML or missing required fields
    print(f"Config error: {e}")
except FileNotFoundError:
    # YAML file not found
    print("Config file not found")
```

---

## Config Examples by Dataset

All datasets require the same base configuration structure. Adjust only:

1. `dataset.name` - ir_datasets identifier
2. `milvus.document_store_kwargs.collection_name` - Unique collection name per dataset
3. `retrieval.documents_to_retrieve` - Tune based on corpus size and query complexity

### FIQA Configuration

```yaml
dataset:
  name: "beir/fiqa/test"

milvus:
  document_store_kwargs:
    collection_name: "fiqa"

retrieval:
  documents_to_retrieve: 50
```

### SciFact Configuration

```yaml
dataset:
  name: "beir/scifact/test"

milvus:
  document_store_kwargs:
    collection_name: "scifact"

retrieval:
  documents_to_retrieve: 50
```

### NFCorpus Configuration

```yaml
dataset:
  name: "beir/nfcorpus/test"

milvus:
  document_store_kwargs:
    collection_name: "nfcorpus"

retrieval:
  documents_to_retrieve: 50
```

### TREC-19 Configuration

```yaml
dataset:
  name: "trec-19"

milvus:
  document_store_kwargs:
    collection_name: "trec19"

retrieval:
  documents_to_retrieve: 100  # Larger corpus
```

### TREC-20 Configuration

```yaml
dataset:
  name: "trec-20"

milvus:
  document_store_kwargs:
    collection_name: "trec20"

retrieval:
  documents_to_retrieve: 100  # Larger corpus
```

---

## Advanced Configuration

### Custom Model Kwargs

```yaml
llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  model_kwargs:
    torch_dtype: "float16"  # Quantization
    load_in_8bit: true      # 8-bit loading
    device_map: "auto"      # Automatic GPU placement
  tokenizer_kwargs:
    padding: "max_length"
    truncation: true
```

### Tokenizer Configuration

```yaml
llm:
  tokenizer_kwargs:
    max_length: 2048
    padding: "max_length"
    truncation: true
    return_tensors: "pt"
```

### Device Management

```yaml
# GPU inference (preferred)
llm:
  device: "cuda"

embedding:
  model_kwargs:
    device: "cuda"

# CPU inference
llm:
  device: "cpu"

# Multi-GPU for Listwise
ranker:
  device: "cuda"
  num_gpus: 2  # Use 2 GPUs
```

### Distributed Setup

```yaml
ranker:
  device: "cuda"
  num_gpus: 4  # Distributed across 4 GPUs
  # Requires torch.distributed setup
```

---

## See Also

- [Ranker APIs](./rankers.md) - Ranker parameters
- [Evaluation API](./evaluation.md) - Evaluation configuration
- [Configuration Definitions](../src/rankers/config/) - Source code for config classes
- [Pipeline Examples](../src/rankers/pipelines/) - Complete pipeline configs
- [Main Documentation](./README.md) - Configuration workflow overview
