# Ranker APIs

Rankers are Haystack components that rerank documents based on their relevance to a query using Large Language Models. Each ranker implements a different ranking technique.

All rankers implement the same interface:

```python
result = ranker.run(documents=documents, query=query)
ranked_docs = result["documents"]  # List of Document objects, ordered by relevance
```

---

## PairwiseLLMRanker

Pairwise ranking compares documents in pairs to determine relevance ranking. The LLM compares each pair of documents and the results are aggregated using a sorting algorithm to produce a final ranking.

### Parameters

- `model_name` (str, required): Hugging Face model identifier or local path to the LLM to use for comparisons.
  - Example: `"meta-llama/Llama-3.1-8B-Instruct"`, `"mistralai/Mistral-7B-Instruct-v0.1"`
  - The model must support instruction-following and text generation.

- `method` (str, default: `"allpair"`): Sorting algorithm for aggregating pairwise comparisons.
  - `"allpair"`: Compares every possible pair of documents. Provides comprehensive comparison.
  - `"heapsort"`: Builds a max-heap structure via pairwise comparisons, then extracts top-k documents.
  - `"bubblesort"`: Iteratively moves the most relevant document to the front via sliding comparisons. Supports early termination when ranking stabilizes.

- `top_k` (int, default: `10`): Number of top-ranked documents to return from the input list.
  - Must be greater than 0.
  - If `top_k` is greater than the number of input documents, all documents are returned.

- `device` (str or None, default: `None`): Device for model inference.
  - `"cuda"`: Use GPU (requires CUDA-capable GPU).
  - `"cpu"`: Use CPU (universally available).
  - `None`: Auto-detect device (prefers CUDA if available).
  - Examples: `"cuda:0"`, `"cuda:1"` for multi-GPU setups.

- `model_kwargs` (dict, default: `{}`): Additional keyword arguments passed to the model during initialization.
  - Example: `{"torch_dtype": "float16"}` for quantized inference.
  - `device` specified here is automatically set to `None` to avoid conflicts with Haystack component device management.

- `tokenizer_kwargs` (dict, default: `{}`): Additional keyword arguments passed to the tokenizer during initialization.
  - Example: `{"padding": "max_length", "truncation": True}`

- `model_class` (type or None, default: `None`): Custom model class to use instead of the default from transformers.
  - Advanced usage only. Leave as `None` for standard models.

- `tokenizer_class` (type or None, default: `None`): Custom tokenizer class to use instead of the default from transformers.
  - Advanced usage only. Leave as `None` for standard models.

### Output

Returns a dictionary with key `"documents"` containing a list of ranked `Document` objects:

```python
{
    "documents": [Document(...), Document(...), ...]
}
```

Documents are ordered by relevance (most relevant first). Non-top-k documents are appended at the end in their original order.

### Usage Example

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

# Prepare documents
documents = [
    Document(content="Machine learning algorithms require careful hyperparameter tuning."),
    Document(content="Deep neural networks are trained using backpropagation."),
    Document(content="Python is used in web development."),
]

# Rank documents
result = ranker.run(
    documents=documents,
    query="What are the latest advances in deep learning architectures?",
    method="heapsort",  # Override initialization method
    top_k=10            # Override initialization top_k
)

ranked_docs = result["documents"]
for i, doc in enumerate(ranked_docs, 1):
    print(f"{i}. {doc.content[:60]}...")
```

### Algorithm Details

**Pairwise Comparison (Tie-breaking):**

Each pair of documents (A, B) is compared in both directions:

- "Is A or B more relevant to the query?"
- "Is B or A more relevant to the query?"

Only when both comparisons agree on a winner is that result used. If they disagree or both indicate a tie, the pair scores are split equally.

**Score Aggregation:**

For each document, a score is computed based on the number of pairwise wins:

- Win: +1 point
- Loss: 0 points
- Tie: +0.5 points

Documents are then sorted by score (descending).

### Sorting Methods

**Allpair Method:**

- Compares every possible pair of documents
- Best for: Small sets (<50 docs) for comprehensive comparison

**Heapsort Method:**

- Builds a max-heap structure through pairwise comparisons, then extracts top-k
- Best for: General purpose use (default recommendation)

**Bubblesort Method:**

- Iteratively moves the most relevant document to the front via sliding comparisons
- Supports early termination when ranking stabilizes
- Best for: Rapid ranking when top-k stabilizes early

---

## SetwiseLLMRanker

Setwise ranking extends pairwise ranking by comparing multiple documents simultaneously (typically 3-5). Instead of comparing pairs, the LLM selects the most relevant document from a set.

### Parameters

- `model_name` (str, required): Hugging Face model identifier or local path to the LLM to use for set comparisons.
  - Example: `"meta-llama/Llama-3.1-8B-Instruct"`, `"mistralai/Mistral-7B-Instruct-v0.1"`

- `method` (str, default: `"heapsort"`): Sorting algorithm for aggregating set comparisons.
  - `"heapsort"`: Builds a multi-child max-heap (children per node = `num_child`), then extracts top-k documents. Efficient tree-based sorting.
  - `"bubblesort"`: Uses a sliding window of size `num_child + 1` to iteratively identify and move relevant documents.

- `num_child` (int, default: `3`): Number of documents to compare simultaneously in each set.
  - For `"heapsort"`: Number of children per node in the heap.
  - For `"bubblesort"`: Window size = `num_child + 1`.
  - Typical range: 2-5 documents.
  - Must be greater than 0.

- `num_permutation` (int, default: `1`): Number of permutations for reranking (kept for backward compatibility).
  - Currently not actively used but maintained for API stability.

- `top_k` (int, default: `10`): Number of top-ranked documents to return.
  - Must be greater than 0.

- `device` (str or None, default: `None`): Device for model inference.
  - `"cuda"`: Use GPU.
  - `"cpu"`: Use CPU.
  - `None`: Auto-detect device.

- `model_kwargs` (dict, default: `{}`): Additional keyword arguments for model initialization.
  - `device` specified here is automatically set to `None`.

- `tokenizer_kwargs` (dict, default: `{}`): Additional keyword arguments for tokenizer initialization.

- `model_class` (type or None, default: `None`): Custom model class.

- `tokenizer_class` (type or None, default: `None`): Custom tokenizer class.

### Output

Returns a dictionary with key `"documents"` containing a list of ranked `Document` objects:

```python
{
    "documents": [Document(...), Document(...), ...]
}
```

Documents are ordered by relevance (most relevant first).

### Usage Example

```python
from rankers import SetwiseLLMRanker
from haystack import Document

# Initialize ranker with 3 documents per comparison
ranker = SetwiseLLMRanker(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    method="heapsort",
    num_child=3,
    top_k=10,
    device="cuda"
)

# Prepare documents
documents = [
    Document(content="Financial markets analysis and trading strategies."),
    Document(content="Machine learning applications in finance."),
    Document(content="Python programming basics."),
    Document(content="Deep learning for market prediction."),
    Document(content="Risk management in investment portfolios."),
]

# Rank documents
result = ranker.run(
    documents=documents,
    query="What are machine learning techniques in finance?",
    method="bubblesort",  # Override initialization method
    top_k=5,              # Override initialization top_k
    num_child=3           # Override initialization num_child
)

ranked_docs = result["documents"]
for i, doc in enumerate(ranked_docs, 1):
    print(f"{i}. {doc.content}")
```

### Algorithm Details

**Multi-child Heap Structure:**

Unlike binary heaps (2 children per node), setwise uses a multi-child heap where each parent node can have `num_child` children. This matches the number of documents compared in each LLM call.

**Heapsort Method:**

1. Build a max-heap by comparing each parent with all its children.
2. Repeatedly extract the root (largest element) and re-heapify.
3. Stop after extracting `top_k` elements.

**Bubblesort Method:**

1. Use a sliding window of size `num_child + 1` (default: 4 documents when `num_child=3`).
2. Compare documents in the window and move the best one to the front.
3. Slide the window downward and repeat until the top `top_k` are identified.
4. Supports early termination when ranking stabilizes.

---

## ListwiseLLMRanker

Listwise ranking processes entire lists of documents using models specifically trained for ranking tasks. It uses a sliding window technique: documents are reranked in windows that move upward through the initial ranking, incorporating decisions from previous windows.

Unlike Pairwise and Setwise rankers that work with any LLM, Listwise rankers use models trained specifically for ranking (RankZephyr, RankVicuna, RankGPT).

### Parameters

- `model_path` (str, default: `"castorini/rank_zephyr_7b_v1_full"`): Path or Hugging Face identifier of the pre-trained ranking model.
  - Supported models: RankZephyr, RankVicuna (local models), RankGPT (OpenAI API).
  - Examples:
    - `"castorini/rank_zephyr_7b_v1_full"` - Full RankZephyr model
    - `"castorini/rank_zephyr_7b_v1_lora"` - LoRA-adapted RankZephyr
    - `"castorini/rank_vicuna_7b_v1"` - RankVicuna model
    - `"gpt-4"` - For RankGPT (requires OpenAI API keys)

- `ranker_type` (str, default: `"zephyr"`): Type of reranker to instantiate.
  - `"zephyr"`: RankZephyr (7B model trained on ranking tasks).
  - `"vicuna"`: RankVicuna (open-source ranking model based on Vicuna).
  - `"rank_gpt"`: RankGPT (uses OpenAI API; requires `openai_api_keys`).

- `context_size` (int, default: `4096`): Maximum context length in tokens that the model can handle.
  - Determines the maximum number of tokens for query + documents in a single window.
  - Must be greater than 0.
  - Common values: 2048, 4096, 8192 depending on the model.

- `num_few_shot_examples` (int, default: `0`): Number of few-shot examples to include in ranking prompts.
  - Range: 0-10 (typical).
  - Higher values provide more in-context examples but consume more context.
  - Useful for adapting the ranker to specific domains with examples.

- `sliding_window_size` (int, default: `20`): Number of documents to rerank in each sliding window.
  - Larger windows consider more context but consume more tokens.
  - Must be greater than 0.
  - Typical range: 10-50 documents per window.

- `sliding_window_step` (int, default: `10`): Number of positions the window moves upward after each reranking step.
  - Controls overlap between consecutive windows.
  - Smaller steps = more overlap = more computation but potentially better ranking quality.
  - Typical range: 5-20.

- `device` (str, default: `"cuda"`): Device for model inference.
  - `"cuda"`: GPU inference.
  - `"cpu"`: CPU inference.
  - For local models (Zephyr, Vicuna) only.

- `num_gpus` (int, default: `1`): Number of GPUs to use for distributed inference.
  - For local models only.
  - Requires proper distributed setup (e.g., `torch.distributed`).

- `variable_passages` (bool, default: `False`): Whether to allow variable number of passages per request.
  - If `False`: All requests must have exactly `sliding_window_size` documents (except the last window).
  - If `True`: Allows variable-length windows for more flexible reranking.

- `system_message` (str, default: `"You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."`): System prompt to guide LLM behavior.
  - Customize to specify ranking criteria (relevance, authority, recency, etc.).
  - Example: `"You are an expert fact-checker. Rank passages by their factual accuracy and relevance to the query."`

- `prompt_mode` (PromptMode, default: `PromptMode.RANK_GPT`): Prompt format to use for LLM instructions.
  - `PromptMode.RANK_GPT`: Default RankGPT prompt format.
  - Other modes may be available depending on the RankLLM version.

- `openai_api_keys` (list[str] or None, default: `None`): List of OpenAI API keys for RankGPT.
  - Required if `ranker_type="rank_gpt"`.
  - Example: `["sk-...", "sk-...", "sk-..."]` for key rotation.
  - Raise error: `ValueError` if `ranker_type="rank_gpt"` and this is `None`.

- `openai_key_start_id` (int or None, default: `None`): Starting index for OpenAI API key rotation.
  - Useful when rotating between multiple keys to distribute rate limit burden.
  - Example: Start from key index 2 if keys 0-1 are rate-limited.

- `openai_proxy` (str or None, default: `None`): Proxy URL for OpenAI API requests.
  - Example: `"http://proxy.company.com:8080"`
  - Useful for corporate networks or proxying through regional endpoints.

- `api_type` (str or None, default: `None`): API type for Azure OpenAI services.
  - `"azure"`: Use Azure OpenAI instead of standard OpenAI.
  - Leave as `None` for standard OpenAI API.

- `api_base` (str or None, default: `None`): API base URL for Azure OpenAI services.
  - Example: `"https://myresource.openai.azure.com/"`
  - Leave as `None` for standard OpenAI API.

- `api_version` (str or None, default: `None`): API version for Azure OpenAI services.
  - Example: `"2023-05-15"`
  - Leave as `None` for standard OpenAI API.

### Output

Returns a dictionary with key `"documents"` containing a list of reranked `Document` objects:

```python
{
    "documents": [Document(...), Document(...), ...]
}
```

Documents are ordered by final ranking (most relevant first). Document scores reflect the final ranking position.

### Usage Example (RankZephyr)

```python
from rankers import ListwiseLLMRanker
from haystack import Document

# Initialize with RankZephyr
ranker = ListwiseLLMRanker(
    model_path="castorini/rank_zephyr_7b_v1_full",
    ranker_type="zephyr",
    device="cuda",
    sliding_window_size=20,
    sliding_window_step=10,
    top_k=10
)

# Prepare documents
documents = [
    Document(content="Document 1 about topic X", meta={"source": "source1"}),
    Document(content="Document 2 about topic Y", meta={"source": "source2"}),
    # ... more documents
]

# Rank documents
result = ranker.run(
    query="What is the main topic discussed?",
    documents=documents,
    top_k=10  # Optional: override top_k at runtime
)

ranked_docs = result["documents"]
for i, doc in enumerate(ranked_docs, 1):
    print(f"{i}. Score: {doc.score:.2f} | {doc.content[:50]}...")
```

### Usage Example (RankGPT)

```python
from rankers import ListwiseLLMRanker

# Initialize with RankGPT (requires OpenAI API keys)
ranker = ListwiseLLMRanker(
    model_path="gpt-4",
    ranker_type="rank_gpt",
    sliding_window_size=20,
    openai_api_keys=["sk-...", "sk-..."],
    openai_key_start_id=0,
    openai_proxy=None,  # Leave as None unless using a proxy
)

result = ranker.run(
    query="Your query here",
    documents=documents,
    top_k=10
)
```

### Sliding Window Algorithm

The sliding window technique works as follows:

1. **Window Setup**: Create a window of `sliding_window_size` documents starting from the bottom of the initial ranking.
2. **Reranking**: Use the LLM to rerank documents within the window.
3. **Window Movement**: Move the window upward by `sliding_window_step` positions.
4. **Repeat**: Continue until all documents have been considered in a window.
5. **Final Assembly**: Merge reranking decisions from all windows to produce the final ranking.

**Example**:

- Initial ranking: [D0, D1, D2, ..., D49]
- Window size: 20, Step: 10

```
Window 1: D49-D30 → rerank locally
Window 2: D39-D20 → rerank locally (overlaps with Window 1)
Window 3: D29-D10 → rerank locally
Window 4: D19-D0  → rerank locally
```

Final ranking combines all decisions with preference for earlier windows.

---

## Common Patterns

### Haystack Pipelines

Rankers integrate with Haystack pipelines:

```python
from haystack import Pipeline
from rankers import PairwiseLLMRanker

pipeline = Pipeline()
pipeline.add_component("ranker", PairwiseLLMRanker(...))
result = pipeline.run({"ranker": {"documents": docs, "query": query}})
```

### Chaining Rankers

Combine multiple rankers for iterative reranking:

```python
# First pass: Setwise
setwise = SetwiseLLMRanker(method="heapsort", top_k=20)
result1 = setwise.run(documents=docs, query=query)

# Second pass: Pairwise on top-20
pairwise = PairwiseLLMRanker(method="allpair", top_k=10)
result2 = pairwise.run(documents=result1["documents"], query=query)
```

### Dynamic Parameters

Override parameters at runtime:

```python
ranker = PairwiseLLMRanker(model_name="...", method="heapsort")

# Override method per query
result1 = ranker.run(documents=docs, query=q1, method="heapsort")
result2 = ranker.run(documents=docs, query=q2, method="bubblesort")
result3 = ranker.run(documents=docs, query=q3, top_k=20)
```

### Error Handling

```python
from haystack import Document

try:
    if not documents:
        raise ValueError("Empty documents list")
    
    result = ranker.run(documents=documents, query=query)
    
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    return []
except RuntimeError as e:
    logger.error(f"Ranker error (OOM?): {e}")
    # Fall back to CPU
    ranker.device = "cpu"
    return ranker.run(documents=documents, query=query)
```

---

## See Also

- [Configuration Reference](./configuration.md) - Ranker config parameters
- [Evaluation Metrics](./evaluation.md) - How to evaluate ranker quality
- [Pipeline Examples](../src/rankers/pipelines/) - Full working examples
- [Main Documentation](./README.md) - Overview and quick start
