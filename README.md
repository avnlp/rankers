# Rankers

**Paper:** [LLM Rankers](paper/rankers.pdf)

Large Language Models (LLMs) can be efficiently used for document ranking in information retrieval tasks. Compared to traditional methods, LLMs provide more accurate and contextually relevant rankings. The zero-shot capability of LLMs allows them to be used for ranking without extensive training on specific datasets.

We release implementations for the three types of LLM based ranking techniques:

- [Pairwise Ranking](https://arxiv.org/abs/2306.17563)
- [Setwise Ranking](https://arxiv.org/abs/2310.09497)
- [Listwise Ranking](https://arxiv.org/abs/2305.02156)

Our implementation includes modular Pairwise, Setwise, and Listwise ranker components for the [Haystack](https://haystack.deepset.ai/) LLM framework. These rankers leverage structured generation and robust Pydantic validation to ensure accurate zero-shot ranking, even on smaller LLMs.

**Key Features**:

- **Efficient Sorting Algorithms**: The Pairwise and Setwise rankers utilize efficient sorting methods (Heapsort and Bubblesort) to speed up inference.
- **Sliding Window Technique**: The Listwise ranker uses a sliding window approach to handle long lists of candidates efficiently.
- **Integration with RankLLM**: The Listwise ranker integrates with the [RankLLM](https://github.com/castorini/rank_llm) framework and supports LLMs specifically trained for ranking (such as RankGPT, RankLlama, and RankZephyr).
- **Evaluation Toolkit**: We provide a custom Evaluator and Dataloader for evaluating rankers on standard metrics (NDCG, MAP, Recall, Precision) at various cutoffs. The Dataloader efficiently loads and processes datasets using the `ir_datasets` library.

The core sorting and ranking implementation for the Pairwise and Setwise rankers is based on the implementation by [ielab/llm-rankers](https://github.com/ielab/llm-rankers).

## Ranking Techniques

### Pairwise Ranking

- In Pairwise ranking, the LLM compares two candidate documents at a time to determine which is more relevant to the query.
- Each pair is independently fed into the LLM, and the preferred document is recorded.
- An aggregation function then assigns a score to each document based on the pairwise preferences, and the final ranking is determined by these scores.
- Our implementation uses efficient sorting algorithms (like Heapsort) to minimize the number of comparisons.

### Setwise Ranking

- Setwise ranking extends the pairwise approach by comparing multiple documents simultaneously.
- The LLM is prompted to select the most relevant document from a set of candidates.
- This method reduces the number of inference steps required compared to pairwise ranking.
- Like the pairwise ranker, our setwise implementation uses efficient sorting algorithms for speed.

### Listwise Ranking

- Listwise ranking processes a list of candidate documents in a single step.
- Our implementation uses a sliding window technique: it re-ranks a window of candidate documents, starting from the bottom of the initial ranking and moving upwards.
- This approach is integrated with the RankLLM framework and supports specialized ranking models (e.g., RankGPT, RankLlama, RankZephyr).

## Evaluation

We evaluated the rankers using pipelines built with the Haystack framework. The evaluation was conducted on the following datasets: FIQA, SciFact, NFCorpus, TREC-19, and TREC-20.

The evaluation pipelines can be found in the [pipelines](src/rankers/pipelines) directory.

**Models Used**:  

- **Pairwise and Setwise Rankers**: `Mistral`, `Phi-3`, and `Llama-3`.
- **Listwise Ranker**: `RankLlama` and `RankZephyr` (models specifically trained for ranking).

**Evaluation Results**:

We report the `NDCG@10` scores for each dataset and method in the table below:

| **Model**        | **Ranker**    | **Method**     | **FiQA**   | **SciFACT** | **NFCorpus** | **TREC-19** | **TREC-20** |
|---------------|------------|-------------|--------:|-----------:|------------:|-----------:|-----------:|
| `Instructor-XL` |     \-    |      \-    | 0.4650 | 0.6920  | 0.4180   | 0.5230  | 0.5040  |
| `Mistral`       | Pairwise  | Heapsort   | 0.4660 | 0.6940  | 0.4270   | 0.7080  | 0.6890  |
| `Mistral`       | Pairwise  | Bubblesort | 0.4660 | 0.6940  | 0.4280   | 0.7090  | 0.6920  |
| `Mistral`       | Setwise   | Bubblesort | 0.4680 | 0.6950  | 0.4300   | 0.7110  | 0.6940  |
| `Mistral`       | Setwise   | Heapsort   | 0.4680 | 0.6960  | 0.4310   | 0.7140  | 0.6950  |
| `Phi-3`         | Setwise   | Bubblesort | 0.4690 | 0.7060  | 0.4350   | 0.7190  | 0.7010  |
| `Phi-3`         | Setwise   | Bubblesort | 0.4700 | 0.7080  | 0.4360   | 0.7190  | 0.7010  |
| `Phi-3`         | Pairwise  | Heapsort   | 0.4710 | 0.7110  | 0.4380   | 0.7210  | 0.7020  |
| `Phi-3`         | Setwise   | Heapsort   | 0.4710 | 0.7120  | 0.4390   | 0.7220  | 0.7030  |
| `Llama-3`       | Pairwise  | Bubblesort | 0.4730 | 0.7630  | 0.4400   | 0.7390  | 0.7220  |
| `Llama-3`       | Pairwise  | Heapsort   | 0.4740 | 0.7670  | 0.4410   | 0.7410  | 0.7230  |
| `Llama-3`       | Setwise   | Bubblesort | 0.4740 | 0.7720  | 0.4420   | 0.7440  | 0.7250  |
| `Llama-3`       | Setwise   | Heapsort   | 0.4760 | 0.7760  | 0.4430   | 0.7460  | 0.7270  |
| `RankLlama`     | Listwise  | \-         | 0.4796 | 0.7812  | 0.4518   | 0.7511  | 0.7642  |
| `RankZephyr`    | Listwise  | \-         | **0.4892** | **0.7891**  | **0.4578**   | **0.7693**  | **0.7743**  |

- All rankers performed closely across all datasets.
- `RankLlama` and `RankZephyr` (used with the Listwise ranker) consistently achieved slightly better results than the other rankers.
- Among the models not explicitly trained for reranking, the `Llama-3` model with the Setwise and Pairwise ranker performed the best.
