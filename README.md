# Rankers

**Paper:** [LLM Rankers](paper/rankers.pdf)

- Haystack components were created for the Listwise, Setwise, Pairwise techniques.
- Also, Haystack components were created for the RankZephyr and RankVicuna (RankLLM).
- The pipelines for setwise, pairwise, listwise and pointwise rankers were created on the FIQA, Sci-Fact, NF-Corpus, TREC-19, and TREC-20 datasets.
- The Dense retrieval pipelines were created using the Mistral, Llama-3, and Phi-3 models.
- The Dense retrieval pipeline using the LLama-3 model and setwise ranking gave the best performance.
- Each dense retrieval pipeline was created for the Mistral, Phi-3 and LLama-3 models and evaluated on the NDCG metric.

## Pointwise Ranking
- In the pointwise ranking method, the reranker takes both the query and a candidate document to directly generate a relevance score. These independent scores assigned to each document are then used to reorder the from the set of documents.
- LLMs are asked to generate whether the candidate document provided is relevant to the query, with the process repeated for each candidate document.
## Pairwise Ranking
- The Pairwise Ranking Prompting, a pair of candidate items along with the user query serve as prompts to guide the LLMs to determine which document is the most relevant to the given query.
- Pairs are then independently fed into the LLM, and the preferred document is determined for each pair. Subsequently, an aggregation function is employed to assign a score to each document based on the inferred pairwise preferences, and the final ranking is established based on the total score assigned to each document.
## Listwise Ranking
- The Listwise Reranker with a LLM takes the query and a list of documents as input and returns a reordered list of the input document identifiers.
- Current listwise approaches use a sliding window method. This involves re-ranking a window of candidate documents, starting from the bottom of the original ranking list and progressing upwards.
## Setwise Ranking
- The Setwise prompting approach instructs LLMs to select the most relevant document to the query from a set of candidate documents.
- The Setwise prompting technique improves the efficiency of Pairwise prompting (PRP) by comparing multiple documents at each step, as opposed to just a pair. The Setwise prompting approach instructs LLMs to select the most relevant document to the query from a set of candidate documents.

## Results
- The RankLlama and RankZephyr rankers performed best on the FIQA, TREC-19, SciFact and NFCorpus datasets.
- The setwise ranker with the heapsort method using the LLama-3 model gave the best performance for the FIQA, TREC-19, SciFact, and NFcorpus datasets.
- The RankLlama and RankZephyr models with the pointwise and listwise ranking gave the best output.  
