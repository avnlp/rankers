import copy
from typing import Optional

import weave
from haystack import Document, component
from transformers import PreTrainedModel, PreTrainedTokenizer

from rankers.setwise.setwise_ranker_output_validator import IDENTIFIERS, SetwiseRankingOutput
from rankers.setwise.setwise_ranking_prompt import SYSTEM_PROMPT, USER_PROMPT
from rankers.utils import StructuredGeneration


@component
class SetwiseLLMRanker:
    """A Setwise LLM Ranker that reranks documents using a language model to compare passages in a setwise manner.

    The ranker supports different methods like heapsort and bubblesort for reranking.

    Usage example:
    ```python
    from haystack.components.rankers import SetwiseLLMRanker
    from haystack import Document

    ranker = SetwiseLLMRanker(model_name="meta-llama/Llama-3.1-8B", top_k=10)
    documents = [Document(content="Paris"), Document(content="Berlin"), Document(content="Madrid")]
    result = ranker.run(documents=documents, query="Which city is the capital of France?")
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_class: Optional[type[PreTrainedModel]] = None,
        tokenizer_class: Optional[type[PreTrainedTokenizer]] = None,
        method: str = "heapsort",
        num_permutation: int = 1,
        num_child: int = 3,
        top_k: int = 10,
    ):
        """Initialize the Setwise LLM Ranker.

        Args:
            model_name: Name of the LLM model to use.
            device: Device to run the model on.
            model_kwargs: Additional keyword arguments for model initialization.
            tokenizer_kwargs: Additional keyword arguments for tokenizer initialization.
            model_class: (Optional) Specific model class.
            tokenizer_class: (Optional) Specific tokenizer class.
            method: Reranking method; either "heapsort" or "bubblesort".
            num_permutation: Number of permutations (kept for compatibility with the original signature).
            num_child: Number of children per node (used in multi-child heapify and sliding-window bubblesort).
            top_k: Maximum number of top documents to rerank.
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.method = method
        self.num_permutation = num_permutation
        self.num_child = num_child
        self.top_k = top_k

        self._structured_generation_model = None
        self._is_warmed_up = False

    def _warm_up(self):
        """Initialize the LLM and tokenizer if they have not been initialized already."""
        if not self._is_warmed_up:
            self._structured_generation_model = StructuredGeneration(
                model_name=self.model_name,
                device=self.device,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                model_class=self.model_class,
                tokenizer_class=self.tokenizer_class,
            )
            self._is_warmed_up = True

    def compare(self, query: str, docs: list[Document]) -> str:
        """Use the LLM to compare a list of documents (passages).

        Args:
            query: The query string.
            docs: A list of Document objects whose content will be compared.

        Returns:
            A string corresponding to one of the identifiers from IDENTIFIERS.
        """
        # Build the passages string using the provided identifiers.
        passages = "\n\n".join([f'Passage {IDENTIFIERS[i]}: "{doc.content}"' for i, doc in enumerate(docs)])
        user_prompt = USER_PROMPT.format(query=query, passages=passages)
        output = self._structured_generation_model.generate(
            user_prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            output_format=SetwiseRankingOutput,
        )
        return output.selected_passage

    def heapify(self, arr: list[Document], n: int, i: int, query: str):
        """Heapify the sub-tree rooted at index i for a multi-child heap using the LLM for comparisons."""
        # Only proceed if there are children for node i.
        if self.num_child * i + 1 < n:
            # Create a list containing the parent followed by all its children.
            docs = [arr[i]] + arr[self.num_child * i + 1 : min(self.num_child * (i + 1) + 1, n)]
            # Maintain a corresponding list of indices.
            inds = [i, *list(range(self.num_child * i + 1, min(self.num_child * (i + 1) + 1, n)))]
            output = self.compare(query, docs)
            try:
                best_ind = IDENTIFIERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If the parent is not the best, swap with the best child and continue heapifying.
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapsort(self, arr: list[Document], query: str, k: int):
        """Sort the list using a modified heapsort that leverages the LLM for comparisons.

        Extraction stops once k elements have been placed in their final positions.
        """
        n = len(arr)
        ranked = 0
        # Build a max heap.
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        # Extract the largest element repeatedly.
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            self.heapify(arr, i, 0, query)

    def bubblesort(self, arr: list[Document], query: str):
        """Sort the list using a sliding-window bubblesort that leverages the LLM for comparisons.

        The method repeatedly examines windows of (num_child + 1) documents and swaps the best
        document (if not already at the start of the window).
        """
        n = len(arr)
        last_start = n - (self.num_child + 1)
        for i in range(self.top_k):
            start_ind = last_start
            end_ind = last_start + (self.num_child + 1)
            is_change = False
            while True:
                start_ind = max(start_ind, i)
                end_ind = min(end_ind, n)
                current_slice = arr[start_ind:end_ind]
                if not current_slice:
                    break
                output = self.compare(query, current_slice)
                try:
                    best_ind = IDENTIFIERS.index(output)
                except ValueError:
                    best_ind = 0
                if best_ind != 0:
                    # Swap the best document to the beginning of the window.
                    arr[start_ind], arr[start_ind + best_ind] = arr[start_ind + best_ind], arr[start_ind]
                    if not is_change:
                        is_change = True
                        # Adjust the sliding window if the best element was at the end.
                        if last_start != n - (self.num_child + 1) and best_ind == len(arr[start_ind:end_ind]) - 1:
                            last_start += len(arr[start_ind:end_ind]) - 1
                if start_ind == i:
                    break
                if not is_change:
                    last_start -= self.num_child
                start_ind -= self.num_child
                end_ind -= self.num_child

    def rerank(self, query: str, ranking: list[Document]) -> list[Document]:
        """Reorder the given list of Documents according to the selected method.

        The top_k documents (as determined by LLM comparisons) are placed first, and the rest
        are appended in their original order.
        """
        original_ranking = copy.deepcopy(ranking)

        if self.method == "heapsort":
            self.heapsort(ranking, query, self.top_k)
            # Reverse the sorted portion to have the highest ranked document first.
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            self.bubblesort(ranking, query)
        else:
            msg = f"Method {self.method} is not implemented."
            raise NotImplementedError(msg)

        # Assemble the final ranking:
        # 1. Place the top_k documents (as ordered by the sorting).
        # 2. Append documents that were not moved into the top_k, preserving original order.
        result = []
        top_doc_ids = set()
        for doc in ranking[: self.top_k]:
            top_doc_ids.add(doc.id)
            result.append(doc)
        for doc in original_ranking:
            if doc.id not in top_doc_ids:
                result.append(doc)

        return result

    @weave.op
    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        query: str,
        top_k: Optional[int] = None,
        num_permutation: Optional[int] = None,
        num_child: Optional[int] = None,
        method: Optional[str] = None,
    ) -> dict:
        """Rerank Haystack Document objects given a query using the Setwise ranking method.

        The selected method can be "heapsort" or "bubblesort".

        Returns:
            A dictionary with the key "documents" mapping to the reordered list of Document objects.
        """
        if not documents:
            return {"documents": []}

        # Update parameters if provided during runtime
        self.top_k = top_k or self.top_k
        self.num_permutation = num_permutation or self.num_permutation
        self.num_child = num_child or self.num_child
        self.method = method or self.method

        if self.top_k <= 0:
            msg = f"top_k must be > 0, but got {self.top_k}"
            raise ValueError(msg)
        if self.num_child <= 0:
            msg = f"num_child must be > 0, but got {self.num_child}"
            raise ValueError(msg)
        if self.method not in ["heapsort", "bubblesort"]:
            msg = f"Method {self.method} is not supported. Use 'heapsort' or 'bubblesort'."
            raise ValueError(msg)

        self._warm_up()

        # Work on a copy of the documents so as not to modify the original list.
        ranking = copy.deepcopy(documents)
        reordered = self.rerank(query, ranking)
        return {"documents": reordered}
