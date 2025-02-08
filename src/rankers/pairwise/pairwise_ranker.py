from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Optional

import weave
from haystack import Document, component
from transformers import PreTrainedModel, PreTrainedTokenizer

from rankers.pairwise.pairwise_ranker_output_validator import PairwiseRankingOutput
from rankers.pairwise.pairwise_ranking_prompt import SYSTEM_PROMPT, USER_PROMPT
from rankers.utils import StructuredGeneration


@component
class PairwiseLLMRanker:
    """Pairwise LLM Ranker using structured generation for document comparisons.

    This class implements a pairwise ranking algorithm using a Large Language Model (LLM)
    to compare documents in pairs and determine their relevance ranking for a given query.

    Attributes:
        model_name: The name of the LLM model to use.
        device: The device to run the model on (e.g., 'cpu', 'cuda').
        model_kwargs: Additional keyword arguments for model initialization.
        tokenizer_kwargs: Additional keyword arguments for tokenizer initialization.
        model_class: The class of the model to use (optional).
        tokenizer_class: The class of the tokenizer to use (optional).
        method: The ranking method to use (one of 'allpair', 'heapsort', 'bubblesort').
        top_k: The number of top documents to return.
        _generator: The StructuredGeneration instance used for generation.
        _is_warmed_up: A flag indicating whether the generator is initialized.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_class: Optional[type[PreTrainedModel]] = None,
        tokenizer_class: Optional[type[PreTrainedTokenizer]] = None,
        method: str = "allpair",
        top_k: int = 10,
    ):
        """Initialize the PairwiseLLMRanker.

        Args:
            model_name: The name of the LLM model to use.
            device: The device to run the model on. Defaults to None.
            model_kwargs: Additional keyword arguments for model initialization.
                Defaults to None.
            tokenizer_kwargs: Additional keyword arguments for tokenizer initialization.
                Defaults to None.
            model_class: The class of the model to use. Defaults to None.
            tokenizer_class: The class of the tokenizer to use. Defaults to None.
            method: The ranking method to use. One of 'allpair', 'heapsort', 'bubblesort'.
                Defaults to 'allpair'.
            top_k: The number of top documents to return. Defaults to 10.
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.method = method
        self.top_k = top_k
        self._generator = None
        self._is_warmed_up = False

    def _warm_up(self) -> None:
        """Initialize the StructuredGeneration instance if not already initialized."""
        if not self._is_warmed_up:
            self._generator = StructuredGeneration(
                model_name=self.model_name,
                device=self.device,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                model_class=self.model_class,
                tokenizer_class=self.tokenizer_class,
            )
            self._is_warmed_up = True

    def _compare_pair(self, query: str, doc_a: Document, doc_b: Document) -> str:
        """Compare two documents by generating text using the LLM and return the result.

        Args:
            query: The query string to compare the documents against.
            doc_a: The first document to compare.
            doc_b: The second document to compare.

        Returns:
            str: 'A' if doc_a is more relevant, 'B' if doc_b is more relevant, or 'tie' if they are equal.

        Raises:
            ValueError: If the result is not one of 'A', 'B', or 'tie'.
        """
        # Original order comparison
        prompt_ab = USER_PROMPT.format(query=query, doc_a=doc_a.content, doc_b=doc_b.content)
        result_ab = self._generator.generate(
            PairwiseRankingOutput, user_prompt=prompt_ab, system_prompt=SYSTEM_PROMPT
        ).selected_passage.upper()

        # Reverse order comparison
        prompt_ba = USER_PROMPT.format(query=query, doc_a=doc_b.content, doc_b=doc_a.content)
        result_ba = self._generator.generate(
            PairwiseRankingOutput, user_prompt=prompt_ba, system_prompt=SYSTEM_PROMPT
        ).selected_passage.upper()

        # Resolve conflicts
        if result_ab == "A" and result_ba == "B":
            return "A"
        elif result_ab == "B" and result_ba == "A":
            return "B"
        return "tie"  # Handle ties and invalid responses

    def _allpair_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Re-rank documents using the all-pairs comparison method.

        Args:
            query: The query string used for comparison.
            docs: The list of documents to re-rank.

        Returns:
            List[Document]: The top_k documents sorted by their pairwise comparison scores.

        Note:
            This method compares all possible pairs of documents and computes a score for each document
            based on the number of times it is selected as more relevant than another document.
        """
        scores = defaultdict(float)
        for doc1, doc2 in combinations(docs, 2):
            result = self._compare_pair(query, doc1, doc2)
            if result == "A":
                scores[doc1.id] += 1
            elif result == "B":
                scores[doc2.id] += 1
            else:
                scores[doc1.id] += 0.5
                scores[doc2.id] += 0.5

        return sorted(docs, key=lambda d: scores[d.id], reverse=True)[: self.top_k]

    def _heapify(self, array: list[Document], query: str, root: int, size: int) -> None:
        """Maintains the max-heap property by ensuring the parent node is larger than its children.

        This function is used in the heap-sort algorithm to re-rank documents based on pairwise
        comparisons. It starts from a given root index and ensures that the subtree rooted at this
        index satisfies the max-heap property. If a child node is found to be larger than the parent,
        they are swapped, and the function is called recursively on the affected subtree.

        Args:
            array: The list of documents being re-ranked.
            query: The query string used for document comparison.
            root: The index of the root node to start the heapify process from.
            size: The size of the heap.

        Returns:
            None
        """
        current_largest = root
        left_child = 2 * root + 1
        right_child = 2 * root + 2

        if left_child < size:
            comparison_result = self._compare_pair(query, array[left_child], array[current_largest])
            if comparison_result == "A":
                current_largest = left_child

        if right_child < size:
            comparison_result = self._compare_pair(query, array[right_child], array[current_largest])
            if comparison_result == "A":
                current_largest = right_child

        if current_largest != root:
            array[root], array[current_largest] = array[current_largest], array[root]
            self._heapify(array, query, current_largest, size)

    def _heapsort_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Re-rank documents using a heap-sort inspired pairwise comparison method.

        Args:
            query: The query string used for comparison.
            docs: The list of documents to re-rank.

        Returns:
            List[Document]: The top_k documents sorted using a heap-based approach.
        """
        array = deepcopy(docs)
        n = len(array)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(array, query, i, n)
        for i in range(n - 1, 0, -1):
            array[i], array[0] = array[0], array[i]
            self._heapify(array, query, 0, i)
        return list(reversed(array))[: self.top_k]

    def _bubblesort_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """Re-rank documents using an optimized bubble-sort inspired pairwise comparison method.

        Args:
            query: The query string used for comparison.
            docs: The list of documents to re-rank.

        Returns:
            List[Document]: The top_k documents sorted using an optimized bubble-sort approach.
        """
        ranking = deepcopy(docs)
        n = len(ranking)
        k = min(self.top_k, n)
        last_end = n - 1  # Track the end of the unsorted portion

        for i in range(k):
            current_ind = last_end
            is_change = False  # Flag to check if any swaps occurred

            while True:
                if current_ind <= i:
                    break  # Exit if reached the sorted portion

                doc1 = ranking[current_ind]
                doc2 = ranking[current_ind - 1]
                result = self._compare_pair(query, doc1, doc2)

                if result == "B":  # If doc2 should come after doc1
                    ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]
                    if not is_change:
                        is_change = True
                        if last_end != n - 1:  # Skip unchanged pairs at the bottom
                            last_end += 1

                if not is_change:
                    last_end -= 1
                current_ind -= 1

            if not is_change:
                break  # Early termination if no swaps occurred

        return ranking[:k]

    @weave.op
    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        query: str,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, list[Document]]:
        """Execute the pairwise ranking using the specified method.

        Args:
            documents: The list of documents to rank.
            query: The query string used for ranking.
            method: The ranking method to use. One of 'allpair', 'heapsort', 'bubblesort'.
                Defaults to the class's method.
            top_k: The number of top documents to return. Defaults to the class's top_k.

        Returns:
            Dict[str, List[Document]]: A dictionary containing the ranked documents.

        Raises:
            ValueError: If the specified method is not supported.
            ValueError: If documents is empty or top_k is less than or equal to 0.
        """
        self._warm_up()
        self.method = method or self.method
        self.top_k = top_k or self.top_k

        if self.method not in ["allpair", "heapsort", "bubblesort"]:
            msg = f"Unsupported method: {self.method}"
            raise ValueError(msg)

        if not documents or self.top_k <= 0:
            return {"documents": []}

        working_docs = deepcopy(documents)

        if self.method == "allpair":
            sorted_docs = self._allpair_rerank(query, working_docs)
        elif self.method == "heapsort":
            sorted_docs = self._heapsort_rerank(query, working_docs)
        elif self.method == "bubblesort":
            sorted_docs = self._bubblesort_rerank(query, working_docs)

        # Preserve original order for non-top_k docs
        top_ids = {doc.id for doc in sorted_docs}
        remaining = [doc for doc in documents if doc.id not in top_ids]

        return {"documents": sorted_docs + remaining}
