"""Pairwise ranking output validation model."""

from pydantic import BaseModel, Field


class PairwiseRankingOutput(BaseModel):
    """Represents validated output for pairwise document comparison selection.

    Validates that the selected passage identifier is either 'A' or 'B' in uppercase.
    Ensures the LLM's response matches the required format for reliable ranking.

    Attributes:
        selected_passage: The label of the more relevant passage. Must be uppercase
            'A' or 'B' without parentheses.

    Example:
        >>> PairwiseRankingOutput(selected_passage="A")
        PairwiseRankingOutput(selected_passage='A')

    Raises:
        ValueError: If selected_passage is not 'A' or 'B' in uppercase
    """

    selected_passage: str = Field(
        ...,
        pattern=r"^[AB]$",
        description=(
            "Identifier of the more relevant passage. Must be uppercase 'A' or 'B' "
            "without additional formatting. Example: 'A' for first passage, 'B' for second."
        ),
        examples=["A", "B"],
    )
