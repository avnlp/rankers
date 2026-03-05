"""Setwise ranking output validation model."""

import string

from pydantic import BaseModel, Field


# Generate all possible valid identifiers
# A0, A2, ... B1, B2, B3 ... Z9
IDENTIFIERS: list[str] = [
    f"({alphabet}{digit})"  # Add parentheses around the identifier
    for alphabet in string.ascii_uppercase
    for digit in string.digits
]


class SetwiseRankingOutput(BaseModel):
    """Validated output for passage ranking selection using alphanumeric identifiers.

    Validates that the selected passage identifier matches the expected format
    (uppercase letter followed by digit in parentheses) and exists in the
    predefined list of valid identifiers.

    Attributes:
        selected_passage: The label of the most relevant passage in '(A0)'-'(Z9)'
            format. Must be an uppercase letter followed by a single digit,
            enclosed in parentheses.

    Example:
        >>> SetwiseRankingOutput(selected_passage="(B4)")
        SetwiseRankingOutput(selected_passage='(B4)')
    """

    selected_passage: str = Field(
        ...,
        pattern=r"^\([A-Z]\d\)$",
        description=(
            "The label of the most relevant passage formatted as an uppercase letter "
            "followed by a digit, enclosed in parentheses (e.g., '(A0)', '(B3)'). "
            "Must match one of the identifiers provided in the original passage list."
        ),
        examples=["(A0)", "(B3)", "(Z9)"],
    )
