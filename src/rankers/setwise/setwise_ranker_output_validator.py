import string

from pydantic import BaseModel, Field, ValidationInfo, field_validator

# Generate all possible valid identifiers
# A0, A2, ... B1, B2, B3 ... Z9
IDENTIFIERS: list[str] = [
    f"({alphabet}{digit})"  # Add parentheses around the identifier
    for alphabet in string.ascii_uppercase
    for digit in string.digits
]


class SetwiseRankingOutput(BaseModel):
    """Represents validated output for passage ranking selection using alphanumeric identifiers in parentheses.

    Validates that the selected passage identifier matches the expected format (uppercase letter
    followed by digit in parentheses) and exists in the predefined list of valid identifiers.

    Attributes:
        selected_passage: The label of the most relevant passage in '(A0)'-'(Z9)' format.
            Must be an uppercase letter followed by a single digit, enclosed in parentheses.

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

    @field_validator("selected_passage")
    @classmethod
    def validate_identifier(cls, identifier: str, values: ValidationInfo) -> str:
        """Validate that the selected passage identifier matches the required format.

        Args:
            identifier: The candidate passage identifier to validate.
            values: Validation context containing other field values.

        Returns:
            The validated identifier if it passes all checks.

        Raises:
            ValueError: If the identifier doesn't match the required format or isn't in the
                valid identifiers list

        Example:
            Valid identifier:
            >>> SetwiseRankingOutput(selected_passage="(B4)")

            Invalid identifier:
            >>> SetwiseRankingOutput(selected_passage="(B)")
            ValueError: Invalid passage identifier format: '(B)'. Must be uppercase letter followed by digit in parentheses (e.g., '(A0)', '(B3)').
        """
        if identifier not in IDENTIFIERS:
            msg = (
                f"Invalid passage identifier format: {identifier}. "
                "Must be uppercase letter followed by digit in parentheses (e.g., '(A0)', '(B3)')."
            )
            raise ValueError(msg)
        return identifier
