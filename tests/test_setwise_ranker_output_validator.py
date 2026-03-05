"""Tests for setwise ranker output validation."""

import pytest
from pydantic import ValidationError

from rankers.setwise.setwise_ranker_output_validator import (
    IDENTIFIERS,
    SetwiseRankingOutput,
)


class TestSetwiseRankingOutput:
    """Tests for SetwiseRankingOutput validator and IDENTIFIERS list."""

    def test_identifiers_length(self) -> None:
        """Test IDENTIFIERS contains exactly 260 entries (26 letters × 10 digits)."""
        assert len(IDENTIFIERS) == 260

    def test_identifiers_first_entry(self) -> None:
        """Test first entry in IDENTIFIERS is '(A0)'."""
        assert IDENTIFIERS[0] == "(A0)"

    def test_identifiers_last_entry(self) -> None:
        """Test last entry in IDENTIFIERS is '(Z9)'."""
        assert IDENTIFIERS[-1] == "(Z9)"

    def test_valid_identifier_a0(self) -> None:
        """Test '(A0)' is accepted as a valid selected_passage value."""
        output = SetwiseRankingOutput(selected_passage="(A0)")
        assert output.selected_passage == "(A0)"

    def test_valid_identifier_z9(self) -> None:
        """Test '(Z9)' is accepted as a valid selected_passage value."""
        output = SetwiseRankingOutput(selected_passage="(Z9)")
        assert output.selected_passage == "(Z9)"

    def test_valid_identifier_m5(self) -> None:
        """Test '(M5)' is a valid identifier and is accepted."""
        output = SetwiseRankingOutput(selected_passage="(M5)")
        assert output.selected_passage == "(M5)"

    def test_invalid_no_parentheses(self) -> None:
        """Test 'A0' without parentheses raises ValidationError."""
        with pytest.raises(ValidationError):
            SetwiseRankingOutput(selected_passage="A0")

    def test_invalid_lowercase_letter(self) -> None:
        """Test '(a0)' with lowercase letter raises ValidationError."""
        with pytest.raises(ValidationError):
            SetwiseRankingOutput(selected_passage="(a0)")

    def test_invalid_missing_digit(self) -> None:
        """Test '(A)' without digit raises ValidationError."""
        with pytest.raises(ValidationError):
            SetwiseRankingOutput(selected_passage="(A)")

    def test_invalid_empty_string(self) -> None:
        """Test empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            SetwiseRankingOutput(selected_passage="")
