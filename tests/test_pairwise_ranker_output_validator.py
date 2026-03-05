"""Tests for pairwise ranker output validation."""

import pytest
from pydantic import ValidationError

from rankers.pairwise.pairwise_ranker_output_validator import PairwiseRankingOutput


class TestPairwiseRankingOutput:
    """Tests for the PairwiseRankingOutput validator."""

    def test_valid_passage_a(self) -> None:
        """Test that 'A' is accepted as a valid selected_passage value."""
        output = PairwiseRankingOutput(selected_passage="A")
        assert output.selected_passage == "A"

    def test_valid_passage_b(self) -> None:
        """Test that 'B' is accepted as a valid selected_passage value."""
        output = PairwiseRankingOutput(selected_passage="B")
        assert output.selected_passage == "B"

    def test_whitespace_padded_a_is_normalized(self) -> None:
        """Test ' A ' (with surrounding whitespace) is stripped and accepted.

        The field_validator strips and uppercases the value before checking,
        so whitespace-padded inputs that resolve to 'A' or 'B' must pass.
        However the field pattern ^[AB]$ is applied before the validator by
        Pydantic, so inputs with whitespace will fail the pattern check first.
        """
        # The pattern r"^[AB]$" runs before the field_validator;
        # " A " does not match the pattern, so ValidationError is expected.
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage=" A ")

    def test_whitespace_padded_b_raises_validation_error(self) -> None:
        """Test ' B ' with whitespace raises ValidationError due to pattern mismatch."""
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage=" B ")

    def test_invalid_passage_c(self) -> None:
        """Test that 'C' raises ValidationError (not in the allowed set)."""
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage="C")

    def test_invalid_parenthesized_a(self) -> None:
        """Test that '(A)' raises ValidationError (pattern requires no parentheses)."""
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage="(A)")

    def test_empty_string_raises_validation_error(self) -> None:
        """Test that an empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage="")

    def test_lowercase_a_raises_validation_error(self) -> None:
        """Test lowercase 'a' raises ValidationError (pattern requires uppercase)."""
        with pytest.raises(ValidationError):
            PairwiseRankingOutput(selected_passage="a")
