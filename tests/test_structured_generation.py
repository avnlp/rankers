"""Tests for the StructuredGeneration utility class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from rankers.utils.structured_generation import (
    DEFAULT_SYSTEM_PROMPT,
    StructuredGeneration,
)


class MockModel:
    """Mock transformer model for testing."""

    pass


class SampleOutputSchema(BaseModel):
    """Test output schema for validation."""

    answer: str
    confidence: float


class CustomSchema(BaseModel):
    """Test custom output schema for validation."""

    summary: str


class TestStructuredGenerationInitialization:
    """Test class for StructuredGeneration initialization."""

    @pytest.fixture
    def mock_transformers(self):
        """Fixture for mocking transformer components."""
        tokenizer_mock = MagicMock()
        tokenizer_mock.apply_chat_template.side_effect = lambda messages, **kwargs: (
            " ".join([f"{m['role']}: {m['content']}" for m in messages])
        )
        tokenizer_mock.get_vocab.return_value = {"dummy": 0}
        hf_model_mock = MagicMock()

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer_mock,
            ) as mock_tokenizer,
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=hf_model_mock,
            ) as mock_hf_model,
            patch.object(
                PreTrainedModel, "from_pretrained", return_value=hf_model_mock
            ),
            patch.object(
                PreTrainedTokenizer, "from_pretrained", return_value=tokenizer_mock
            ),
            patch(
                "rankers.utils.structured_generation.from_transformers",
                return_value=MockModel(),
            ) as mock_from_transformers,
        ):
            yield mock_tokenizer, mock_hf_model, mock_from_transformers

    @pytest.fixture
    def structured_gen(self, mock_transformers):
        """Fixture for creating a basic StructuredGeneration instance."""
        return StructuredGeneration(
            model_name="test-model",
            device="cpu",
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"use_fast": True},
        )

    @pytest.fixture
    def mock_generator(self):
        """Fixture for mocking the JSON generator."""
        with patch("rankers.utils.structured_generation.Generator") as mock_gen:
            mock_instance = Mock()
            mock_instance.return_value = SampleOutputSchema(
                answer="Test answer", confidence=0.9
            )
            mock_gen.return_value = mock_instance
            yield mock_gen

    def test_default_initialization(self, mock_transformers):
        """Test initialization with minimal required parameters."""
        sg = StructuredGeneration(model_name="test-model")

        assert sg.model_name == "test-model"
        assert sg.device is None
        assert sg.model_kwargs == {}
        assert sg.tokenizer_kwargs == {}

    def test_full_initialization(self, mock_transformers):
        """Test initialization with all optional parameters."""
        sg = StructuredGeneration(
            model_name="test-model",
            device="cuda",
            model_kwargs={"key": "value"},
            tokenizer_kwargs={"another_key": "another_value"},
            model_class=PreTrainedModel,
            tokenizer_class=PreTrainedTokenizer,
        )

        assert sg.device == "cuda"
        assert sg.model_kwargs == {"key": "value"}
        assert sg.tokenizer_kwargs == {"another_key": "another_value"}
        assert sg.model_class == PreTrainedModel
        assert sg.tokenizer_class == PreTrainedTokenizer

    def test_model_initialization_calls(self, mock_transformers):
        """Verify model and tokenizer initialization calls."""
        mock_tokenizer, mock_hf_model, mock_from_transformers = mock_transformers

        StructuredGeneration(model_name="test-model")

        mock_tokenizer.assert_called_once_with("test-model")
        mock_hf_model.assert_called_once_with("test-model")
        mock_from_transformers.assert_called_once_with(
            mock_hf_model.return_value, mock_tokenizer.return_value
        )

    def test_valid_prompt_creation(self, structured_gen):
        """Test prompt creation with valid inputs."""
        user_prompt = "What's the weather?"
        system_prompt = "You are a meteorologist."

        result = structured_gen._prepare_prompt(user_prompt, system_prompt)

        assert "system: You are a meteorologist." in result
        assert "user: What's the weather?" in result

    def test_missing_user_prompt(self, structured_gen):
        """Test missing user prompt raises error."""
        with pytest.raises(ValueError) as exc_info:
            structured_gen._prepare_prompt(None, "System prompt")

        assert "User prompt must be provided." in str(exc_info.value)

    def test_default_system_prompt(self, structured_gen):
        """Test automatic use of default system prompt."""
        user_prompt = "Test question"

        result = structured_gen._prepare_prompt(user_prompt, None)

        assert DEFAULT_SYSTEM_PROMPT in result

    def test_empty_user_prompt(self, structured_gen):
        """Test empty string user prompt raises error."""
        with pytest.raises(ValueError) as exc_info:
            structured_gen._prepare_prompt("", "System prompt")

        assert "User prompt must be provided." in str(exc_info.value)

    def test_successful_generation(self, structured_gen, mock_generator):
        """Test successful generation with valid inputs."""
        result = structured_gen.generate(
            output_format=SampleOutputSchema, user_prompt="Test", system_prompt="Test"
        )

        assert isinstance(result, SampleOutputSchema)
        mock_generator.assert_called_once_with(structured_gen.model, SampleOutputSchema)

    def test_missing_prompts(self, structured_gen):
        """Test error handling for missing prompts."""
        with pytest.raises(ValueError):
            structured_gen.generate(output_format=SampleOutputSchema, user_prompt=None)

    def test_custom_output_schema(self, structured_gen, mock_generator):
        """Test generation with different output schemas."""
        mock_generator.return_value.return_value = CustomSchema(summary="Test")

        result = structured_gen.generate(
            output_format=CustomSchema, user_prompt="Test", system_prompt="Test"
        )

        assert isinstance(result, CustomSchema)
        assert result.summary == "Test"

    def test_generator_configuration(self, structured_gen, mock_generator):
        """Verify generator is configured with correct model and schema."""
        structured_gen.generate(
            output_format=SampleOutputSchema, user_prompt="Test", system_prompt="Test"
        )

        mock_generator.assert_called_once_with(structured_gen.model, SampleOutputSchema)
