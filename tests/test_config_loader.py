"""Tests for configuration loading functionality."""

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from rankers.config.config_loader import load_config


class SimpleModel(BaseModel):
    """A simple Pydantic model used as a test fixture for config loading."""

    name: str
    value: int


class TestConfigLoader:
    """Tests for the load_config function in config_loader.py."""

    def test_valid_yaml_returns_correct_instance(self, tmp_path: Path) -> None:
        """Test that a valid YAML file is loaded and validated into the correct model.

        Writes a minimal YAML file matching SimpleModel, loads it via load_config,
        and verifies the returned instance has the expected field values.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"name": "hello", "value": 42}))

        result = load_config(config_file, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "hello"
        assert result.value == 42

    def test_missing_required_field_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        """Test that a YAML missing a required field raises a Pydantic ValidationError.

        SimpleModel requires both 'name' and 'value'. Omitting 'value' must trigger
        model_validate to raise ValidationError.
        """
        config_file = tmp_path / "invalid.yaml"
        # 'value' field is omitted — required by SimpleModel
        config_file.write_text(yaml.dump({"name": "hello"}))

        with pytest.raises(ValidationError):
            load_config(config_file, SimpleModel)

    def test_nonexistent_file_raises_file_not_found_error(self, tmp_path: Path) -> None:
        """Test that loading a non-existent file propagates FileNotFoundError.

        Path.read_text() raises FileNotFoundError when the file does not exist;
        load_config must not swallow this exception.
        """
        missing_file = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(missing_file, SimpleModel)
