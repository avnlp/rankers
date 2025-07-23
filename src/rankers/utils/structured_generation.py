from outlines import generate, models
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class StructuredGeneration:
    """A class for structured text generation using transformer models.

    This class facilitates structured generation of text outputs constrained by
    a predefined schema using the Outlines library and Transformer models.

    Attributes:
        model_name: Identifier for the pretrained transformer model.
        device: Hardware device for model execution (e.g., 'cuda', 'cpu').
        model_kwargs: Additional keyword arguments for model initialization.
        tokenizer_kwargs: Additional keyword arguments for tokenizer initialization.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        model_class: type[PreTrainedModel] | None = None,
        tokenizer_class: type[PreTrainedTokenizer] | None = None,
    ) -> None:
        """Initialize the StructuredGeneration with model and tokenizer configuration.

        Args:
            model_name: Identifier for the pretrained transformer model.
            device: Hardware device to load the model on (e.g., 'cuda', 'cpu').
            model_kwargs: Additional parameters for model initialization.
            tokenizer_kwargs: Additional parameters for tokenizer initialization.
            model_class: Custom model class to override default model loading.
            tokenizer_class: Custom tokenizer class to override default tokenizer.
        """
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = models.transformers(
            model_name=self.model_name,
            device=self.device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            model_class=self.model_class,
            tokenizer_class=self.tokenizer_class,
        )

    def _prepare_prompt(self, user_prompt: str | None, system_prompt: str | None) -> str:
        """Construct a formatted prompt using the tokenizer's chat template.

        Args:
            user_prompt: User's input query/message.
            system_prompt: System-level instructions/context.

        Returns:
            Formatted prompt string ready for model input.

        Raises:
            ValueError: If either prompt argument is None or empty.
        """
        if not user_prompt:
            msg = "User prompt must be provided."
            raise ValueError(msg)

        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def generate(
        self,
        output_format: type[BaseModel],
        user_prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Generate structured output according to a specified schema.

        Args:
            output_format: Pydantic model defining the desired output structure.
            user_prompt: User's input query/message (required).
            system_prompt: System-level instructions/context (required).

        Returns:
            Instance of the output_format model populated with generated content.

        Raises:
            ValueError: If either prompt argument is None or empty.
        """
        prompt = self._prepare_prompt(user_prompt, system_prompt)
        generator = generate.json(self.model, output_format)
        result = generator(prompt)
        return result
