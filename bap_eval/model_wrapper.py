import os


class BaseModelWrapper:
    """
    Abstract base class for all model wrappers.

    Subclasses must implement the `generate` method.
    """

    def generate(self, prompt: str) -> str:
        """
        Generate text for a given prompt.

        Args:
            prompt (str): Input prompt string.

        Returns:
            str: Generated text response.
        """
        raise NotImplementedError


class HuggingFaceModelWrapper(BaseModelWrapper):
    """
    Wrapper for Hugging Face models using `transformers`.

    Args:
        model_name (str): Hugging Face model name (e.g., "mistral-7b").
        device (str, optional): Device to load the model on ("cpu" or "cuda").
    """

    def __init__(self, model_name: str, device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate(self, prompt: str) -> str:
        """
        Generate text response from the Hugging Face model.

        Args:
            prompt (str): Input prompt.

        Returns:
            str: Model-generated response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class OpenAIModelWrapper(BaseModelWrapper):
    """
    Wrapper for OpenAI models via API.

    Args:
        model_name (str): OpenAI model name (e.g., "gpt-4.1").
        api_key (str): API key for your model.
    """

    def __init__(self, model_name: str, api_key: str = None):
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        """
        Generate text response from an OpenAI model.

        Args:
            prompt (str): Input prompt.

        Returns:
            str: Model-generated response.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class OllamaModelWrapper(BaseModelWrapper):
    """
    Wrapper for locally running Ollama models.

    Args:
        model_name (str): Ollama model name (e.g., "llama3").
        base_url (str, optional): Base URL for the Ollama API. Defaults to "http://localhost:11434".
    """

    def __init__(self, model_name: str, base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """
        Generate text response from an Ollama model.

        Args:
            prompt (str): Input prompt.

        Returns:
            str: Model-generated response.
        """
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]
