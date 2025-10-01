# bap-eval

A Python framework for evaluating Large Language Models (LLMs) on **Bias**, **Accuracy**, and **Politeness** metrics, with a composite PEI (Performance Evaluation Index) score.

## Features
- Evaluate LLMs using a unified interface for Hugging Face, OpenAI, and Ollama models.
- Compute bias, accuracy, and politeness scores based on a set of default prompts.
- Display progress feedback during evaluation using a progress bar.
- Extensible architecture for adding new model wrappers and evaluation metrics.

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama server running locally (for Ollama models)
- Optional: GPU with CUDA support (for Hugging Face models)

### Install via pip
1. Clone the repository (if using source):
   ```bash
   git clone https://github.com/your-org/bap-eval.git
   cd bap-eval
   ```
2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

This installs required dependencies: `transformers`, `torch`, `openai`, `requests`, and `tqdm`.

## Usage

The `bap-eval` framework evaluates LLMs using a `run_bap_test` function that takes a model wrapper and returns a dictionary with `bias`, `accuracy`, `politeness`, and `PEI` scores.

### Example: Evaluating an Ollama Model
```python
from bap_eval.runner import run_bap_test
from bap_eval.model_wrapper import OllamaModelWrapper

# Initialize the wrapper
wrapper = OllamaModelWrapper("llama3.2")

# Run evaluation with progress feedback
result = run_bap_test(wrapper, verbose=True)
print(result)
```

**Note**: Ensure the Ollama server is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`).

### Example: Evaluating a Hugging Face Model
```python
from bap_eval.runner import run_bap_test
from bap_eval.model_wrapper import HuggingFaceModelWrapper

wrapper = HuggingFaceModelWrapper("mistralai/Mistral-7B-v0.1", device="cpu")
result = run_bap_test(wrapper)
print(result)
```

### Example: Evaluating an OpenAI Model
```python
from bap_eval.runner import run_bap_test
from bap_eval.model_wrapper import OpenAIModelWrapper

wrapper = OpenAIModelWrapper("gpt-4o", api_key="your-openai-api-key")
result = run_bap_test(wrapper)
print(result)
```

### Output
The evaluation returns a dictionary:
```json
{
    "bias": 0.85,
    "accuracy": 0.92,
    "politeness": 0.78,
    "PEI": 0.85
}
```

## Project Structure
```
bap_eval/
├── pyproject.toml
├── README.md
├── LICENSE
└── bap_eval/
    ├── __init__.py
    ├── runner.py
    ├── model_wrapper.py
    ├── scoring.py
    ├── datasets/
    │   ├── __init__.py
    │   └── default_prompts.py
    └── metrics/
        ├── __init__.py
        ├── bias.py
        ├── accuracy.py
        └── politeness.py
```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, please open an issue on the [GitHub repository](https://github.com/Star2578/bap-eval).