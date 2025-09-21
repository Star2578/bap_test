from bap_eval import run_bap_test, HuggingFaceModelWrapper, OpenAIModelWrapper, OllamaModelWrapper

# Example 1: Hugging Face
wrapper = HuggingFaceModelWrapper("mistral-7b")
print(run_bap_test(wrapper))

# Example 2: OpenAI
wrapper = OpenAIModelWrapper("gpt-4.1")
print(run_bap_test(wrapper))

# Example 3: Ollama
wrapper = OllamaModelWrapper("llama3.2")
print(run_bap_test(wrapper))
