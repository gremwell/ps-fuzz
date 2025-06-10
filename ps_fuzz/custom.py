from transformers import pipeline
from loguru import logger

# Sample custom LLM integration using a lightweight model

MODEL_NAME = "google/flan-t5-small"


def validate_api_keys():
    """Custom models typically run locally and do not require API keys."""
    return True


def initialize_client(model_name: str):
    """Initialize a text2text-generation pipeline for instruction-tuned models."""
    return pipeline("text2text-generation", model=MODEL_NAME)


def test_prompt(client, model: str, system_prompt: str, user_prompt: str):
    """Generate a response using the provided pipeline."""
    try:
        prompt = f"{system_prompt}\n{user_prompt}" if system_prompt else user_prompt
        result = client(prompt, max_new_tokens=100)
        generated_text = result[0]["generated_text"]
        response = generated_text.strip()
        logger.info(f"User Prompt: {user_prompt}")
        logger.debug(f"Generated text: {response}")
        return response, False
    except Exception as exc:
        return f"Error: {exc}", True


def validate_model(model_name: str) -> bool:
    """Check whether the model can be loaded."""
    try:
        pipeline("text-generation", model=MODEL_NAME)
        return True
    except Exception as exc:
        print(f"Error loading model {MODEL_NAME}: {exc}")
        return False
