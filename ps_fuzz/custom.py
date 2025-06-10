from transformers import pipeline
from loguru import logger

# Sample custom LLM integration using a lightweight model

MODEL_NAME = "google/flan-t5-small"

def initialize_client(model_name: str):
    """Initialize a text2text-generation pipeline for instruction-tuned models."""
    return pipeline("text2text-generation", model=MODEL_NAME)


def generate(client, prompt: str):
    """Generate a response using the provided pipeline."""
    result = client(prompt, max_new_tokens=100)
    generated_text = result[0]["generated_text"]
    response = generated_text.strip()
    logger.debug(f"Prompt: {prompt}")
    logger.debug(f"Generated text: {response}")
    return response
