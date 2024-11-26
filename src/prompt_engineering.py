import logging
from typing import Dict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts(prompt_file: str) -> Dict[str, str]:
    """
    Loads prompt templates from a YAML file.

    Args:
        prompt_file (str): Path to the YAML file containing prompts.

    Returns:
        Dict[str, str]: A dictionary mapping prompt names to their templates.
    """
    try:
        logger.info(f"Loading prompts from {prompt_file}")
        with open(prompt_file, "r") as file:
            prompts_yaml = yaml.safe_load(file)

        prompts = {}
        for key, value in prompts_yaml.get("prompts", {}).items():
            prompts[key] = {
                "version": value.get("version"),
                "description": value.get("description"),
                "template": value.get("template"),
                "temperature": value.get("model_settings", {}).get("temperature"),
                "top_p": value.get("model_settings", {}).get("top_p"),
                "top_k": value.get("model_settings", {}).get("top_k"),
                "max_tokens": value.get("model_settings", {}).get("max_tokens"),
            }

        return prompts

    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
