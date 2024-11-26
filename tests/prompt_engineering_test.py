import os
from typing import Any, Dict

from src.prompt_engineering import load_prompts


def load_prompts_test(prompt_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Test the load_prompts function to ensure it parses the prompts.yml file correctly.
    """
    # Resolve the path to the actual YAML file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "../prompts/prompts.yml")

    # Load the prompts using the function
    prompts = load_prompts(prompt_file)

    # Assert that the result is a dictionary
    assert isinstance(
        prompts, dict
    ), "The output of load_prompts should be a dictionary"

    # Assert that the dictionary is not empty
    assert (
        len(prompts) > 0
    ), "The dictionary returned by load_prompts should not be empty"
