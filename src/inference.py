import os
import logging
import argparse
from typing import Dict
from llama_cpp import Llama
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModel:
    """
    A class for an LLM for sentiment analysis.
    """

    def __init__(self, model_path: str, device: str = "mps", n_gpu_layers: int = 2):
        """
        Initializes the SentimentModel with the specified model.

        Args:
            model_path (str): path to the model.
            device (str): 'mps' for macos, 'cpu' or 'cuda' for GPU.
            n_gpu_layers (int): Number of layers to offload to GPU.
        """
        try:
            logger.info(f"Loading model from {model_path} on {device}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=32768,
                device=device,
                n_gpu_layers=n_gpu_layers if device in ("mps", "cuda") else 0,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def classify_sentiment(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> str:
        """
        Classifies the sentiment based on the provided prompt.

        Args:
            system_prompt (str): The system prompt for the model.
            prompt (str): The user prompt for the model.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            top_k (int): Top-K sampling parameter.

        Returns:
            str: The model's sentiment prediction.
        """
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "assistant", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Here is the movie review to analyze:<review>{prompt}</review>",
                    },
                ],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            sentiment = response["choices"][0]["message"]["content"].strip()
            html_tag = sentiment.find("<sentiment>")
            if html_tag != -1:
                sentiment = sentiment[
                    html_tag + len("<sentiment>") : sentiment.find("</sentiment>")
                ]

            return sentiment
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return "Error"


def get_device() -> str:
    """
    Determines the available device for inference.

    Returns:
        str: 'cuda' if a CUDA-compatible GPU is available,
             'mps' if an MPS-compatible GPU is available,
             otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def initialize_model(model_path: str) -> Dict[str, SentimentModel]:
    """
    Initializes both the 1.5B and 0.5B models.

    Args:
        model_path (str): path to where the model is stored.

    Returns:
        Dict[str, SentimentModel]: A dictionary of initialized models.
    """
    device = get_device()
    n_gpu_layers = 2 if device in ("mps", "cuda") else 0

    model = SentimentModel(
        model_path=model_path,
        device=device,
        n_gpu_layers=n_gpu_layers,
    )

    return model


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis using Llama-based Models"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf",
        help="Model version, e.g. quantized.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=True,
        help="System prompt for sentiment analysis.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt for sentiment analysis.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-K sampling parameter.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    model = initialize_model(args.model_path)

    if not model:
        logger.error(f"Model was not load, check params:\n {args}.")
        return

    sentiment = model.classify_sentiment(
        system_prompt=args.system_prompt,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    logger.info(f"Sentiment: {sentiment}")
    print(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
