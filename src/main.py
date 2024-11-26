import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.dataset import load_and_prepare_dataset
from src.evaluation import evaluate_predictions
from src.inference import initialize_model
from src.prompt_engineering import load_prompts
from src.visualization import plot_confusion_matrix, plot_metrics


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/subset.csv",
        help="Path to save/load the subset dataset."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help="Number of samples to include in the subset."
    )
    parser.add_argument(
        "--model_paths",
        nargs='+',
        default=[
            "models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf",
            "models/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"
        ],
        help="List of model file paths to initialize."
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="./prompts/prompts.yml",
        help="Path to the prompts configuration file."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save results."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level."
    )
    return parser.parse_args()


def setup_logging(log_level):
    logging.basicConfig(level=log_level.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def main(args, logger):
    """
    Main function to execute the sentiment analysis pipeline.
    """
    try:
        logger.info("Step 1: Preparing the dataset")
        df_subset = load_and_prepare_dataset(
            output_path=args.output_path, subset_size=args.subset_size
        )

        logger.info("Step 2: Setting up the models")
        models = {}
        for model_path in args.model_paths:
            model_name = os.path.basename(model_path).split('-Instruct')[0]
            models[model_name] = initialize_model(model_path)
        logger.info(f"Models loaded: {list(models.keys())}")
        logger.info("Step 3: Loading prompts")
        prompts = load_prompts(args.prompts_path)

        logger.info("Step 4 & 5: Performing inference with prompt engineering")
        results = []

        for prompt_name, prompt_config in prompts.items():
            for model_name, model in models.items():
                logger.info(f"Running {model_name} with prompt '{prompt_name}'")
                predictions = []
                for _, row in df_subset.iterrows():
                    system_prompt = prompt_config["template"]
                    prompt = f"""Here is the movie review to analyze:<review>{row["review"]}</review>"""

                    sentiment = model.classify_sentiment(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        max_tokens=prompt_config.get("max_tokens", 100),
                        temperature=prompt_config.get("temperature", 0.7),
                        top_p=prompt_config.get("top_p", 0.9),
                        top_k=prompt_config.get("top_k", 50),
                    )
                    # Handle unexpected sentiments
                    os.makedirs(os.path.join(args.results_dir, "predictions"), exist_ok=True)
                    if sentiment not in ("Positive", "Negative"):
                        with open(
                            os.path.join(args.results_dir, "predictions", f"{model_name}_{prompt_name}.txt"),
                            "a"
                        ) as f:
                            f.write(f"Review: {row['review']}\n")
                            f.write(f"Predicted Sentiment: {sentiment}\n")
                            f.write(
                                f"Actual Sentiment: {'Positive' if row['label'] == 0 else 'Negative'}\n"
                            )
                            f.write("-" * 50 + "\n")

                    predictions.append(sentiment)

                # Collect results
                unique_predictions = np.unique(predictions)
                with open(
                    os.path.join(args.results_dir, "predictions", f"{model_name}_{prompt_name}_unique_preds.txt"),
                    "w"
                ) as f:
                    f.write(f"Unique Predictions: {unique_predictions}\n")
                    f.write(f"Unique Predictions Count: {len(unique_predictions)}\n")

                # Normalize predictions
                predictions = [
                    "Negative" if label not in ("Positive", "Negative") else label
                    for label in predictions
                ]
                results.append(
                    {
                        "Model": model_name,
                        "Prompt": f"{prompt_name}_{prompt_config.get('version', 'v1')}",
                        "Predictions": predictions,
                    }
                )

        logger.info("Step 6: Evaluating model performance")
        evaluation_results = []
        for result in results:
            metrics = evaluate_predictions(
                y_true=df_subset["label"].tolist(),
                y_pred=result["Predictions"]
            )
            metrics.update({"Model": result["Model"], "Prompt": result["Prompt"]})
            evaluation_results.append(metrics)

        metrics_df = pd.DataFrame(evaluation_results)
        logger.info(f"Performance metrics:\n{metrics_df}")
        os.makedirs(args.results_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(args.results_dir, "metrics.csv"), index=False)

        logger.info("Step 7: Analyzing results and creating visualizations")
        os.makedirs(os.path.join(args.results_dir, "performance_plots"), exist_ok=True)
        plot_metrics(metrics_df, os.path.join(args.results_dir, "performance_plots", "metrics.png"))

        # Generate confusion matrices
        for result in results:
            cm = evaluate_predictions(
                y_true=df_subset["label"].tolist(),
                y_pred=result["Predictions"],
                return_confusion_matrix=True,
            )
            os.makedirs(os.path.join(args.results_dir, "confusion_matrices"), exist_ok=True)
            plot_confusion_matrix(
                cm=cm["confusion_matrix"],
                model=result["Model"],
                prompt=result["Prompt"],
                save_path=os.path.join(
                    args.results_dir,
                    "confusion_matrices",
                    f"cm_{result['Model']}_{result['Prompt']}.png"
                ),
            )

        logger.info("Sentiment analysis pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    main(args, logger)
