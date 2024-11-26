"""
Main Execution Script
---------------------
This script orchestrates the sentiment analysis pipeline, including dataset preparation,
model inference, prompt engineering, evaluation, and visualization.
"""

import logging
import pandas as pd
import numpy as np
import os
from src.dataset import load_and_prepare_dataset
from src.inference import initialize_model, SentimentModel
from src.prompt_engineering import load_prompts
from src.evaluation import evaluate_predictions
from src.visualization import plot_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to execute the sentiment analysis pipeline.
    """
    try:
        logger.info("Step 1: Preparing the dataset")
        df_subset = load_and_prepare_dataset(
            output_path="../data/subset.csv", subset_size=10
        )

        logger.info("Step 2: Setting up the models")
        models = {
            "Qwen2.5-0.5B": initialize_model(
                "models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf"
            ),
            "Qwen2.5-1.5B": initialize_model(
                "models/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"
            ),
        }

        logger.info("Step 3: Loading prompts")
        prompts = load_prompts("./prompts/prompts.yml")

        logger.info("Step 4 & 5: Performing inference with prompt engineering")
        results = []

        for prompt_name, prompt_config in prompts.items():
            for model_name, model in models.items():
                logger.info(f"Running {model_name} with prompt '{prompt_name}'")
                predictions = []
                for _, row in df_subset.iterrows():
                    system_prompt = prompt_config["template"]
                    prompt = f"Here is the movie review to analyze:<review>{row["review"]}</review>"

                    # print(f"System Prompt: {system_prompt}")
                    # print(f"Prompt: {prompt}")

                    sentiment = model.classify_sentiment(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        max_tokens=prompt_config["max_tokens"],
                        temperature=prompt_config["temperature"],
                        top_p=prompt_config["top_p"],
                        top_k=prompt_config["top_k"],
                    )
                    # Despite instructions, models can return strange answers, so I will save the predictions in a file for further analysis.
                    os.makedirs("./results/predictions", exist_ok=True)
                    if sentiment not in ("Positive", "Negative"):
                        with open(
                            f"./results/predictions/{model_name}_{prompt_name}.txt", "a"
                        ) as f:
                            f.write(f"Review: {row['review']}\n")
                            f.write(f"Pred   Sentiment: {sentiment}\n")
                            f.write(
                                f"Actual Sentiment: {'Positive' if row['label'] == 0 else 'Negative'}\n"
                            )
                            f.write("-" * 50)
                            f.write("\n")

                    predictions.append(sentiment)

                # Collect results
                # TODO: If the model predicts a label other than "Positive" or "Negative", we will keep Negative label. Makes sense to add Unknown label or something similar.
                unique_predictions = np.unique(predictions)
                with open(
                    f"./results/predictions/{model_name}_{prompt_name}_unique_preds.txt",
                    "w",
                ) as f:
                    f.write(f"Unique Predictions: {unique_predictions}\n")
                    f.write(f"Unique Predictions Count: {len(unique_predictions)}\n")

                predictions = [
                    "Negative" if label not in ("Positive", "Negative") else label
                    for label in predictions
                ]
                # print(f"Predictions: {np.unique(predictions)}")
                results.append(
                    {
                        "Model": model_name,
                        "Prompt": prompt_name + "_" + prompt_config["version"],
                        "Predictions": predictions,
                    }
                )

        logger.info("Step 6: Evaluating model performance")
        evaluation_results = []
        for result in results:
            metrics = evaluate_predictions(
                y_true=df_subset["label"].tolist(), y_pred=result["Predictions"]
            )
            metrics.update({"Model": result["Model"], "Prompt": result["Prompt"]})
            evaluation_results.append(metrics)

        metrics_df = pd.DataFrame(evaluation_results)
        logger.info(f"Performance metrics:\n{metrics_df}")
        os.makedirs("./results", exist_ok=True)
        metrics_df.to_csv("./results/metrics.csv", index=False)

        logger.info("Step 7: Analyzing results and creating visualizations")
        plot_metrics(metrics_df, "./results/performance_plots/metrics.png")

        # Generate confusion matrices
        for result in results:
            cm = evaluate_predictions(
                y_true=df_subset["label"].tolist(),
                y_pred=result["Predictions"],
                return_confusion_matrix=True,
            )
            plot_confusion_matrix(
                cm=cm["confusion_matrix"],
                model=result["Model"],
                prompt=result["Prompt"],
                save_path=f"./results/confusion_matrices/cm_{result['Model']}_{result['Prompt']}.png",
            )

        logger.info("Sentiment analysis pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
