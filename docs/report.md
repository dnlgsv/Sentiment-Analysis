# Report: Sentiment Analysis with Small Language Models

## Table of Contents

1. [Introduction](#1-introduction)
   - [Background of the Task](#background-of-the-task)
   - [Objectives of the Report](#objectives-of-the-report)
2. [Dataset Preparation](#2-dataset-preparation)
   - [Selection Criteria for Subset](#selection-criteria-for-subset)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Model Inference Implementation](#3-model-inference-implementation)
   - [Models Used](#models-used)
   - [Inference Setup](#inference-setup)
   - [Inference Parameters and Justification](#inference-parameters-and-justification)
   - [Implementation Details](#implementation-details)
4. [Prompt Engineering](#4-prompt-engineering)
   - [Initial Prompt Structures](#initial-prompt-structures)
   - [Refinement Process](#refinement-process)
   - [Final Prompt Selection](#final-prompt-selection)
   - [Challenges and Solutions](#challenges-and-solutions)
5. [Evaluation Metrics](#5-evaluation-metrics)
   - [Chosen Metrics and Justification](#chosen-metrics-and-justification)
   - [Calculation Methods](#calculation-methods)
6. [Results and Visualization](#6-results-and-visualization)
   - [Performance Comparison](#performance-comparison)
   - [Effect of Prompt Engineering](#effect-of-prompt-engineering)
   - [Visualization](#visualization)
   - [Resource Utilization](#resource-utilization)
7. [Analysis of Results](#7-analysis-of-results)
   - [Insights and Patterns](#insights-and-patterns)
   - [Trade-offs Observed](#trade-offs-observed)
   - [Limitations](#limitations)
8. [Discussion](#8-discussion)
   - [Real-World Applications](#real-world-applications)
   - [Model Performance](#model-performance)
   - [Performance Bottlenecks](#performance-bottlenecks)
9. [Conclusion](#9-conclusion)
   - [Summary of Findings](#summary-of-findings)
   - [Final Thoughts](#final-thoughts)
10. [Future Work](#10-future-work)
    - [Possible Next Steps](#possible-next-steps)
    - [Scalability Considerations](#scalability-considerations)
11. [References](#11-references)

---

## 1. Introduction

### Background of the Task

Small Language Models (SLMs) with <=1.5 billion parameters have emerged as powerful tools for specific tasks, offering a balance between performance and computational efficiency. Unlike their larger counterparts, SLMs can be deployed on devices with limited resources, making them suitable for applications like real-time text editing and sentiment analysis without the need for cloud-based servers. They also can be used with AI Agents.

### Objectives of the Report

This report aims to:

- **Explore the dataset** and prepare a representative subset for sentiment analysis.
- **Implement local inference** for two SLMs on the task of sentiment analysis of movie reviews.
- **Design and refine prompts** to maximize accuracy.
- **Compare the performance and efficiency** of a 1.5B-parameter model and a 0.5B-parameter model.
- **Analyze results** to draw meaningful insights.
- **Document** the research process, findings, and potential real-world applications.

---

## 2. Dataset Preparation

### Selection Criteria for Subset

The dataset **ajaykarthick/imdb-movie-reviews** contains 50,000 entries, which is impractical for a full local run given computational constraints. To create a manageable yet representative subset, the following criteria were applied:

- **Size**: Selected a subset of 500 reviews to balance computational feasibility with statistical significance.
- **Class Balance**: Ensured an equal number of positive and negative reviews (250 each) to prevent bias.
- **Random Sampling**: Randomly selected reviews to maintain diversity.

### Data Preprocessing

- **Dataset subset creation**: 250 positive and 250 negative reviews were randomly sampled from the original dataset and saved to data directory.

### Exploratory Data Analysis (EDA)

- **Review Length Distribution**: Analyzed the distribution of review lengths to understand text complexity.
- **Sentiment Distribution**: Confirmed the balanced distribution of positive and negative reviews in the subset.
- **Word Frequency**: Identified the most common words in positive vs. negative reviews to detect any prominent patterns.

---

## 3. Model Inference Implementation

### Models Used

- **bartowski/Qwen2.5-1.5B-Instruct-GGUF**: A 1.5-billion-parameter instruction-tuned model with enhanced capabilities in coding, mathematics, and handling structured data.
- **bartowski/Qwen2.5-0.5B-Instruct-GGUF**: A 500-million-parameter model, three times smaller than its counterpart, optimized for efficiency.

### Inference Setup

- **Hardware**: Conducted experiments on both MPS and CPU environments to test performance and efficiency.
- **Libraries and Tools**:
  - Used `llama-cpp-python` for efficient MPS/GPU/CPU inference of quantized GGUF models.

### Inference Parameters and Justification

- **Temperature**: Set to 0.0 to ensure accurate and deterministic predictions for sentiment analysis. However, a value of 0.7 was used for self-consistency prompting tecnhnique where divercity is important.
- **Top_p**: Set to default value: 0.95.
- **Top_k**: Set to default value: 40.
- **Max Tokens**: Limited to 256 to prevent excessively long outputs.

These parameters were chosen based on initial experimentation to optimize the trade-off between response quality and inference time. Since the main task is sentiment analysis, deterministic predictions were preferred over diverse outputs.

### Implementation Details

- Developed a Python script to load the models and perform inference on the dataset.
- Ensured reproducibility by setting random seeds where applicable.

---

## 4. Prompt Engineering

### Initial Prompt Structures

- **Simple Prompt**:
A baseline, straightforward prompt asking the model to determine the sentiment:
```
Your goal is to say whether a given review expresses a <sentiment>Positive</sentiment> or <sentiment>Negative</sentiment> sentiment towards the movie. Only return the sentiment in the format: <sentiment>Positive</sentiment> or <sentiment>Negative</sentiment>. No additional text is allowed.
```

- **Roleplay Prompt**:
A prompt that uses previous ideas, but also includes role of a film critic:

```
You are an experienced film critic. Your goal is to say whether a given review expresses a 'Positive' or 'Negative' sentiment towards the movie. Only return the sentiment in the format: <sentiment>Positive</sentiment> or <sentiment>Negative</sentiment>. No additional text is allowed.
```

- **Chain of Thought prompt**:
A concise prompt that includes previous ideas, but also instructs the model to use chain of thoughts prompting technique before giving a final answer with "Positive" or "Negative" sentiment only:

```
You are a sentiment analysis assistant. Determine whether the sentiment of a movie review is 'Positive' or 'Negative'. To decide, follow these steps:

      1. Identify key phrases that reflect opinions or emotions in the review.
      2. Determine whether these phrases are positive or negative in tone.
      3. Count the positive and negative sentiments to decide the overall polarity.
      4. Conclude whether the review is 'Positive' or 'Negative'.
      Return only the sentiment with html tags in your response.
      Format your response as follows:
      <sentiment>sentiment</sentiment>
```

- **Chain of Thought prompt with few-shot examples Prompt**:
A concise prompt that includes previous ideas, but also includes few-shot examples to guide the model:

```
You are a sentiment analysis assistant. Determine whether the sentiment of a movie review is positive or negative. To decide, follow these steps:

      1. Identify key phrases that reflect opinions or emotions in the review.
      2. Determine whether these phrases are positive or negative in tone.
      3. Count the positive and negative sentiments to decide the overall polarity.
      4. Conclude whether the review is 'Positive' or 'Negative'.
      Return only the sentiment with html tags in your response.
      Format your response as follows:
      <sentiment>sentiment</sentiment>

      Example 1:
      Review: "The movie was absolutely fantastic. The acting was brilliant, and the story was captivating."
      Step 1: Key phrases: "absolutely fantastic," "acting was brilliant," "story was captivating."
      Step 2: All phrases are positive.
      Step 3: Positive count = 3, Negative count = 0.
      Step 4: Overall sentiment: <sentiment>Positive</sentiment>.

      Example 2:
      Review: "The plot was dull, and the characters were uninteresting."
      Step 1: Key phrases: "plot was dull," "characters were uninteresting."
      Step 2: All phrases are negative.
      Step 3: Positive count = 0, Negative count = 2.
      Step 4: Overall sentiment: <sentiment>Negative</sentiment>.

      Now analyze the review:
      Step 1: Key phrases:
      Step 2: Positive or negative tone for each:
      Step 3: Positive count = X, Negative count = Y.
      Step 4: Overall sentiment: [<sentiment>Positive</sentiment>/<sentiment>Negative</sentiment>].
      Format your response as follows:
      <sentiment>sentiment</sentiment>
```

- **Self-consistency prompt**:
A concise prompt that includes any of the previous ideas, but temperature is increased from 0 to 0.7 to allow for more diverse outputs and then choose the most common sentiment.

### Refinement Process
- **Explicit Instructions**: Explicit Instructions: Specified the desired format of the output (e.g., "Answer with 'Positive' or 'Negative' only", use HTML tags around the final answer). This helps the model understand exactly what is expected. It is espicially useful for smaller models.
- **Added Context**: Included examples in the prompt to guide the model. Providing concrete examples helps the model emulate the desired response pattern.
- **Role Assignment**: Assigned a specific role to the model (e.g., "You are an experienced film critic", "You are a sentiment analysis assistant") to influence the style and content of the response. This helps the model adopt a perspective that is more aligned with the task.
- **Chain-of-Thought Guidance**: Instructed the model to follow specific reasoning steps before arriving at the final answer (e.g., identifying key phrases, determining tone, counting sentiments). This encourages the model to perform a structured analysis, which can improve accuracy.
- **Temperature Adjustment**: Increased the temperature setting (from 0 to 0.7) in the self-consistency prompt to allow for more diverse outputs. Sampling multiple responses at higher temperature and selecting the most common sentiment can improve reliability.
- **Formatting Instructions**: Provided clear formatting guidelines (e.g., "Return only the sentiment with HTML tags in your response", "Format your response as follows: <sentiment>sentiment</sentiment>") to ensure consistent output that can be easily parsed.
- **Few-Shot Learning**: Included few-shot examples in the prompt to demonstrate the desired reasoning process and output format. This helps the model understand how to apply the instructions to new inputs.
- **Constraint Enforcement**: Emphasized that no additional text is allowed beyond the specified format to prevent the model from adding unnecessary information. This keeps the responses concise and focused.
- **Iterative Prompt Refinement**: Adjusted the prompts iteratively based on the outputs received. If the model's responses didn't meet expectations, the prompts were refined to provide clearer instructions or additional guidance.
- **Simplification of Language**: Simplified the language in the prompts to reduce ambiguity and make instructions clearer. This helps prevent misinterpretation of the instructions by the model.
- **Self-Consistency Mechanism**: Employed a self-consistency mechanism by generating multiple outputs with the same prompt (using higher temperature) and then choosing the most common sentiment among them. This leverages the model's probabilistic nature to improve accuracy.


### Final Prompt Selection

After iterative testing, the final prompt used was:

Chain of Thought prompt with few-shot examples is the most effective prompt for the sentiment analysis task. It provides clear instructions, includes examples for guidance, and follows a structured reasoning process. The prompt is concise, yet comprehensive, and has shown to improve the model's performance significantly. The most essential part here is to choose the right examples and make sure the model understands the reasoning process.

**Justification**:

- **Clarity**: Provides clear instructions to the model.
- **Conciseness**: Limits the model's output to the required information.
- **Format Specification**: Reduces variability in responses, aiding in evaluation.

### Challenges and Solutions

- **Overly Verbose Responses**: Models sometimes generated lengthy explanations.

  - **Solution**: Explicitly instructed models to provide concise answers.

- **Ambiguous Outputs**: Some outputs were unclear (e.g., "The review seems positive overall but has negative aspects").

  - **Solution**: Tightened the prompt to request a single-word answer.

- **Inconsistent Formatting**: Addressed by specifying the exact expected output format in the prompt.

---

## 5. Evaluation Metrics

### Chosen Metrics and Justification

- **Accuracy**: Primary metric to measure the proportion of correct predictions.
- **Precision and Recall**: To evaluate the model's performance on each class.
- **F1 Score**: Harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: To visualize the performance and identify any biases toward a particular class.

Depends on the data and audience, the metrics can be adjusted. For example, if the data is imbalanced, F1 score is a better metric than accuracy. If the data is balanced, accuracy is a good metric and very easy to understand for techenical and non-techenical audience.
In this case, the data is balanced and accuracy is a good metric to use.

### Calculation Methods

- Used scikit-learn's `metrics` library to compute the evaluation metrics.
- Ensured that the labels were correctly mapped between the true sentiments and the model's predictions.

---

## 6. Results and Visualization

### Performance Comparison
#### The best results achieved after prompt engineering:

ID | Model                 |Prompt                 | Accuracy | Precision | Recall | F1 Score |
---|-----------------------|-----------------------|----------|-----------|--------|----------|
0  | Qwen2.5-1.5B-Instruct |Chain of Thought(V2)   | 93.4%    | 92.5%     | 94.4%  | 93.5%    |
1  | Qwen2.5-0.5B-Instruct |Roleplay(V2)           | 87.0%    | 97.4%     | 76.0%  | 85.4%    |

#### All results(sorted by accuracy):

ID | Model                 |Prompt                 | Accuracy | Precision | Recall | F1 Score |
---|-----------------------|-----------------------|----------|-----------|--------|----------|
0  | Qwen2.5-1.5B-Instruct |Chain of Thought(V2)   | 93.4%    | 92.5%     | 94.4%  | 93.5%    |
1  | Qwen2.5-1.5B-Instruct |CoT with Few Shot(V2)  | 92.8%    | 90.5%     | 95.6%  | 93.0%    |
2  | Qwen2.5-1.5B-Instruct |Simple(V3)             | 92.6%    | 89.6%     | 96.4%  | 92.9%    |
3  | Qwen2.5-0.5B-Instruct |Roleplay(V2)           | 87.0%    | 97.4%     | 76.0%  | 85.4%    |
4  | Qwen2.5-1.5B-Instruct |Roleplay(V2)           | 80.1%    | 72.8%     | 98.4%  | 83.7%    |
5  | Qwen2.5-0.5B-Instruct |Simple(V3)             | 76.0%    | 99.2%     | 52.4%  | 68.6%    |
6  | Qwen2.5-0.5B-Instruct |CoT with Few Shot(V2)  | 66.4%    | 59.9%     | 99.2%  | 74.7%    |
7  | Qwen2.5-0.5B-Instruct |Chain of Thought(V2)   | 55.4%    | 52.9%     | 99.6%  | 69.1%    |

### Effect of Prompt Engineering
Results:

- The 0.5B model's accuracy varies from **55%** to **87%** after refining the prompt.
- The 1.5B model's accuracy improved from **80%** to **93%** after refining the prompt.

Conclusions:

- The 0.5B model shows a significant improvement in accuracy after prompt engineering. However, it is still less accurate than the 1.5B model. Moreover the 0.5B model is more sensitive to prompt changes and can drastically change the accuracy.
- The 1.5B model shows steady improvement in accuracy after prompt engineering. It is more robust to prompt changes and maintains a high level of accuracy.
- Prompt engineering is crucial for improving the performance of smaller models, but the impact is more pronounced for less capable models.

### Visualization

![Metrics Comparison](results/perfomance_plots/metrics.png)

*Figure 1: Metrics of both models with defferent prompt engineering approaches.*

## 7. Analysis of Results

### Insights and Patterns

- **Model Size vs. Performance**: The larger model outperformed the smaller one but at the cost of increased computational resources.
- **Impact of Prompt Engineering**: Significant improvement observed in the smaller model, indicating that prompt engineering can help bridge the performance gap. However, the larger model also benefited from prompt refinement. Moreover, the smaller model is more sensitive to prompt changes and can drastically change the accuracy to lower values.

### Trade-offs Observed

- **Accuracy vs. Efficiency**: The 1.5B model offers higher accuracy but requires more resources and time per inference.
- **Prompt Sensitivity**: The smaller model is more sensitive to prompt changes, suggesting that prompt engineering is more crucial for less capable models.

### Limitations

- **Computational Constraints**: Limited ability to experiment with larger datasets or more complex models.
- **Model Generalization**: Models may not generalize well beyond the dataset due to limited training data exposure during this task.

---

## 8. Discussion

### Real-World Applications

- **On-Device Sentiment Analysis**: Smaller models can be deployed on devices with limited resources for real-time sentiment analysis.
- **Customized Text Editors**: Integration into applications to provide feedback on writing tone and sentiment.
- **AI Agents**: Incorporation into AI agents for sentiment-aware responses in conversational interfaces.

### Model Performance

- **Prompt Engineering Effectiveness**: Demonstrated that with carefully designed prompts, smaller models can achieve performance closer to larger models.
- **Scalability**: Smaller models offer a scalable solution for applications where deploying large models is impractical.

### Performance Bottlenecks

- **Inference Speed**: Addressed by optimizing code and using batch processing.
- **Memory Usage**: Managed by using quantized models and efficient libraries.

---

## 9. Conclusion

### Summary of Findings

- Successfully implemented local inference for two SLMs on sentiment analysis.
- Achieved up to **93.4% accuracy** with the 1.5B model and **87.0%** with the 0.5B model.
- Demonstrated that prompt engineering significantly improves the performance of small language models.
- Identified trade-offs between model size, accuracy, and computational resources.

### Final Thoughts

This study highlights the potential of small language models for practical applications, especially when resource constraints are a consideration. With effective prompt engineering, smaller models can perform competitively, making them viable for deployment in various real-world scenarios.

---

## 10. Future Work

### Possible Next Steps

- **Expand Dataset Size**: Test models on larger subsets to validate findings.
- **Fine-Tuning Models**: Explore fine-tuning smaller models on the specific task to improve performance.
- **Advanced Prompt Techniques**: Investigate few-shot learning by including more examples in prompts. We can:
   - **1.** Use more diverse examples to cover a wider range of sentiments.
   - **2.** Include examples with varying levels of complexity to test the model's adaptability.
   - **3.** Use similarity search to identify relevant examples for a given review.
---

## 11. References

- **Ajay Karthick**. (n.d.). *IMDB Movie Reviews Dataset*. Retrieved from [Hugging Face Datasets](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews)
- **Bartosz Staszewski**. (n.d.). *Qwen2.5 Models*. Retrieved from [Hugging Face Models](https://huggingface.co/bartowski)
- **Scikit-learn Metrics**. (n.d.). Retrieved from [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Llama CPP Python**. (n.d.). Retrieved from [GitHub Repository](https://github.com/abetlen/llama-cpp-python)

---

**Note**: This report summarizes the methodology and findings of implementing sentiment analysis using small language models. All code and instructions for reproducing the results are available in the accompanying GitHub repository.

---
