# Sentiment Analysis with Small Language Models

## Project Overview

This project explores the use of Small Language Models (SLMs) for sentiment analysis of movie reviews. Specifically, it compares the performance and efficiency of two quantized models from the Qwen2.5 family:

- `bartowski/Qwen2.5-1.5B-Instruct-GGUF`: A 1.5 billion-parameter model.
- `bartowski/Qwen2.5-0.5B-Instruct-GGUF`: A 500 million-parameter model.

The task involves implementing local inference, prompt engineering, evaluating model performance, and visualizing results.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dnlgsv/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create a Virtual Environment
By default Python 3.12.7 was used.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Models
Download the models from the Hugging Face model hub and save them in the models/ directory.
- **Download the 1.5B model**
```bash
curl -L -o ./models/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf
```

- **Download the 0.5B model**
```bash
curl -L -o ./models/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf
```

### 5. Prepare the Dataset
This will generate a balanced subset of 500 movie reviews in data/subset.csv.
```bash
python src/dataset.py
```

### 6. Run the Sentiment Analysis Pipeline
This script will perform inference, evaluate performance, and generate visualizations. Results will be saved in the results/ directory. Original subset size is 500 reviews, but can be changed with the `--subset_size` argument.

```bash
python src/main.py --subset_size 10
```
If ModuleNotFoundError appears, as tmp solution, run:
```bash
export PYTHONPATH=.
```
