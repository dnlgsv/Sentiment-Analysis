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
cd sentiment_analysis
```

### 2. Create a Virtual Environment
    
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
    
```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
This will generate a balanced subset of 1,000 movie reviews in data/subset.csv.
```bash
python src/dataset.py
```

### 5. Run the Sentiment Analysis Pipeline
This script will perform inference, evaluate performance, and generate visualizations. Results will be saved in the results/ directory.

```bash
python src/main.py
```

