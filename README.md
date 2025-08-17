# Sentiment Analysis with BERT

This project demonstrates how to fine-tune a BERT model for sentiment analysis using the Hugging Face Transformers library. The model is trained to classify text as positive or negative sentiment.

## Features

- Fine-tunes a BERT model for sentiment classification
- Uses Hugging Face Transformers for easy model handling
- Includes data preprocessing and tokenization
- Provides a simple interface for testing the model

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Pandas
- Scikit-learn

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/BoozeLee/sentiment-analysis-bert.git
   cd sentiment-analysis-bert
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Create a CSV file named `data.csv` with two columns: `text` and `label`
   - Labels should be 0 for negative sentiment and 1 for positive sentiment

2. Run the training script:
   ```bash
   python main.py
   ```

3. The trained model will be saved in the `sentiment-model` directory

## Example

After training, you can use the model to classify new text:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="./sentiment-model", tokenizer="bert-base-uncased")
result = classifier("This is a great movie!")
print(result)
```

## License

MIT