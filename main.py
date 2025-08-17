```python
# main.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# Load and prepare data (example with a CSV file)
# The CSV should have columns: 'text' and 'label'
# Label should be 0 for negative, 1 for positive
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df['text'].tolist(), df['label'].tolist()

# Tokenize the data
def tokenize_data(texts, labels, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return encodings, labels

# Dataset class for PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Fine-tune BERT model
def train_model(train_dataset, val_dataset, model_name="bert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return trainer

# Main execution
if __name__ == "__main__":
    # Load data
    texts, labels = load_data('data.csv')
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    
    # Tokenize data
    train_encodings, train_labels = tokenize_data(train_texts, train_labels)
    val_encodings, val_labels = tokenize_data(val_texts, val_labels)
    
    # Create datasets
    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    
    # Train model
    trainer = train_model(train_dataset, val_dataset)
    
    # Save model
    trainer.save_model('./sentiment-model')
    
    # Test with a sample
    classifier = pipeline("sentiment-analysis", model="./sentiment-model", tokenizer="bert-base-uncased")
    result = classifier("This is a great movie!")
    print(result)
```