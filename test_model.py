# test_model.py
from transformers import pipeline

def test_model():
    # Load the trained model
    classifier = pipeline("sentiment-analysis", model="./sentiment-model", tokenizer="bert-base-uncased")
    
    # Test examples
    test_texts = [
        "This movie is fantastic!",
        "I hated this film.",
        "Not bad, could be better.",
        "Amazing experience, highly recommend!",
        "Boring and predictable plot."
    ]
    
    # Run predictions
    for text in test_texts:
        result = classifier(text)
        print(f"Text: {text}")
        print(f"Prediction: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()