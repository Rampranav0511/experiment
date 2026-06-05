from transformers import pipeline

print("Loading model...")

# Sentiment analysis - is text positive or negative?
classifier = pipeline("sentiment-analysis")

texts = [
    "I love learning about machine learning!",
    "This is really confusing and frustrating.",
    "The weather today is okay I guess.",
    "EleutherAI does amazing open source research!"
]

print("\n--- Sentiment Analysis ---")
for text in texts:
    result = classifier(text)
    label = result[0]['label']
    score = round(result[0]['score'] * 100, 2)
    print(f"Text: {text}")
    print(f"Result: {label} ({score}% confidence)")
    print()

# Text generation - complete a sentence
print("--- Text Generation ---")
generator = pipeline("text-generation", model="gpt2")
output = generator("Machine learning is", max_length=30, num_return_sequences=1)
print(f"Generated: {output[0]['generated_text']}")