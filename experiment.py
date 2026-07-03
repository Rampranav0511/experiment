from transformers import pipeline


# to check whether a sentence has a positive or a negative tone
classifier = pipeline("sentiment-analysis")

texts = [
    "I love learning about machine learning!",
    "This is really confusing and frustrating.",
    "The weather today is okay I guess.",
    "EleutherAI does amazing open source research!"
]

# so basically the output/return type of the pipeline function is a list of dictionaries
"""thats why for each text , we get our own dictionary , so here , result[0] basically accesses that dictionary and dict[label] 
gives us the confidence in the tone of the sentence """
for text in texts:
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score'] * 100
    print(f"Text: {text}")
    print(f"Result: {label} ({score}% confidence)")
    print()

# Text generation- to simply complete a sentence
print("text Generation")
generator = pipeline("text-generation", model="gpt2")
output = generator("Machine learning is", max_length=30, num_return_sequences=1)
print(f"Generated: {output[0]['generated_text']}")