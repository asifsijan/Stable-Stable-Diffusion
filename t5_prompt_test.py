import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re


tokenizer = T5Tokenizer.from_pretrained("model")
model = T5ForConditionalGeneration.from_pretrained("model")

# Input descriptions to test
input_descriptions = [
    "A large hunter",
    # Add more descriptions as needed
]

# Tokenize input descriptions
tokenized_inputs = [tokenizer(f"generate name: {desc}", return_tensors="pt") for desc in input_descriptions]

# Generate predictions
with torch.no_grad():
    outputs = model.generate(**tokenized_inputs[0])  # Use the first tokenized input as an example

# Decode predictions
decoded_predictions = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Print intermediate results for debugging
print("Generated Tokens:", outputs[0])
print("Decoded Tokens:", decoded_predictions)

# Print final result
print("Generated Animal Name:", decoded_predictions)

