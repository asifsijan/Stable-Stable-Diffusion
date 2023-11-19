

# Commented out IPython magic to ensure Python compatibility.
# # @title Initial Dependencies
# %%capture
# !pip install transformers
# !pip install sentencepiece
# !pip install accelerate
# !pip install datasets

# @title More Imports
# %%capture
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, classification_report
import re

f1 = 'name.csv'
f2 = 'description.csv'

dfn = pd.read_csv(f1, header=None, names=['Column1'])
dfd = pd.read_csv(f2, header=None, names=['Column2'])
merged_data = pd.concat([dfn, dfd], axis=1)

merged_file_path = 'merged_data.csv'
merged_data.to_csv(merged_file_path, index=False)

merged_file_path = 'indexed_merged_data.csv'
merged_data.to_csv(merged_file_path, index=True)



data_path = 'merged_data.csv'
data = pd.read_csv(data_path)

# Display the DataFrame
# data.head()

df1 = pd.read_csv('merged_data.csv')
df2 = pd.read_csv('merged_data.csv')

df_doubled = pd.concat([df1, df2], ignore_index=True)


df_doubled.rename(columns={'Column1': 'Name', 'Column2': 'Description'}, inplace=True)
df_temp1 = pd.DataFrame(df_doubled)
df_temp2 = pd.DataFrame(df_doubled)


df_troubled = pd.concat([df_temp1, df_temp2], ignore_index=True)





#Little tweaking
df = pd.DataFrame(df_troubled)

# Remove all words from 'Column1' in 'Column2' for each row
df['Description'] = df.apply(lambda row: re.sub(r'\b(?:%s)\b' % '|'.join(map(re.escape, row['Name'].split())), '', row['Description'], flags=re.IGNORECASE).strip(), axis=1)

# Display the modified dataframe
df.head()



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['Name'], test_size=0.2, random_state=42)

# Tokenizer and Model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Define a custom dataset with dynamic padding
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = f"generate animal name: {self.texts.iloc[idx]}"
        target_text = self.labels.iloc[idx]

        # Tokenize input text and target text
        inputs = self.tokenizer(input_text, max_length=self.max_length, return_tensors="pt", truncation=True, padding='max_length')
        labels = self.tokenizer(target_text, max_length=self.max_length, return_tensors="pt", truncation=True, padding='max_length')

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

# Create dataloaders
train_dataset = CustomDataset(X_train, y_train, tokenizer)
test_dataset = CustomDataset(X_test, y_test, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.numel()
        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    training_accuracy = correct_predictions / total_samples

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Training Loss: {average_loss:.4f}")
    print(f"  Training Accuracy: {training_accuracy:.2%}")

# ... (rest of your code)

# Ensure consistent lengths (remove this line)
# y_test = y_test[:len(all_predictions)]

# Save the fine-tuned model
model_save_path = "model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


# Evaluation
model.eval()
all_predictions = []

for batch in tqdm(test_dataloader, desc="Evaluating"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model.generate(input_ids, max_length=20)  # Adjust max_length as needed
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    all_predictions.extend(predictions)

# Ensure consistent lengths
y_test = y_test[:len(all_predictions)]

# Evaluate the model
accuracy = accuracy_score(y_test, all_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, all_predictions))

# Load the fine-tuned T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("model")
model = T5ForConditionalGeneration.from_pretrained("model")

# Input descriptions to test
input_descriptions = [
    "This is a medium-sized heron that is widespread across North America and parts of Asia. .",
    # Add more descriptions as needed
]

# Tokenize input descriptions
tokenized_inputs = [tokenizer(f"generate an animal name: {desc}", return_tensors="pt") for desc in input_descriptions]

# Generate predictions
with torch.no_grad():
    outputs = model.generate(**tokenized_inputs[0])  # Use the first tokenized input as an example

# Decode predictions
decoded_predictions = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Print intermediate results for debugging
print("Generated Tokens:", outputs[0])
print("Decoded Tokens:", decoded_predictions)
