import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Hyperparameters
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

# Convert labels to integers
label_mapping = {"negative": 0, "positive": 1}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df['review'].tolist(), df['sentiment'].tolist(), test_size=0.1, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

def train(model, train_loader, val_loader, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        
        train_acc = total_correct / len(train_dataset)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

def evaluate(model, val_loader):
    model.eval()
    total_correct = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
    
    return total_correct / len(val_dataset)

# Train the model
train(model, train_loader, val_loader, optimizer, epochs=EPOCHS)

# Save the trained model
torch.save(model.state_dict(), "./bert_imdb_reviews.pth")