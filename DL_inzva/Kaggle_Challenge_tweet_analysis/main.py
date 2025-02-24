import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import random
from tqdm import tqdm

# Hyperparameters
MODEL_PATH = "./bert_imdb_reviews.pth"
DATASET_PATH = "./data.csv"
BATCH_SIZE = 4
MAX_LENGTH = 512

class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def load_model(model_path: str, device = torch.device("cpu")):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_all(model, dataloader, device, df):
    predictions = []
    model.eval()
    batch_count = len(dataloader)
    print(f"Predicting {batch_count} batches")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    return ["positive" if pred == 1 else "negative" for pred in predictions]

def predict_n(n, model, df, tokenizer, device):
    correct = 0
    for i in range(n):
        ind = random.randint(0, len(df)-1)
        sample = df['review'][ind]
        sentiment = df['sentiment'][ind]
        prediction = predict(model, sample, tokenizer, device)
        if sentiment == prediction:
            correct += 1
    return correct / n


def predict(model, sample, tokenizer, device):
    model.eval()
    encoding = tokenizer(sample, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(output.logits, dim=1).item()
    
    return "positive" if prediction == 1 else "negative"

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    texts = df['review'].tolist()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Prepare dataset and dataloader
    dataset = ReviewDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    print( "\n\n\n")
    print("Predicting...")
    # Predict random sample
    sample = random.choice(texts)
    prediction = predict(model, sample, tokenizer, device)
    print(f"Sample:\n {sample} \n Prediction: {prediction}")

    # Predict n samples
    n = 100
    accuracy = predict_n(n, model, df, tokenizer, device)
    print(f"Accuracy on {n} samples: {accuracy:.6f}")
    
    # Predict all samples and calculate accuracy
    """predictions = predict_all(model, dataloader, device, df)
    if len(predictions) == len(df):
        df['predictions'] = predictions
        accuracy = (df['sentiment'] == df['predictions']).mean()
        print(f"Accuracy: {accuracy:.6f}")

         # Save results
        df.to_csv("./predicted_data.csv", index=False)
        print("Predictions saved to predicted_data.csv")"""

    # Save model
    #* save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()