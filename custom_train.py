import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Load your dataset
file_path = 'F:\\finetunining sample\\combined_dataset.csv'
dataset = pd.read_csv(file_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 1: Lowercasing
dataset['input_text'] = dataset['input_text'].str.lower()

# Step 2: Remove unnecessary special characters but keep numbers and meaningful punctuation
dataset['input_text'] = dataset['input_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', '', x))

# Step 3: Tokenization
dataset['input_text_tokens'] = dataset['input_text'].apply(lambda x: tokenizer.tokenize(x))

# Step 4: Padding/Truncation
max_length = 128
dataset['input_text_padded'] = dataset['input_text_tokens'].apply(
    lambda x: tokenizer.convert_tokens_to_ids(x)[:max_length] + [0] * (max_length - len(x))
)

# Encode the target variables
label_encoder_priority = LabelEncoder()
label_encoder_resolution = LabelEncoder()

dataset['priority_encoded'] = label_encoder_priority.fit_transform(dataset['priority_binned'])
dataset['resolution_encoded'] = label_encoder_resolution.fit_transform(dataset['bug_resolution_time'])

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Convert columns to tensors
def convert_to_tensor(df, feature_col, target_col):
    features = torch.tensor(df[feature_col].tolist(), dtype=torch.long)
    targets = torch.tensor(df[target_col].tolist(), dtype=torch.long)
    return features, targets

train_features, train_priority = convert_to_tensor(train_df, 'input_text_padded', 'priority_encoded')
train_features, train_resolution = convert_to_tensor(train_df, 'input_text_padded', 'resolution_encoded')

val_features, val_priority = convert_to_tensor(val_df, 'input_text_padded', 'priority_encoded')
val_features, val_resolution = convert_to_tensor(val_df, 'input_text_padded', 'resolution_encoded')

# Create TensorDataset instances
train_dataset = TensorDataset(train_features, train_priority, train_resolution)
val_dataset = TensorDataset(val_features, val_priority, val_resolution)

# Define a custom multi-output BERT model
class MultiOutputBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        return logits_task1, logits_task2

# Initialize the model
num_labels_priority = len(label_encoder_priority.classes_)
num_labels_resolution = len(label_encoder_resolution.classes_)

model = MultiOutputBERT(num_labels_priority, num_labels_resolution)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids, priority_labels, resolution_labels = [x.to(device) for x in batch]
        logits_task1, logits_task2 = model(input_ids)
        
        loss_task1 = criterion(logits_task1, priority_labels)
        loss_task2 = criterion(logits_task2, resolution_labels)
        
        loss = loss_task1 + loss_task2
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels_task1 = []
    all_labels_task2 = []
    all_preds_task1 = []
    all_preds_task2 = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids, priority_labels, resolution_labels = [x.to(device) for x in batch]
            logits_task1, logits_task2 = model(input_ids)
            
            loss_task1 = criterion(logits_task1, priority_labels)
            loss_task2 = criterion(logits_task2, resolution_labels)
            
            loss = loss_task1 + loss_task2
            total_loss += loss.item()
            
            preds_task1 = logits_task1.argmax(dim=-1)
            preds_task2 = logits_task2.argmax(dim=-1)
            
            all_labels_task1.extend(priority_labels.cpu().numpy())
            all_labels_task2.extend(resolution_labels.cpu().numpy())
            all_preds_task1.extend(preds_task1.cpu().numpy())
            all_preds_task2.extend(preds_task2.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    
    # Metrics for task 1
    accuracy_task1 = accuracy_score(all_labels_task1, all_preds_task1)
    precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(all_labels_task1, all_preds_task1, average='weighted')
    
    # Metrics for task 2
    accuracy_task2 = accuracy_score(all_labels_task2, all_preds_task2)
    precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(all_labels_task2, all_preds_task2, average='weighted')

    metrics = {
        'val_loss': avg_loss,
        'accuracy_task1': accuracy_task1,
        'precision_task1': precision_task1,
        'recall_task1': recall_task1,
        'f1_task1': f1_task1,
        'accuracy_task2': accuracy_task2,
        'precision_task2': precision_task2,
        'recall_task2': recall_task2,
        'f1_task2': f1_task2,
    }

    return metrics

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training the model
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss}")
    
    val_metrics = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_metrics['val_loss']}")
    print(f"Task 1 - Accuracy: {val_metrics['accuracy_task1']}, F1 Score: {val_metrics['f1_task1']}")
    print(f"Task 2 - Accuracy: {val_metrics['accuracy_task2']}, F1 Score: {val_metrics['f1_task2']}")
