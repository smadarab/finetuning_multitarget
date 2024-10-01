import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Load your dataset
file_path = 'F:\\finetunining sample\\updated_combined_data.csv'
dataset = pd.read_csv(file_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Adjust tokenization for shorter inputs (15 words)
def tokenize_and_pad(text, tokenizer, max_length=125):  # Adjusted to handle short inputs (15 words)
    tokens = tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_length)
    return torch.tensor(tokens).to('cuda')

# Lowercasing and cleaning text
dataset['input_text'] = dataset['input_text'].str.lower()
dataset['input_text_mod'] = dataset['input_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', x))

dataset.to_csv("hello.csv")
print("*"*10,"done","*"*10)
# Tokenization
dataset['input_text_padded'] = dataset['input_text'].apply(lambda x: tokenize_and_pad(x, tokenizer))

# Encode the target variables
label_encoder_priority = LabelEncoder()
label_encoder_resolution = LabelEncoder()

dataset['priority_encoded'] = label_encoder_priority.fit_transform(dataset['priority_binned'])
dataset['resolution_encoded'] = label_encoder_resolution.fit_transform(dataset['bug_resolution_time'])

# Retain the original input text for analysis
dataset['original_input_text'] = dataset['input_text']

# Drop the columns after retaining the original text for later use
dataset = dataset.drop(['priority_binned', 'input_text', 'bug_resolution_time'], axis=1)

# Prepare input and target data
input_data = np.stack([tensor.cpu().numpy() for tensor in dataset['input_text_padded']])
target_data = np.array(list(zip(dataset['priority_encoded'], dataset['resolution_encoded'])))

# Convert columns to tensors with attention masks
def convert_to_tensor_with_mask(df, feature_col, target_col):
    features = torch.stack(df[feature_col].tolist())
    attention_mask = (features != 0).long().cuda()
    targets = torch.tensor(df[target_col].tolist(), dtype=torch.long).cuda()
    return features, attention_mask, targets

# Define a multi-output BERT model
class MultiOutputBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.5)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        return logits_task1, logits_task2

# Mixed precision training scaler
scaler = GradScaler()

# Training function
def train_epoch(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in dataloader:
        b_input_ids, b_attention_mask, b_priority, b_resolution = batch
        
        optimizer.zero_grad()
        
        with autocast():
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
        
            loss_priority = F.cross_entropy(logits_priority, b_priority)
            loss_resolution = F.cross_entropy(logits_resolution, b_resolution)
        
            loss = loss_priority + loss_resolution
        
        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluation function with storage of predictions and actual values
def evaluate_epoch_with_storage(model, dataloader, dataset):
    model.eval()
    all_priority_preds = []
    all_resolution_preds = []
    all_priority_labels = []
    all_resolution_labels = []
    all_input_texts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            b_input_ids, b_attention_mask, b_priority, b_resolution = batch
            
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
            
            # Get predictions
            priority_preds = logits_priority.argmax(dim=1).cpu().numpy()
            resolution_preds = logits_resolution.argmax(dim=1).cpu().numpy()
            
            # Extend lists for analysis
            all_priority_preds.extend(priority_preds)
            all_resolution_preds.extend(resolution_preds)
            all_priority_labels.extend(b_priority.cpu().numpy())
            all_resolution_labels.extend(b_resolution.cpu().numpy())
            
            # Use 'original_input_text' for analysis
            all_input_texts.extend(dataset.iloc[batch_idx * dataloader.batch_size:(batch_idx + 1) * dataloader.batch_size]['original_input_text'].values)

    # Calculate metrics
    accuracy_priority = accuracy_score(all_priority_labels, all_priority_preds)
    precision_priority, recall_priority, f1_priority, _ = precision_recall_fscore_support(
        all_priority_labels, all_priority_preds, average='weighted', zero_division=1)

    accuracy_resolution = accuracy_score(all_resolution_labels, all_resolution_preds)
    precision_resolution, recall_resolution, f1_resolution, _ = precision_recall_fscore_support(
        all_resolution_labels, all_resolution_preds, average='weighted', zero_division=1)

    # Store predictions and actuals in a DataFrame for analysis
    analysis_df = pd.DataFrame({
        'input_text': all_input_texts,
        'actual_priority': all_priority_labels,
        'predicted_priority': all_priority_preds,
        'actual_resolution': all_resolution_labels,
        'predicted_resolution': all_resolution_preds
    })

    # Save to a CSV file for detailed analysis
    analysis_df.to_csv('prediction_analysis.csv', index=False)

    return {
        'accuracy_priority': accuracy_priority,
        'precision_priority': precision_priority,
        'recall_priority': recall_priority,
        'f1_priority': f1_priority,
        'accuracy_resolution': accuracy_resolution,
        'precision_resolution': precision_resolution,
        'recall_resolution': recall_resolution,
        'f1_resolution': f1_resolution
    }, analysis_df

# Stratified Cross-validation
n_splits = 5
skf = MultilabelStratifiedKFold(n_splits=n_splits)

# Main training loop
for fold, (train_idx, val_idx) in enumerate(skf.split(input_data, target_data)):
    print(f"\nFold {fold+1}/{n_splits}")

    # Split data into train and validation based on indices
    train_fold = dataset.iloc[train_idx]
    val_fold = dataset.iloc[val_idx]

    # Convert data to tensors
    train_features, train_attention_mask, train_priority = convert_to_tensor_with_mask(train_fold, 'input_text_padded', 'priority_encoded')
    train_features, train_attention_mask, train_resolution = convert_to_tensor_with_mask(train_fold, 'input_text_padded', 'resolution_encoded')

    val_features, val_attention_mask, val_priority = convert_to_tensor_with_mask(val_fold, 'input_text_padded', 'priority_encoded')
    val_features, val_attention_mask, val_resolution = convert_to_tensor_with_mask(val_fold, 'input_text_padded', 'resolution_encoded')

    train_dataset = TensorDataset(train_features, train_attention_mask, train_priority, train_resolution)
    val_dataset = TensorDataset(val_features, val_attention_mask, val_priority, val_resolution)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)

    # Reinitialize the model and optimizer for each fold
    model = MultiOutputBERT(num_labels_task1=len(label_encoder_priority.classes_), 
                            num_labels_task2=len(label_encoder_resolution.classes_)).cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.02)

    # Training and evaluation
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} | Fold {fold+1}/{n_splits}")
        train_loss = train_epoch(model, train_dataloader, optimizer, scaler)
        
        eval_metrics, _ = evaluate_epoch_with_storage(model, val_dataloader, val_fold)
        eval_metrics_training, _ = evaluate_epoch_with_storage(model, train_dataloader, train_fold)

        print(f"Epoch {epoch+1}/{epochs} | Fold {fold+1}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Metrics: {eval_metrics_training}")
        print(f"Validation Metrics: {eval_metrics}")
