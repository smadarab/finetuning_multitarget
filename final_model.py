import pandas as pd
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# Load the full dataset of 20k records
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

# Calculate class weights
class_weights_priority = compute_class_weight('balanced', classes=np.unique(dataset['priority_encoded']), y=dataset['priority_encoded'])
class_weights_resolution = compute_class_weight('balanced', classes=np.unique(dataset['resolution_encoded']), y=dataset['resolution_encoded'])

# Convert class weights to tensors and move to GPU
class_weights_priority = torch.tensor(class_weights_priority, dtype=torch.float).cuda()
class_weights_resolution = torch.tensor(class_weights_resolution, dtype=torch.float).cuda()

# Define loss functions with class weights
loss_fn_priority = nn.CrossEntropyLoss(weight=class_weights_priority)
loss_fn_resolution = nn.CrossEntropyLoss(weight=class_weights_resolution)

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Convert columns to tensors with attention masks
def convert_to_tensor_with_mask(df, feature_col, target_col):
    features = torch.tensor(df[feature_col].tolist(), dtype=torch.long).cuda()
    attention_mask = (features != 0).long().cuda()
    targets = torch.tensor(df[target_col].tolist(), dtype=torch.long).cuda()
    return features, attention_mask, targets

train_features, train_attention_mask, train_priority = convert_to_tensor_with_mask(train_df, 'input_text_padded', 'priority_encoded')
train_features, train_attention_mask, train_resolution = convert_to_tensor_with_mask(train_df, 'input_text_padded', 'resolution_encoded')

val_features, val_attention_mask, val_priority = convert_to_tensor_with_mask(val_df, 'input_text_padded', 'priority_encoded')
val_features, val_attention_mask, val_resolution = convert_to_tensor_with_mask(val_df, 'input_text_padded', 'resolution_encoded')

train_dataset = TensorDataset(train_features, train_attention_mask, train_priority, train_resolution)
val_dataset = TensorDataset(val_features, val_attention_mask, val_priority, val_resolution)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)

# Define a custom multi-output BERT model with attention mask and dropout
class MultiOutputBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2, dropout_rate=0.3):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        return logits_task1, logits_task2

# Load the custom model and move to GPU
model = MultiOutputBERT(num_labels_task1=len(label_encoder_priority.classes_), 
                        num_labels_task2=len(label_encoder_resolution.classes_),
                        dropout_rate=0.3).cuda()

# Define optimizer with weight decay for L2 regularization
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Mixed precision training scaler
scaler = GradScaler()

# Early stopping parameters
early_stopping_patience = 5
min_val_loss = np.inf
patience_counter = 0

# Training loop with early stopping
def train_epoch(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in dataloader:
        b_input_ids, b_attention_mask, b_priority, b_resolution = batch
        
        optimizer.zero_grad()
        
        with autocast():
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
        
            loss_priority = loss_fn_priority(logits_priority, b_priority)
            loss_resolution = loss_fn_resolution(logits_resolution, b_resolution)
        
            loss = loss_priority + loss_resolution
        
        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluation loop
def evaluate_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    all_priority_preds = []
    all_resolution_preds = []
    all_priority_labels = []
    all_resolution_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_priority, b_resolution = batch
            
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
            
            loss_priority = loss_fn_priority(logits_priority, b_priority)
            loss_resolution = loss_fn_resolution(logits_resolution, b_resolution)
            total_loss += loss_priority.item() + loss_resolution.item()

            all_priority_preds.extend(logits_priority.argmax(dim=1).cpu().numpy())
            all_resolution_preds.extend(logits_resolution.argmax(dim=1).cpu().numpy())
            all_priority_labels.extend(b_priority.cpu().numpy())
            all_resolution_labels.extend(b_resolution.cpu().numpy())
    
    accuracy_priority = accuracy_score(all_priority_labels, all_priority_preds)
    precision_priority, recall_priority, f1_priority, _ = precision_recall_fscore_support(
        all_priority_labels, all_priority_preds, average='weighted', zero_division=1)

    accuracy_resolution = accuracy_score(all_resolution_labels, all_resolution_preds)
    precision_resolution, recall_resolution, f1_resolution, _ = precision_recall_fscore_support(
        all_resolution_labels, all_resolution_preds, average='weighted', zero_division=1)

    avg_loss = total_loss / len(dataloader)

    return {
        'val_loss': avg_loss,
        'accuracy_priority': accuracy_priority,
        'precision_priority': precision_priority,
        'recall_priority': recall_priority,
        'f1_priority': f1_priority,
        'accuracy_resolution': accuracy_resolution,
        'precision_resolution': precision_resolution,
        'recall_resolution': recall_resolution,
        'f1_resolution': f1_resolution
    }

# Training and evaluation with early stopping
epochs = 20
for epoch in range(epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, scaler)
    eval_metrics = evaluate_epoch(model, val_dataloader)
    eval_metrics_train = evaluate_epoch(model, train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"train Priority - Accuracy: {eval_metrics_train['accuracy_priority']:.4f}, Precision: {eval_metrics_train['precision_priority']:.4f}, Recall: {eval_metrics_train['recall_priority']:.4f}, F1: {eval_metrics_train['f1_priority']:.4f}")
    print(f"train Resolution - Accuracy: {eval_metrics_train['accuracy_resolution']:.4f}, Precision: {eval_metrics_train['precision_resolution']:.4f}, Recall: {eval_metrics_train['recall_resolution']:.4f}, F1: {eval_metrics_train['f1_resolution']:.4f}")

   
    print(f"Validation Loss: {eval_metrics['val_loss']:.4f}")
    print(f"Priority - Accuracy: {eval_metrics['accuracy_priority']:.4f}, Precision: {eval_metrics['precision_priority']:.4f}, Recall: {eval_metrics['recall_priority']:.4f}, F1: {eval_metrics['f1_priority']:.4f}")
    print(f"Resolution - Accuracy: {eval_metrics['accuracy_resolution']:.4f}, Precision: {eval_metrics['precision_resolution']:.4f}, Recall: {eval_metrics['recall_resolution']:.4f}, F1: {eval_metrics['f1_resolution']:.4f}")

    # Early stopping based on validation loss
    if eval_metrics['val_loss'] < min_val_loss:
        min_val_loss = eval_metrics['val_loss']
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        print("Validation loss improved. Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load the best model after early stopping
model.load_state_dict(torch.load('best_model.pth'))

# Final evaluation on the test dataset
test_features, test_attention_mask, test_priority = convert_to_tensor_with_mask(test_df, 'input_text_padded', 'priority_encoded')
test_features, test_attention_mask, test_resolution = convert_to_tensor_with_mask(test_df, 'input_text_padded', 'resolution_encoded')
test_dataset = TensorDataset(test_features, test_attention_mask, test_priority, test_resolution)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

test_metrics = evaluate_epoch(model, test_dataloader)
print(f"Final Test Metrics:")
print(f"Priority - Accuracy: {test_metrics['accuracy_priority']:.4f}, Precision: {test_metrics['precision_priority']:.4f}, Recall: {test_metrics['recall_priority']:.4f}, F1: {test_metrics['f1_priority']:.4f}")
print(f"Resolution - Accuracy: {test_metrics['accuracy_resolution']:.4f}, Precision: {test_metrics['precision_resolution']:.4f}, Recall: {test_metrics['recall_resolution']:.4f}, F1: {test_metrics['f1_resolution']:.4f}")
