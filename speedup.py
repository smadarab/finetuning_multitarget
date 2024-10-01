import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Load your dataset
file_path = 'F:\\finetunining sample\\updated_combined_data.csv'
dataset = pd.read_csv(file_path)
# dataset = dataset.iloc[:10000]

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Step 3: Tokenization and conversion to input IDs
def tokenize_and_pad(text, tokenizer, max_length=128):
    # Tokenize and convert to input IDs
    tokens = tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_length)
    return torch.tensor(tokens).to('cuda')  # Move to GPU


# Step 1: Lowercasing
dataset['input_text'] = dataset['input_text'].str.lower()

# Step 2: Remove unnecessary special characters but keep numbers and meaningful punctuation
dataset['input_text'] = dataset['input_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', '', x))

# # Step 3: Tokenization
# dataset['input_text_tokens'] = dataset['input_text'].apply(lambda x: tokenizer.tokenize(x))

# # Step 4: Padding/Truncation
# max_length = 128
# dataset['input_text_padded'] = dataset['input_text_tokens'].apply(
#     lambda x: tokenizer.convert_tokens_to_ids(x)[:max_length] + [0] * (max_length - len(x))
# )

# Assuming the dataset is a pandas DataFrame
dataset['input_text_padded'] = dataset['input_text'].apply(lambda x: tokenize_and_pad(x, tokenizer))


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

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability for the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, labels.data.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Initialize Focal Loss with class weights
alpha_priority = torch.tensor(compute_class_weight('balanced', classes=np.unique(dataset['priority_encoded']), y=dataset['priority_encoded']), dtype=torch.float).cuda()
alpha_resolution = torch.tensor(compute_class_weight('balanced', classes=np.unique(dataset['resolution_encoded']), y=dataset['resolution_encoded']), dtype=torch.float).cuda()

focal_loss_priority = FocalLoss(gamma=2.0, alpha=alpha_priority)
focal_loss_resolution = FocalLoss(gamma=2.0, alpha=alpha_resolution)

# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[['priority_encoded', 'resolution_encoded']])
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df[['priority_encoded', 'resolution_encoded']])

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
    def __init__(self, num_labels_task1, num_labels_task2):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.4)  # Adding dropout to avoid overfitting
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        logits_task1 = self.classifier_task1(pooled_output)
        logits_task2 = self.classifier_task2(pooled_output)
        return logits_task1, logits_task2

# Load the custom model and move to GPU
model = MultiOutputBERT(num_labels_task1=len(label_encoder_priority.classes_), 
                        num_labels_task2=len(label_encoder_resolution.classes_)).cuda()

# Define optimizer with weight decay for regularization
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Mixed precision training scaler
scaler = GradScaler()

# Training loop using Focal Loss
def train_epoch_focal(model, dataloader, optimizer, scaler):
    model.train()
    total_loss = 0
    for batch in dataloader:
        b_input_ids, b_attention_mask, b_priority, b_resolution = batch
        
        optimizer.zero_grad()
        
        with autocast():
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
        
            loss_priority = focal_loss_priority(logits_priority, b_priority)
            loss_resolution = focal_loss_resolution(logits_resolution, b_resolution)
        
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
    all_priority_preds = []
    all_resolution_preds = []
    all_priority_labels = []
    all_resolution_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_attention_mask, b_priority, b_resolution = batch
            
            logits_priority, logits_resolution = model(b_input_ids, attention_mask=b_attention_mask)
            
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

    return {
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
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(epochs):
    train_loss = train_epoch_focal(model, train_dataloader, optimizer, scaler)
    eval_metrics = evaluate_epoch(model, val_dataloader)
    eval_metrics_train = evaluate_epoch(model, train_dataloader)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Train Metrics: {eval_metrics_train}")
    print(f"Validation Metrics: {eval_metrics}")

    # Early stopping
    if eval_metrics['f1_priority'] + eval_metrics['f1_resolution'] > best_val_loss:
        best_val_loss = eval_metrics['f1_priority'] + eval_metrics['f1_resolution']
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break
