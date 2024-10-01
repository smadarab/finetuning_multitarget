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
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from imblearn.combine import SMOTEENN 

# Load your dataset
file_path = 'F:\\finetunining sample\\combined_dataset.csv'
dataset = pd.read_csv(file_path)
# dataset = dataset.iloc[:10000]

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing steps
dataset['input_text'] = dataset['input_text'].str.lower()
dataset['input_text'] = dataset['input_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', '', x))
dataset['input_text_tokens'] = dataset['input_text'].apply(lambda x: tokenizer.tokenize(x))
max_length = 128
dataset['input_text_padded'] = dataset['input_text_tokens'].apply(
    lambda x: tokenizer.convert_tokens_to_ids(x)[:max_length] + [0] * (max_length - len(x))
)

# Encode the target variables
label_encoder_priority = LabelEncoder()
label_encoder_resolution = LabelEncoder()

dataset['priority_encoded'] = label_encoder_priority.fit_transform(dataset['priority_binned'])
dataset['resolution_encoded'] = label_encoder_resolution.fit_transform(dataset['bug_resolution_time'])

val = dataset[["input_text_padded","priority_encoded","resolution_encoded"]]
index_priority = list(val[val['priority_encoded'] ==1].index)[:2335] + list(val[val['priority_encoded'] ==2].index)[:2335] + list(val[val['priority_encoded'] ==3].index)[:2335]
dataset = val.iloc[index_priority]


# Split dataset into train, validation, and test sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[['priority_encoded', 'resolution_encoded']])
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df[['priority_encoded', 'resolution_encoded']])

# Convert columns to tensors
def convert_to_tensor_with_mask(df, feature_col, target_col):
    features = torch.tensor(df[feature_col].tolist(), dtype=torch.long).cuda()
    attention_mask = (features != 0).long().cuda()
    targets = torch.tensor(df[target_col].tolist(), dtype=torch.long).cuda()
    return features, attention_mask, targets

train_features, train_attention_mask, train_priority = convert_to_tensor_with_mask(train_df, 'input_text_padded', 'priority_encoded')
train_features, train_attention_mask, train_resolution = convert_to_tensor_with_mask(train_df, 'input_text_padded', 'resolution_encoded')

val_features, val_attention_mask, val_priority = convert_to_tensor_with_mask(val_df, 'input_text_padded', 'priority_encoded')
val_features, val_attention_mask, val_resolution = convert_to_tensor_with_mask(val_df, 'input_text_padded', 'resolution_encoded')

# SMOTE for class imbalance
X_train = np.array(train_df['input_text_padded'].tolist())
y_train_priority = np.array(train_df['priority_encoded'])
y_train_resolution = np.array(train_df['resolution_encoded'])

print(f"Original X_train size: {X_train.shape}")
print(f"Original y_train_priority size: {y_train_priority.shape}")
print(f"Original y_train_resolution size: {y_train_resolution.shape}")
# Apply SMOTE to each target separately
smote_priority = SMOTE(random_state=42)
X_train_smote, y_train_priority_smote = smote_priority.fit_resample(X_train, y_train_priority)

smote_resolution = SMOTE(random_state=42)
X_train_smote, y_train_resolution_smote = smote_resolution.fit_resample(X_train, y_train_resolution)


print(f"SMOTE X_train_smote size: {X_train_smote.shape}")
print(f"SMOTE y_train_priority_smote size: {y_train_priority_smote.shape}")
print(f"SMOTE y_train_resolution_smote size: {y_train_resolution_smote.shape}")


# Convert to tensors
train_features_smote = torch.tensor(X_train_smote, dtype=torch.long).cuda()
train_attention_mask_smote = (train_features_smote != 0).long().cuda()
train_priority_smote = torch.tensor(y_train_priority_smote, dtype=torch.long).cuda()
train_resolution_smote = torch.tensor(y_train_resolution_smote, dtype=torch.long).cuda()

train_dataset = TensorDataset(train_features_smote, train_attention_mask_smote, train_priority_smote, train_resolution_smote)
val_dataset = TensorDataset(val_features, val_attention_mask, val_priority, val_resolution)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)

# Define Focal Loss with alpha
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        probs = torch.softmax(inputs, dim=1)
        target_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze()
        focal_loss = (1 - target_probs) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Compute class weights and initialize Focal Loss
alpha_priority = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_df['priority_encoded']), y=train_df['priority_encoded']), dtype=torch.float).cuda()
alpha_resolution = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_df['resolution_encoded']), y=train_df['resolution_encoded']), dtype=torch.float).cuda()

focal_loss_priority = FocalLoss(gamma=2.0, alpha=alpha_priority)
focal_loss_resolution = FocalLoss(gamma=2.0, alpha=alpha_resolution)

# Define custom multi-output BERT model
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

# Load model and optimizer
model = MultiOutputBERT(num_labels_task1=len(label_encoder_priority.classes_), 
                        num_labels_task2=len(label_encoder_resolution.classes_)).cuda()
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = GradScaler()

# Training loop
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

# Training process with early stopping
epochs = 30
best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_epoch_focal(model, train_dataloader, optimizer, scaler)
    print(f"Training loss: {train_loss}")
    
    val_metrics = evaluate_epoch(model, val_dataloader)
    train_metrics = evaluate_epoch(model, train_dataloader)

    print(f"train_metrics accuracy (Priority): {train_metrics['accuracy_priority']}")
    print(f"train_metrics F1 Score (Priority): {train_metrics['f1_priority']}")
    print(f"train_metrics accuracy (Resolution): {train_metrics['accuracy_resolution']}")
    print(f"train_metrics F1 Score (Resolution): {train_metrics['f1_resolution']}")
    
    print(f"Validation accuracy (Priority): {val_metrics['accuracy_priority']}")
    print(f"Validation F1 Score (Priority): {val_metrics['f1_priority']}")
    print(f"Validation accuracy (Resolution): {val_metrics['accuracy_resolution']}")
    print(f"Validation F1 Score (Resolution): {val_metrics['f1_resolution']}")
    
    # Early stopping
    if val_metrics['f1_priority'] < best_val_loss:
        best_val_loss = val_metrics['f1_priority']
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break
