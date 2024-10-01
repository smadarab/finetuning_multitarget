import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader

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

# Define compute_metrics_function
def compute_metrics_function(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions.argmax(axis=-1)  # Convert logits to predicted labels
    labels = eval_pred.label_ids

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Load a pre-trained model and modify for multi-output classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',                # Directory to save model checkpoints
    evaluation_strategy='epoch',           # Evaluate at the end of each epoch
    learning_rate=2e-5,                    # Learning rate
    per_device_train_batch_size=32,        # Batch size for training
    per_device_eval_batch_size=32,         # Batch size for evaluation
    num_train_epochs=3,                    # Number of training epochs
    weight_decay=0.01,                     # Weight decay for regularization
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_function,  # Define a function for evaluation metrics
)

# Train the model
try:
    trainer.train()
except TypeError as e:
    print(f"An error occurred: {e}")
