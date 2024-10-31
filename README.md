# Custom Sentiment Analysis

Custom Sentiment Analysis is a machine learning project that uses the DistilBERT model to classify SMS messages as either spam or ham (non-spam). This project fine-tunes a pre-trained DistilBERT model on a labeled SMS dataset, leveraging transformers and TensorFlow to achieve robust classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Code Structure](#code-structure)
6. [Evaluation](#evaluation)
7. [Future Improvements](#future-improvements)
8. [Requirements](#requirements)

---

## Project Overview

The project aims to accurately classify SMS messages into spam and non-spam categories using a pre-trained DistilBERT model. By tokenizing SMS messages and training the DistilBERT model, we create a reliable classifier that can help filter out spam messages effectively.

### Key Features:
- **Spam Detection**: Classifies SMS messages as spam or ham.
- **Transformer-based Model**: Uses the DistilBERT model for sequence classification.
- **Custom Training and Evaluation**: Fine-tunes DistilBERT on SMS data and evaluates performance using confusion matrix and prediction metrics.

## Installation

To set up and run this project, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ss1910singh/Custom_Sentiment_Analysis.git
   cd Custom_Sentiment_Analysis
   ```

2. **Install dependencies**:

   Ensure you have Python installed (version 3.6+ recommended) and install necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**:

   Make sure the `SMSSpamCollection.csv` dataset is in the root directory of the project.

## Dataset Description

This project uses the **SMSSpamCollection** dataset, which contains labeled SMS messages for spam detection. Each row includes:
- `label`: Either "ham" or "spam"
- `message`: The content of the SMS message

## Methodology

### Step 1: Data Loading and Preprocessing

We load the dataset and convert the labels to binary format (0 for ham, 1 for spam). The data is split into training and testing sets for model evaluation.

```python
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', names=["label", "message"])
X = list(df['message'])
y = list(pd.get_dummies(df['label'], drop_first=True)['spam'])
```

### Step 2: Tokenization

We use the `DistilBertTokenizerFast` from Hugging Face to tokenize each message, ensuring compatibility with the DistilBERT model.

```python
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)
```

### Step 3: Model Training

The DistilBERT model is fine-tuned on the SMS dataset using the TensorFlow `TFTrainer`. The training parameters include:
- **Batch Size**: 8 for training, 16 for evaluation
- **Epochs**: 2
- **Weight Decay**: 0.01 to reduce overfitting
- **Warmup Steps**: 500 for learning rate stability

```python
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
```

## Code Structure

The project follows this structure:

1. **Data Loading**: Load and prepare the SMS dataset.
2. **Tokenization**: Encode SMS messages using the DistilBERT tokenizer.
3. **Model Training**: Fine-tune the DistilBERT model on tokenized SMS data.
4. **Evaluation**: Evaluate the model with the test set using various metrics.

## Evaluation

### Metrics

The model’s performance is evaluated using the following metrics:
- **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives.
- **Accuracy**: Overall performance of the classifier.
- **Precision, Recall, F1 Score**: Additional metrics for a deeper performance analysis.

```python
from sklearn.metrics import confusion_matrix
output = trainer.predict(test_dataset)[1]  # Extract predictions
cm = confusion_matrix(y_test, output)
print("Confusion Matrix:\n", cm)
```

### Model Saving

The trained model is saved for later use or deployment.

```python
trainer.save_model('senti_model')
```

## Future Improvements

Potential enhancements to the project include:
1. **Additional Data**: Using a larger or more diverse dataset could improve the classifier’s robustness.
2. **Hyperparameter Optimization**: Tuning parameters such as batch size, learning rate, and epochs.
3. **Exploring Other Architectures**: Trying other transformer models, like BERT or RoBERTa, to see if they yield higher accuracy.