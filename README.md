# BERT News Classification Project

This project utilizes the BERT model for classifying news articles into one of four categories using the AG News dataset. The implementation includes tokenization, data preparation, and training of a sequence classification model using TensorFlow.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Tokenization](#tokenization)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- Transformers
- Datasets

## Installation

Install the required libraries using pip:

```bash
pip install tensorflow
pip install transformers
pip install datasets
```

## Dataset

We are using the AG News dataset, which can be loaded using the `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
```

## Tokenization

We use the `BertTokenizer` to tokenize the text data. Tokenization converts text into input IDs and attention masks required by the BERT model.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## Data Preparation

Convert the tokenized dataset into TensorFlow datasets for training and testing.

```python
import tensorflow as tf

def convert_to_tf_dataset(tokenized_dataset, batch_size=16):
    def gen():
        for i in range(len(tokenized_dataset)):
            yield({
                'input_ids': tokenized_dataset[i]['input_ids'],
                'attention_mask': tokenized_dataset[i]['attention_mask']
            }, tokenized_dataset[i]['label'])
    
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'input_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(128,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    ).batch(batch_size)

train_dataset = convert_to_tf_dataset(tokenized_dataset['train'])
test_dataset = convert_to_tf_dataset(tokenized_dataset['test'])
```

## Model Training

Load the BERT model for sequence classification and compile it with an optimizer, loss function, and evaluation metric.

```python
from transformers import TFRemBertForSequenceClassification

model = TFRemBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=test_dataset, epochs=4)
```

## Evaluation

Evaluate the model on the test dataset to measure its performance.

```python
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
```

## Results

After training the model, you should see the training and validation accuracy improving over the epochs. The final evaluation on the test set will provide the accuracy and loss, indicating the model's performance on unseen data.
