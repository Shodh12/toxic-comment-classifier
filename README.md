# Toxic Comment Classifier using BERT and Streamlit

This project classifies user-generated comments as either **toxic** or **non-toxic** using a fine-tuned BERT model. It includes a simple and interactive **Streamlit** web application for live text classification.

---

## Features

- 🔍 **Binary classification** (Toxic / Non-Toxic)
- 🧠 Fine-tuned BERT model (`bert-base-uncased`)
- 📊 Model performance visualization (accuracy, F1-score)
- ⚠️ Emoji-based results: ✅ Non-Toxic, ⚠️ Toxic
- 📝 Optional feedback form to collect user inputs

---

## Dataset

- **Source**: [Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- **Labels Used**:
  - `0` → Non-Toxic
  - `1` → Toxic
- **Text Column**: `comment_text`

---

## Model Details

- **Base Model**: `bert-base-uncased`
- **Task**: Binary classification
- **Training Framework**: HuggingFace Transformers (`Trainer`)
- **Epochs**: 4
- **Input Length**: 128 tokens
- **Tokenizer**: BERT Tokenizer

---

## Web App with Streamlit

The app allows users to:
- Enter a comment
- Classify it in real-time
- View classification result with emoji
- Submit feedback (optional)

---

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/Shodh12/toxic-comment-classifier.git
cd toxic-comment-classifier
### 2. Install dependencies

pip install -r requirements.txt
### 3. Run the Streamlit app

streamlit run app.py or streamlit run app.py --logger.level=error

## 📂 File Structure

```text
📁 toxic-comment-classifier/
├── app.py                    # Streamlit app
├── toxic_comment_bert/       # Trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── train_bert_binary.py      # BERT training script
├── jigsaw_toxic_data.csv     # Training dataset
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Usage
### 1. Run the app: streamlit run app.py

### 2. Enter a comment like:

"You are an amazing person!" → ✅ Non-Toxic

"You are so stupid!" → ⚠️ Toxic

## Performance
Sample metrics after 3 epochs :

- Accuracy: 0.97

- F1-score: 0.96

## Requirements
- Python 3.8+

- transformers

- torch

- pandas
  
- streamlit

### Acknowledgments
- HuggingFace for pre-trained models

- Kaggle for the Jigsaw dataset

- Streamlit for the beautiful app interface
