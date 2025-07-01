# Quora Question Parity Analyzer

## 📌 Project Overview
The **Quora Question Parity Analyzer** aims to detect duplicate questions on Quora, enhancing content quality by identifying semantically similar questions. This project leverages a combination of traditional machine learning and deep learning techniques to achieve high accuracy in duplicate detection.

**🔗 Dataset:**  
[Quora Question Pairs on Kaggle](https://www.kaggle.com/competitions/quora-question-pairs/)

---

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Data Preprocessing](#-data-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Modeling](#-modeling)
- [Results](#-results)
- [Installation](#-installation)

---

## ✨ Features

- **Text Preprocessing**: Tokenization, stop words removal, and stemming.  
- **Vectorization**: Text-to-vector conversion using Word2Vec.  
- **Custom Feature Generation**:
  - Average word length
  - Word count
  - Average sentence length
  - Fog index
  - Complex word count  
- **Oversampling**: SMOTE for handling class imbalance.  
- **Modeling**: Implementation of `RandomForestClassifier`, `XGBoost`, `SimpleRNN`, and `Bidirectional LSTM`.  
- **Model Persistence**: Model weights saved using `joblib`.

---

## 🧹 Data Preprocessing

We perform extensive text preprocessing to clean and prepare the data:

- **Tokenization**: Splitting text into individual words or tokens.  
- **Stop Words Removal**: Removing common, less meaningful words like “and”, “the”, “is”.  
- **Stemming**: Reducing words to their base form (e.g., “running” → “run”).

---

## 🛠 Feature Engineering

We generate a variety of handcrafted features to enrich the data:

### 🔹 Word2Vec Embeddings:
- Window size: `5`
- Minimum word count: `2`
- Vector size: `30`

### 🔹 TextBlob-Based Features:
- **Average Word Length**
- **Word Count**
- **Average Sentence Length**
- **Fog Index** (readability metric)
- **Complex Word Count** (words with ≥3 syllables)

---

## 🧠 Modeling

We implemented and evaluated several models to identify the best-performing approach:

- **Random Forest Classifier**
  - Accuracy: **85.73%**
  
- **XGBoost**
  - Accuracy: **80.60%**

- **Deep Learning Models**:
  - **SimpleRNN**
    - Loss function: `binary_crossentropy`
    - Optimizer: `Adam`
  - **Bidirectional LSTM**
    - Accuracy: **96.80%**

---

## 📈 Results

The **Bidirectional LSTM** model achieved the best performance with an accuracy of **96.80%**, showcasing its ability to capture semantic meaning and sequential dependencies in the question pairs.

---

## ⚙️ Installation

Follow these steps to set up the project locally:

```bash
# Clone the repository
git clone https://github.com/arnoah55/quora_question_parity_analyzer.git

# Navigate into the project directory
cd quora_question_parity_analyzer

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
