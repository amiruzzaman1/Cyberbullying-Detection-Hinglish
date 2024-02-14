# Cyberbullying Detection App (Hinglish) with Streamlit GUI

## Overview

This repository contains a Cyberbullying Detection App specifically designed for the Hinglish language, which is a combination of Hindi and English. The application utilizes various machine learning and deep learning algorithms fine-tuned for natural language processing and classification tasks. The primary goal is to identify instances of cyberbullying in Hinglish text and classify them into three distinct categories: Non-Abusive, Generalized (NAG), Abusive, Cursing (CAG), and Obscene, Abusive, Graphic (OAG).

## Methodology

### A. Preprocessing

The initial step involves preparing the data for training and evaluation. Text preprocessing techniques such as tokenization, stemming or lemmatization, and the removal of stop words and unnecessary letters are employed. The processed data is then split into training and testing sets.

### B. Feature Representation

Different algorithms require specific feature representation methods. Classic algorithms like Random Forest (RF), Support Vector Machine (SVM), Decision Tree (DT), and Logistic Regression (LR) may use bag-of-words or TF-IDF representation. In contrast, deep learning models like BERT, RNN, ANN, CNN, and BiLSTM benefit from word embeddings or contextual embeddings to capture semantic connections within the text.

### C. Models

Various machine learning and deep learning models are implemented, each serving a specific purpose in cyberbullying detection:

- **BERT (Bidirectional Encoder Representations from Transformers):** Utilized for pre-training on large corpora, collecting contextual information from both left and right word contexts.

- **Random Forest (RF):** An ensemble learning approach constructing decision trees, excelling in high-dimensional data and providing feature importance rankings.

- **Support Vector Machine (SVM):** A robust classification technique seeking hyperplanes to categorize data points, adaptable to linear and non-linear correlations using kernel algorithms.

- **Decision Tree (DT):** A tree-like architecture with interpretable decision nodes, useful for transparent decision-making.

- **Multi-Layer Perceptron (MLP):** A feedforward neural network recording complicated connections, requiring careful modification to minimize overfitting.

- **Recurrent Neural Network (RNN):** Collects sequential information, overcoming gradient difficulties with versions like Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM).

- **Artificial Neural Network (ANN):** The core of deep learning, adaptable to many tasks.

- **Convolutional Neural Network (CNN):** Commonly used for image recognition but demonstrates good results in text classification by identifying local patterns.

- **Naive Bayes (NB):** Utilizes Bayes' theorem with the "naive" assumption of feature independence, performing well in text classification.

- **Logistic Regression (LR):** A linear model used for binary or multiclass classification, balancing simplicity, robustness, and interpretability.

### D. Evaluation

The performance of each algorithm is assessed using various evaluation metrics such as accuracy, precision, recall, and F1-Score.

## Hinglish Dataset Classification Report

The Hinglish dataset categorizes cyberbullying into three unique groups:

1. **NAG (Non-Abusive, Generalized)**
2. **CAG (Abusive, Cursing)**
3. **OAG (Obscene, Abusive, Graphic)**

### Classification Report

| Algorithm     | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Logistic Reg. | 0.60     | 0.61      | 0.58   | 0.59     |
| SVM           | 0.60     | 0.60      | 0.60   | 0.59     |
| Random Forest | 0.58     | 0.58      | 0.58   | 0.58     |
| BiLSTM        | 0.56     | 0.56      | 0.56   | 0.56     |
| Naive Bayes   | 0.56     | 0.61      | 0.56   | 0.54     |

### Model Analysis

1. **Logistic Regression:** Accuracy of 60.19%, with varying performance across different categories.

2. **Support Vector Machine (SVM):** Achieved an accuracy of 60%, displaying balanced accuracy and recall for CAG and OAG but reduced recall for NAG.

3. **Random Forest:** Exhibited an accuracy of 58%, showing nuanced performance across categories, particularly notable precision for OAG.

4. **Bidirectional LSTM:** Produced an overall accuracy of approximately 55.85%, demonstrating reasonably balanced predictive ability across classes.

5. **Naive Bayes:** Achieved an accuracy of around 56.30%, displaying nuanced performance in distinguishing between different forms of Hinglish text.

## Streamlit GUI for Cyberbullying Detection (Hinglish)

To use the Cyberbullying Detection App with Streamlit GUI:

1. Enter a text for cyberbullying detection in the provided input field.
2. View the prediction result and identified cyberbullying type (NAG, CAG, or OAG).
3. Explore bad words present in the text and the filtered output.

[Access the Cyberbullying Detection App (Hinglish) here](https://amiruzzaman-cbhindi.hf.space/#cyberbullying-detection-app-hinglish)

