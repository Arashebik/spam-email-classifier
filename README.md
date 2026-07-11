# SIMPLE-SPAM-CLASSIFIER

## Table of Content


* [Overview](#overview)
* [Project Structure](#project-structure)
* [Features](#features)
* [How To Use](#how_to_use)
* [Future Improvments](#future_improvements)

## Overview

This project is a simple ML model for classifying spam and not spam(ham) emails. Sometimes emails 
contain some spam content and this project diagnose whether an email is spam or not.

I used Naive Bayes algorithm to train this model. This algorithm is a very popular algorithm for
text and spam classifying.

## Project Structer

```Project Pipeline

                    RAW DATASET
         (SpamAssassin Public Corpus)
                      │
                      ▼
             create_dataset.py
                      │
                      ▼
                 emails.csv
                      │
                      ▼
           load_and_clean_dataset()
                      │
                      ▼
             Train / Test Split
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          Training Set      Testing Set
              │                 │
              └────────┬────────┘
                       ▼
        ┌─────────────────────────────┐
        │     Scikit-Learn Pipeline   │
        │                             │
        │  CountVectorizer            │
        │           │                 │
        │           ▼                 │
        │  Multinomial Naive Bayes    │
        └───────────┬─────────────────┘
                    │
                    ▼
           Model Evaluation
                    │
                    ▼
        spam_classifier.pkl
                    │
                    ▼
             predict.py
                    │
                    ▼
          Spam / Ham Prediction

```

## Dataset

This project uses the **Apache SpamAssassin Public Corpus**, a well-known public dataset for spam email classification research.

The original dataset consists of thousands of raw email files organized into separate folders for spam and ham emails.

To simplify the machine learning workflow, the raw emails are converted into a single `emails.csv` file using `create_dataset.py`. During this process, only the email subject and body are extracted, while unnecessary metadata is removed.

*Dataset source* :
 - https://spamassassin.apache.org/old/publiccorpus/

**Note** : This project was built for educational purposes to demonstrate a complete machine learning workflow using a classic spam classification dataset. The SpamAssassin corpus is relatively old, so the trained model may not generalize well to modern spam campaigns (e.g., cryptocurrency scams or smartphone giveaway messages).

## Features

The project consists of four main components:

- **create_dataset.py**
- **dataloader.py**
- **train.py**
- **predict.py**

### create_dataset.py

The original SpamAssassin dataset is distributed as thousands of raw email files. Working directly with these files is inconvenient for machine learning, so this script converts them into a single CSV dataset.

Responsibilities:

- Parse raw email files
- Extract only the email subject and body
- Remove unnecessary metadata
- Remove invalid or empty emails
- Create `emails.csv`

### dataloader.py

This module is responsible for loading and performing basic cleaning on the dataset before training.

Responsibilities:

- Load `emails.csv`
- Remove duplicate emails
- Remove missing values
- Standardize labels (`spam` and `ham`)
- Return a clean dataset for training

### train.py

This file trains the spam classifier using Scikit-Learn.

Training pipeline:

1. Load the dataset
2. Split the dataset into training and testing sets
3. Build a Scikit-Learn pipeline
4. Convert text into numerical features using `CountVectorizer`
5. Train a `Multinomial Naive Bayes` classifier
6. Evaluate the model
7. Save the trained model as `spam_classifier.pkl`

### predict.py

This script allows users to classify new emails using the trained model.

Responsibilities:

- Load the trained model
- Accept email text from the user
- Predict whether the email is **Spam** or **Ham**
- Display the prediction in the terminal

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/simple-spam-classifier.git
cd simple-spam-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset (optional)

Download the Apache SpamAssassin Public Corpus (or any compatible raw emails) and extract it into:
```
data/raw
```

### 4. Create the Dataset
```bash
python src/create_dataset.py
```

### 5. Train the model
```bash
python src/train.py
```

### 6. Predict new emails
```bash
python src/train.py
```

## Future Improvements

This project was intentionally kept simple for educational purposes. Some possible improvements include:

- Replace CountVectorizer with TF-IDF
- Compare multiple machine learning algorithms
- Find a better dataset to generilize
- Perform hyperparameter tuning
- Add unit tests
- Build a web interface using FastAPI