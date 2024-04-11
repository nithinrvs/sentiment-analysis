# SENTIMENT ANALYSIS OF TELUGU PRODUCT REVIEWS WITH MACHINE LEARNING

## 1. Sentiment Analysis

Sentiment analysis is a natural language processing (NLP) technique that involves determining the sentiment or opinion expressed in text data. It categorizes the text into positive, negative, or neutral sentiments based on the language's emotional tone.

## 2. Corpus Used

The project utilizes a part of the Sentiraama corpus created by IIIT Hyderabad. Specifically, electronic product reviews are extracted from this corpus for sentiment analysis. The reviews include both positive and negative sentiments, providing a balanced dataset for training and evaluation.

## 3. Methodology

### Data Preprocessing and Feature Engineering
- Since the data is collected from sentiraama corpus, Data was already pre-processed, ready to train. 

### Model Selection
The project explores various machine learning algorithms for sentiment classification based on Telugu product reviews. Here are the results of model training and evaluation:

| Model                | Accuracy | Recall | F1-Score |
|----------------------|----------|--------|----------|
| Decision Tree        | 50%      | 52%    | 51%      |
| Logistic Regression  | 75%      | 76%    | 75%      |
| Support Vector Machine (SVC) | 42%      | 54%    | 36%      |
| Random Forest        | 57%      | 65%    | 57%      |
| Naive Bayes          | 57%      | 59%    | 57%      |

These results demonstrate the performance of each algorithm in classifying Telugu product reviews based on sentiment, with Logistic Regression achieving the highest accuracy of 75% among the models evaluated.



## 4. Hyperparameter Tuning
Hyperparameter tuning is performed for two selected models:
- Random Forest: Tuning parameters such as number of estimators, maximum depth, and minimum samples split.
- Logistic Regression: Tuning regularization parameters and optimization algorithms.

### Random Forest
- Best parameters: {n_estimators: 300, max_depth: 5, min_samples_split: 2, min_samples_leaf: 2, max_features: 'log2'}

### Logistic Regression
- Best parameters: {C: 0.1, penalty: 'l2', solver: 'liblinear'}

## 5. Model Evaluation and Performance

### Random Forest
- Accuracy: 57%

### Logistic Regression
- Accuracy: 72%


