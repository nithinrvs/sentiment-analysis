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

## 6. Advanced Deep Learning Models

### LSTM Model

1. **Model Architecture:** The LSTM model consists of LSTM layers with dropout regularization and dense layers for classification.
2. **Training Details:**
   - **Epochs:** Trained over 20 epochs to optimize learning and model convergence.
   - **Training Accuracy:** Achieved 51% accuracy on the training data, indicating moderate learning from the dataset.
   - **Validation Accuracy:** Attained 47.5% accuracy on the validation set, which is lower compared to the Bi-Directional LSTM model, suggesting less effective performance.
3.. **Evaluation:** The LSTM model showed lower accuracy compared to the Bi-Directional LSTM, highlighting potential limitations in capturing the complexity of sentiment analysis in Telugu text.

### Bi-Directional LSTM Model

1. **Model Architecture:** The model comprises Bi-Directional LSTM layers with dropout regularization to prevent overfitting. This architecture allows the model to capture contextual information from both past and future sequences, enhancing its understanding of Telugu reviews' sentiment.
2. **Training Details:**
   - **Epochs:** Trained over 10 epochs, optimizing the learning process.
   - **Training Accuracy:** Achieved 100% accuracy on the training data, indicating a strong ability to learn from the dataset.
   - **Validation Accuracy:** Attained 80% accuracy on the validation set, indicating good generalization but showing signs of overfitting due to the large gap between training and validation accuracies.
3. **Evaluation:** While the model performed well in terms of accuracy, the significant difference between training and validation accuracies suggests potential overfitting issues that need to be addressed. Future work could focus on implementing techniques such as cross-validation, early stopping, and data augmentation to mitigate overfitting and improve generalization.


