Twitter Sentiment Analysis
Multi-Class Emotion Detection from Tweets

OVERVIEW
--------
A machine learning pipeline that classifies tweets into six emotions:
sadness, joy, love, anger, fear, and surprise.

Dataset: Emotion Dataset from Kaggle (20,000 tweets)
https://www.kaggle.com/datasets/parulpandey/emotion-dataset

Best Model: Linear SVM with 89.1% test accuracy


```plaintext
Twitter-Sentiment-Analysis/
│
├── Data/
│   ├── tweets_training.csv (16,000 samples)
│   │
│   ├── tweets_validation.csv (2,000 samples)
│   │
│   └── tweets_test.csv (2,000 samples)
│
├── Notebook/
│   └── twitter_sentiment_analysis.ipynb
│
├── Report/
│   └── Twitter_Sentiment_Analysis_Report.pdf
│
└── README.md

PIPELINE
--------
1. EDA: label distribution, text length analysis, word frequency charts
2. Preprocessing: lowercasing, URL/mention removal, stopwords, lemmatisation
3. Feature extraction: TF-IDF (10,000 features, unigrams + bigrams)
4. Model training: Logistic Regression, Linear SVM, Naive Bayes,
   Random Forest, Decision Tree, KNN
5. Evaluation: accuracy, precision, recall, F1, confusion matrices


RESULTS
-------
Model                  Val Acc    Test Acc
Linear SVM             0.8910     0.8910
Logistic Regression    0.8850     0.8815
Random Forest          0.8570     0.8470
Multinomial NB         0.8410     0.8280
Decision Tree          0.8175     0.8175
KNN (cosine, k=7)      0.7735     0.7690


REQUIREMENTS
------------
pandas, numpy, matplotlib, seaborn
nltk, scikit-learn, wordcloud

Install: pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
