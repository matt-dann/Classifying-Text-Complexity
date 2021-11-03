This work aims to classify the textual complexity of sentences from the Simple English Wikipedia edition. We builta machine learning pipeline that used an evolving set of supervised models aided by processes of feature engineeringand unsupervised learning to classify sentences into whether or not they require simplification. Such a task is usefulin domains such as Simple Wikipedia that aims to present simplified articles that are easier to understand, especiallyfor those who  have “limited abilities”1in comprehensionof the English language.

We utilized the “Predicting Text Difficulty” challenge on Kaggle as a framework for structuring our process andsetting goals of our research. Our primary goal is to reach a 0.8 accuracy score proven possible by previousiteration’s submissions.

Our key findings were the following:
1.Visual exploration of data revealed that the training data labels are perfectly balanced.
2.PCA transformation revealed the training data can be generally separated into simple and complexsentences, but heavy overlap in clusters exists.
3.Topic modeling revealed similar topics throughout the data, regardless of complexity label. Topics alonewere not a significant indicator of text difficulty.
4.Clustering analysis revealed that a 2-cluster KMeans model performed best which correlates to our goal ofbinary classification. Our KMeans labels were 61% accurate compared to the training labels which suggeststhat our model is picking up on a true signal considering the dataset is balanced.
5.We trained multiple machine learning models, andobserved best performance from BERT fine-tunedon our training data & engineered features which achieved an accuracy score of 77.6% on the testdatasetprovided by Kaggle which is the second highestscore on the private leaderboard.
6.Deep learning models were quick to overfit and sensitive to hyper parameter tuning.