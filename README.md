# Predicting Text Difficulty
SIADS 695 Milestone II Project<br>
by James Mete & Matt Dannheisser

This work aims to classify the textual complexity of sentences from the Simple English Wikipedia edition. We built a machine learning pipeline that used an evolving set of supervised models aided by processes of feature engineering and unsupervised learning to classify sentences into whether or not they require simplification. Such a task is useful in domains such as Simple Wikipedia that aims to present simplified articles that are easier to understand, especially for those who  have a limited abilities in comprehension of the English language.

We utilized the [Predicting Text Difficulty](https://www.kaggle.com/c/umich-siads-695-predicting-text-difficulty) challenge on Kaggle as a framework for structuring our process and setting goals of our research. Our primary goal is to reach a 0.8 accuracy score proven possible by previous iterations submissions.

Our key findings were the following:
<ol>
<li>Visual exploration of data revealed that the training data labels are perfectly balanced.</li>

<li>PCA transformation revealed the training data can be generally separated into simple and complex sentences, but heavy overlap in clusters exists.</li>

<li>Topic modeling revealed similar topics throughout the data, regardless of complexity label. Topics alone were not a significant indicator of text difficulty.</li>

<li>Clustering analysis revealed that a 2-cluster KMeans model performed best which correlates to our goal of binary classification. Our KMeans labels were 61% accurate compared to the training labels which suggests that our model is picking up on a true signal considering the dataset is balanced.</li>

<li>We trained multiple machine learning models, and observed best performance from BERT fine-tunedon our training data & engineered features which achieved an accuracy score of 77.6% on the test dataset provided by Kaggle.</li>

<li>Deep learning models were quick to over fit and sensitive to hyper parameter tuning.
</ol>

## Data Exploration

Our major dataset were the files presented by the Kaggle challenge, but we added additional sources to help our feature engineering efforts. The data sources mentioned were used for both the unsupervised and supervised sections
of the project. Specifically, the files we used were:

**WikiLarge_Train.csv** [Kaggle]:<br>
Training dataset containing 416,768 sentences from Wikipedia that also contains labels. 0 refers to sentences that are
sufficiently simple, and1 refers to sentences that require review for simplification. Our exploration of the training
data revealed that the data is balanced in terms of labels. This makes classification accuracy more reasonable, but we decided to use both accuracy and F1 scores into consideration for extra information in model evaluation.<br>
**WikiLarge_Test.csv** [Kaggle]:<br>
Test dataset with 119,092 sentences without labels. Predicting the labels of this dataset is the main evaluation task based on the Kaggle challenge. The evaluation metric used is classification accuracy.<br>
Dale_chall.txt [Kaggle]:<br>
List of 3,000 words that are understood in reading by at least 80% of 5th graders. A sentence primarily composed of
these words is most likely to be a simple sentence.<br>
**SAT_words.csv** [External]:<br>
This file is an external CSV resource we collected to complement the existing files. This file contains 6,000 SAT
words which are considered more difficult. Our intuition was that words with more SAT words would be more likely
to be considered complex and thus require simplification.
**AoA_51715_words.csv** [Kaggle]:
This file contains Age of Acquisition (AoA) estimates for about 51k lemmatized English words with corresponding
metrics about the average age the lexical meaning is acquired. We selected the age of acquisition (AoA) of the
lemmatized word and the frequency of use in general English as features. The former represents the average
perceived difficulty of understanding the word and the latter represents the obscurity of the word through lack of
use. We were selective of which features to train the model on as we were attempting to avoid adding too much
noise.
**Concreteness_ratings_Brysbaert_et_al_BRM.txt** [Kaggle]:
This file contains "concreteness ratings" for 40 thousand English lemmatized words known by at least 85% of raters.
We selected the mean concreteness rating to demonstrate the ambiguity associated with these known words. The
concreteness word list was a narrower and better-known list of words compared to the Age of Acquisition list. The
concreteness score thus was focused on different types of ambiguity which added quality to our features.

## Feature Engineering

Our text data is not enough to generate highly robust classifiers. Furthermore, our supervised methods revealed heavy overlap in terms of clusters and topics which increased the need for more robust features to help separate the simple from the complex sentences. Thus, we engineered 30 features using a machine learning cycle framework. We would explore the data, develop the models, and then repeat the cycle exploring and adding new features.

We also utilized external NLP metrics from the following sources:<br>
- **Text_ stats from Textacy** - Automated Readability Index, Coleman Liau Index,
Flesch Reading Index, Gunning Fog Index, Perspicuity Index, and Smog Index
* **Word Embeddings from FastText** - Average Embed and Sentence Embed
* **NLTK** - pronoun count and percentages


