Drugs.com reviews dataset sentiment analysis using Recurrent Neural Networks(RNNs)
Problem Definition- Classification of drug reviews(whether it is a positive/negative review) to determine their sentiments using Python libraries- word embeddings and Natural Language Toolkit,Textblob, Long-Short-Term-Memory(LSTM) RNNs. Achieve the binary classification of opinion/feedback shared by users to determine the usefulness of drugs.


Applications- 
Machine learning has permeated nearly all fields and disciplines of study. One hot topic is using natural language processing and sentiment analysis to identify, extract, and make use of subjective information. The use case of this could be for industries to find out what really people think about their products and can take data driven decisions to improve their businesses.

Following are the important steps of this sentiment classification study-

Data Acquisition- 
I used this dataset which is published in UCI ML repository which was obtained by crawling drugs.com website(also available on Kaggle). The dataset contains 2 csv files- test data and train data files.This dataset provides patient reviews on specific drugs along with related conditions and a 1-10 patient rating system reflecting overall patient satisfaction. This data was published in a study on sentiment analysis of drug experience over multiple facets, ex. sentiments learned on specific aspects such as effectiveness and side effects.


Data Pre-processing- Next important step after acquiring the dataset was to clean the data and make it suitable for classification purpose. In the first step, i did convert all categorical columns to numerical columns and removed null values present in all columns. After that, i removed special characters, numbers, stopwords, hyperlinks and normalized it to lower or upper case case to maintain the uniformity. Then, the review sentences were formatted uniformly to make their length <= 200 characters which we will use to train our model.

Data split- After preprocessing, the entire dataset was splitted into training, test and validation which is used to optimize the hyperparameters of our models

Vectorization- Since the raw text can't be fed as it is to any ML model,  the Tokenizer class of Keras is used for vectorizing a text corpus. This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector.embedding was used along with Tokenization and lemmatization to create a meaningful mathematical representation of words in a text corpus

Classification- At the end, the vectorized reviews were given as an input to ML model such as LSTM RNN for classification, where RNN outperformed other ML algorithms and gave up to 88% test accuracy.

Tools used: Keras, Tensorflow, Jupyter Notebook
