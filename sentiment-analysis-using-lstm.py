#import required libraries
%matplotlib inline

# numbers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GRU
from keras.layers.embeddings import Embedding

#load the dataset
df_train = pd.read_csv('archive/drugsComTrain_raw.csv')
df_test = pd.read_csv('archive/drugsComTest_raw.csv') 
print(f"The training dataset shape has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
print(f"The testing dataset shape has {df_test.shape[0]} rows and {df_test.shape[1]} columns")

#convert date attribute to datatime object
df_train['date']=pd.to_datetime(df_train['date'])
df_test['date']=pd.to_datetime(df_test['date'])

#data pre-processing
#drop null values from both datasets
df_train.dropna(inplace=True,axis=0)
df_test.dropna(inplace=True,axis=0)
print(f"The training dataset after removing null values has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
print(f"The testing dataset after removing null values has {df_test.shape[0]} rows and {df_test.shape[1]} columns")

#lets make new column sentiment in both datasets(If rating >= 5, then sentiment of review is positive, otherwise negative)
df_train.loc[(df_train['rating'] > 5 ),'sentiment'] = 1
df_train.loc[(df_train['rating'] <= 5),'sentiment'] = 0

df_test.loc[(df_test['rating'] > 5 ),'sentiment'] = 1
df_test.loc[(df_test['rating'] <= 5),'sentiment'] = 0

#define input and terget variables
x_train=df_train['review']
y_train=df_train['sentiment']

x_test=df_test['review']
y_test=df_test['sentiment']

#Text conversion into vectors-

#the maximum number of words to keep 
max_words = 15000

#create keras tokenizer object
tokenizer = Tokenizer(num_words=max_words)

#This method creates the vocabulary index based on word frequency, so lower integer means more frequent word
tokenizer.fit_on_texts(x_train)

#takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary
X_train=tokenizer.texts_to_sequences(x_train)
X_test=tokenizer.texts_to_sequences(x_test)

#maximum length of one review we will use for our training
review_length = 150

#adding padding to all reviews so that all reviews have equal length
X_train=pad_sequences(X_train,maxlen=review_length)
X_test=pad_sequences(X_test,maxlen=review_length)


#building lstm model
#build lSTM model
import time
current_time=time.time()

model=Sequential()

#add layers
model.add(layers.Embedding(input_dim=max_words,output_dim=100,input_length=review_length))
model.add(LSTM(30, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

#print summary of the model
print(model.summary())

#compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the model on traning set
hist=model.fit(X_train, y_train,batch_size=64,epochs=6,validation_data=(X_test,y_test))

print(f'Time to train using LSTM model is: {time.time()-current_time}')

#evaluate the model
results = model.evaluate(X_test,y_test)

#plotting accuracy vs loss for training and testing datasets
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#verifying result on first 10 samples
print("Prediction: \n",model.predict_classes(X_test[0:10]))

print("Actual: \n",y_test[0:10])

#predict result on x_test
y_pred = model.predict(X_test)

#concat results of prediction to input columns to test dataset
df = pd.DataFrame(y_pred,columns=['sentiment_score'])

df_test= pd.concat([df_test, df],axis=1)

#if predicted_sentiment value is >0.5, sentiment=positive,if predicted_sentiment value is < 0.5, sentiment=negative
df_test.loc[(df_test['sentiment_score'] > 0.5 ),'predicted_sentiment'] = 'positive'
df_test.loc[(df_test['sentiment_score'] < 0.5),'predicted_sentiment'] = 'negative'

df_test.head(10000)
