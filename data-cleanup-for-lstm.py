import regex as re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from textblob import TextBlob

#clean the review comments
def clean_review_comments(uncleanText):
            
    #Beautiful Soup is a Python library for pulling data out of HTML and XML files.
    #To get human-readable text inside a document or tag, you can use the get_text() method
    cleanText = BeautifulSoup(uncleanText, 'html.parser').get_text()
    
    #remove special characters
    cleanText = re.sub('[^a-zA-Z]', ' ', cleanText)
    
    #convert to lower case
    cleanText = cleanText.lower()
    
    #remove stopwords
    sw = stopwords.words("english")
    nsw = " ".join([word for word in cleanText.split() if word not in sw])
   
    #apply lemmatization
    blob=TextBlob(nsw)
    meaningful_words=" ".join([word.lemmatize() for word in blob.words])
    return meaningful_words
