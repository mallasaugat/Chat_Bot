
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import unicodedata
import pickle


stopwords = stopwords.words()

stemmer = PorterStemmer()
encoder = LabelEncoder()
vect = CountVectorizer(binary=True)

def stem(word):
    return stemmer.stem(word)

def remove_accents(words):
    return [unicodedata.normalize('NFKD', w).encode('ASCII', 'ignore').decode('utf-8') for w in words]

def preprocess_sentence(sentence):
    words = sentence.split(' ')
    words = [w.lower() for w in words if w.lower() not in stopwords]
    words = [stem(w) for w in words]
    words = remove_accents(words)
    sentence = ' '.join(words)
    return sentence

def vectorize(df):
    """
    Parameters
    ----------
    df : pandas DataFrame
        This dataframe contains the training data, 2 columns, the first one being the patterns, the second one being the tags.

    Returns
    -------
    X_train : numpy array 
        This array consists of vectorized patterns with numerical values to be able to train our neural network.
    y_train : numpy array
        Array containing tags or categories of patterns, encoded with a numerical value.
    """
    X_train = vect.fit_transform(df['Pattern']).toarray()
    y_train = encoder.fit_transform(df['Tag'])
    with open('vect.pkl','wb') as f_out: # Save the vectorizer object in a pickle file (serialization)
        pickle.dump(vect, f_out)
    return X_train, y_train

def vectorize_new(vect, sentence):
    """
    Parameters
    ----------
    vect : CountVectorizer object
        Object retrieved in the pickle file allowing us to convert a sentence into a numerical vector and follow the bag of words approach
    sentence : string
        Sentence written by a user in the chat (new input).

    Returns
    -------
    X : numpy array
        Vectorized input using the trained CountVectorizer object.
    """
    X = vect.transform([sentence]).toarray()
    return X
    