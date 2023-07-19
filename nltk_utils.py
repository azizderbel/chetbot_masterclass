import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

#nltk.download('punkt')

# Import the NLTK stmmer
stemmer = PorterStemmer()


# Define the tokenization funtion
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Define the stemming funtion
def stem(word):
    return stemmer.stem(word=word.lower())

# define the bag of words function
def bag_of_words(tokenized_pattern,vocabulary):
    tokenized_pattern = [stem(word) for word in tokenized_pattern]
    bag= np.zeros(len(vocabulary),dtype=np.float32)

    # for each word in the pattern wi create a binary vetor
    for index,word in enumerate(vocabulary):
        if word in tokenized_pattern:
            bag[index] = 1.0

    return bag