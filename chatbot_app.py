# import required packages
import json
import random
import tensorflow as tf
import numpy as np
from nltk_utils import tokenize, bag_of_words
from data_preprocess import vocabulary, tags

# open the intent file
with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

# load the saved chatbot brain
chatbot_brain = tf.keras.models.load_model('Barista.keras')

# Create the console chatbot app
bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    
    # Tokenize the user input
    sentence = tokenize(sentence)
    # Apply bag of words
    X = bag_of_words(sentence, vocabulary= vocabulary)
    X = X.reshape(1,-1)
    # Predict the user intent
    y_pred = chatbot_brain.predict(X,verbose=0)
    intent_class_id = np.argmax(y_pred)
    pred_proba = np.max(y_pred)
    intent_tag = tags[intent_class_id]
    # the chatbot will generate it's response
    if pred_proba > 0.75:
        for intent in intents['intents']:
            if intent_tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Sorry I do not understand.. Could you make it simpler please !")
