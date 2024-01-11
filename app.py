# Setup dependencies
import os
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import streamlit
import re

# setup OS environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the tokenizer and ML models
with open('tokenizer.pickle', 'rb') as tk:
    tokenizer = pickle.load(tk)

json_file = open('model.json', 'r')
loaded_model_json =json_file.read()
json_file.close()

lstm_model = model_from_json(loaded_model_json)
lstm_model.load_weights('model.h5')

# Helper function to cleanup and tokenize input data for sentiment analysis prediction
def sentiment_prediction(review):
    sentiment=[]
    input_review = [review]
    input_review = [x.lower() for x in input_review]
    input_review = [re.sub('[^a-zA-Z0-9\s]','',x) for x in input_review]

    input_feature = tokenizer.texts_to_sequences(input_review)
    input_feature = pad_sequences(input_feature, 1473, padding='pre')

    sentiment = lstm_model.predict(input_feature)[0]
    if(np.argmax(sentiment)==0):
        pred='Negative'
    else:
        pred='Positive'

    return pred

# Function to run when loading the html page
def run():
    streamlit.title('Sentiment analysis with LSTM Model')
    html_temp = """
    """

    streamlit.markdown(html_temp)
    review = streamlit.text_input('Enter the review:')
    prediction=""
    if streamlit.button('Predict Sentiment'):
        prediction=sentiment_prediction(review)
    streamlit.success('The sentiment predicted by the model: {}'.format(prediction))

if __name__ == '__main__':
    run()

