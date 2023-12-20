from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import string
import pickle
import os

class ChatBot:

    def __init__(self, model = None, input_shape = None, tokenizer = None, responses = None) -> None:
        self.tokenizer = tokenizer or pickle.load(open('temp/tokenizer.pkl', 'rb'))
        self.input_shape = input_shape or pickle.load(open('temp/input_shape.pkl', 'rb'))
        self.intents = json.loads(open("data/intents.json").read())
        self.tags_encoder = pickle.load(open("temp/tags_encoder.pkl", "rb"))
        self.model = model or load_model('temp/lstm_based')
        self.responses = responses or pickle.load(open('temp/responses.pkl', 'rb'))

    def predict_class(self, sentence):
        texts_p = []
        prediction_input = sentence

        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)

        # tokenizing and padding
        prediction_input = self.tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], self.input_shape)

        # getting output from model
        output = self.model.predict(prediction_input)
        output = output.argmax()

        # finding the right tag and predicting
        response_tag = self.tags_encoder.inverse_transform([output])[0]
        return self.responses[response_tag]

def train():

    if os.path.isdir('temp'):
        model = load_model('temp/model.keras')
        tokenizer = pickle.load(open('temp/tokenizer.pkl', 'rb'))
        input_shape = pickle.load(open('temp/input_shape.pkl', 'rb'))
        responses = pickle.load(open('temp/responses.pkl', 'rb'))

        return model, input_shape, tokenizer, responses

    intents = json.loads(open("data/intents.json").read())
    tokenizer = Tokenizer(num_words=2000)
    tags_encoder = LabelEncoder()

    collection_list = []
    responses = {}

    intent_list = intents.get('intents')
    for intent in intent_list:
        questions = intent.get('questions')
        tag = intent.get('tag')
        for question in questions:
            collection_list.append([question, tag]) 
        responses[tag] = intent.get('answers')[0]

    main_dataframe = pd.DataFrame(collection_list, columns=['sentence', 'tag'])

    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: ''.join(wrd))

    tokenizer.fit_on_texts(main_dataframe['sentence'])
    train = tokenizer.texts_to_sequences(main_dataframe['sentence'])

    # apply padding
    x_train = pad_sequences(train)

    tag_list = main_dataframe['tag'].unique().tolist()

    tags_encoder.fit(tag_list)

    y_train = tags_encoder.transform(main_dataframe['tag'])

    input_shape = x_train.shape[1]
    vocabulary = len(tokenizer.word_index)
    output_length = tags_encoder.classes_.shape[0]

    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary+1, 10)(i)
    x = LSTM(10, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length, activation='softmax')(x)
    model = Model(i, x)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train = model.fit(x_train, y_train, epochs=200)
    
    os.makedirs(os.path.dirname("temp/tags_encoder.pkl"), exist_ok=True)
    pickle.dump(tags_encoder, open("temp/tags_encoder.pkl", "wb"))
    pickle.dump(tokenizer, open('temp/tokenizer.pkl', 'wb'))
    pickle.dump(input_shape, open('temp/input_shape.pkl', 'wb'))
    pickle.dump(responses, open('temp/responses.pkl', 'wb'))
    model.save('temp/model.keras')

    return model, input_shape, tokenizer, responses