from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten, GRU, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import spacy
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import string
import pickle
import os

spacy_model = "en_core_web_sm"
dataset_path = 'data/intents2.json'

def load_spacy_model():
    if not spacy.util.is_package(spacy_model):
        spacy.cli.download(spacy_model)

    return spacy.load(spacy_model)

def get_lemmatized(model, sentence):
    doc = model(sentence)

    lemmatized = [token.lemma_ for token in doc]
    return ' '.join(lemmatized)

def best_match(question, answers, nlp_model):
    question_words = set(question.lower().split())
    lemma_answers = [get_lemmatized(nlp_model, answer) for answer in answers]
    best_answer = None
    max_matches = 0

    for answer in lemma_answers:
        answer_words = set(answer.lower().split())
        matches = len(question_words & answer_words)

        if matches > max_matches:
            max_matches = matches
            best_answer = answer

    if best_answer:            
        answer_index = lemma_answers.index(best_answer)
        return answers[answer_index]
    else:
        # respond with the generic answer
        return answers[0]

class ChatBot:

    def __init__(self, model = None, input_shape = None, tokenizer = None, responses = None) -> None:
        self.tokenizer = tokenizer or pickle.load(open('temp/tokenizer.pkl', 'rb'))
        self.input_shape = input_shape or pickle.load(open('temp/input_shape.pkl', 'rb'))
        self.intents = json.loads(open(dataset_path).read())
        self.tags_encoder = pickle.load(open("temp/tags_encoder.pkl", "rb"))
        self.model = model or load_model('temp/lstm_based')
        self.responses = responses or pickle.load(open('temp/responses.pkl', 'rb'))

    def predict_class(self, sentence):
        prediction_input = sentence

        nlp = load_spacy_model()

        prediction_input = get_lemmatized(nlp, prediction_input)

        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        raw_question = prediction_input

        # tokenizing and padding
        prediction_input = self.tokenizer.texts_to_sequences([prediction_input])
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], self.input_shape)

        # getting output from model
        output = self.model.predict(prediction_input)
        print(output)
        print(self.tags_encoder.classes_)
        output = output.argmax()

        # finding the right tag and predicting
        response_tag = self.tags_encoder.inverse_transform([output])[0]
        return best_match(question=raw_question, 
                          answers=self.responses[response_tag], 
                          nlp_model=nlp)

def train():

    if os.path.isdir('temp'):
        model = load_model('temp/model.keras')
        tokenizer = pickle.load(open('temp/tokenizer.pkl', 'rb'))
        input_shape = pickle.load(open('temp/input_shape.pkl', 'rb'))
        responses = pickle.load(open('temp/responses.pkl', 'rb'))

        return model, input_shape, tokenizer, responses

    intents = json.loads(open(dataset_path).read())
    tokenizer = Tokenizer(num_words=2000)
    tags_encoder = LabelEncoder()
    nlp = load_spacy_model()

    collection_list = []
    responses = {}

    print('Preprocessing the dataset ..')

    intent_list = intents.get('intents')
    for intent in intent_list:
        questions = intent.get('questions')
        tag = intent.get('tag')
        for question in questions:
            collection_list.append([get_lemmatized(nlp, question), tag]) 
        responses[tag] = intent.get('answers')

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

    print(input_shape)
    print(vocabulary)

    print('Dataset preprocessing done')

    # model = Sequential()
    # model.add(Embedding(vocabulary+1, 64, input_length=input_shape))
    # # model.add(Dropout(0.4))
    # model.add(LSTM(64, return_sequences=True))
    # # model.add(Dropout(0.6))
    # model.add(GRU(64, return_sequences=True))
    # model.add(Flatten())
    # model.add(Dense(output_length, activation='softmax'))

    model = Sequential()
    model.add(Embedding(vocabulary+1, 64, input_length=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Flatten())
    model.add(Dense(output_length, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train = model.fit(x_train, y_train, epochs=300)
    
    os.makedirs(os.path.dirname("temp/tags_encoder.pkl"), exist_ok=True)
    pickle.dump(tags_encoder, open("temp/tags_encoder.pkl", "wb"))
    pickle.dump(tokenizer, open('temp/tokenizer.pkl', 'wb'))
    pickle.dump(input_shape, open('temp/input_shape.pkl', 'wb'))
    pickle.dump(responses, open('temp/responses.pkl', 'wb'))
    model.save('temp/model.keras')

    return model, input_shape, tokenizer, responses