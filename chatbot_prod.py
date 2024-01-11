from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten, GRU, Dropout, BatchNormalization

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import matplotlib.pyplot as plt

import spacy
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import string
import pickle
import os
import shutil

spacy_model = "en_core_web_sm"
dataset_path = 'data/intents2.backup-3.json'


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

    def __init__(self, vector_model, model = None, input_shape = None, tokenizer = None, responses = None) -> None:
        self.tokenizer = tokenizer or pickle.load(open('temp/tokenizer.pkl', 'rb'))
        self.input_shape = input_shape or pickle.load(open('temp/input_shape.pkl', 'rb'))
        self.intents = json.loads(open(dataset_path).read())
        self.tags_encoder = pickle.load(open("temp/tags_encoder.pkl", "rb"))
        self.model = model or load_model('temp/lstm_based')
        self.responses = responses or pickle.load(open('temp/responses.pkl', 'rb'))
        self.fasttext = vector_model

    def chat_query(self, sentence):
        nlp = load_spacy_model()

        prediction_input = sentence
        prediction_input = ''.join([letters.lower() for letters in prediction_input if letters not in string.punctuation])
        prediction_input = get_lemmatized(nlp, prediction_input)
        
        print(prediction_input)
        raw_question = prediction_input

        # tokenizing and padding
        prediction_input = self.tokenizer.texts_to_sequences([prediction_input])
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], self.input_shape)

        # getting output from model
        output = self.model.predict(prediction_input)
        print(output)
        print(self.tags_encoder.classes_)
        output_max_value = output.max()
        output = output.argmax()

        if output_max_value < .1:
            return "Sorry I didn't get that"

        # finding the right tag and predicting
        response_tag = self.tags_encoder.inverse_transform([output])[0]
        return best_match(question=raw_question, 
                          answers=self.responses[response_tag], 
                          nlp_model=nlp)

def train(vector_model):

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
            qs = get_lemmatized(nlp, question)
            collection_list.append([qs, tag]) 
        responses[tag] = intent.get('answers')

    main_dataframe = pd.DataFrame(collection_list, columns=['sentence', 'tag'])

    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: ''.join(wrd))

    tokenizer.fit_on_texts(main_dataframe['sentence'])
    X = tokenizer.texts_to_sequences(main_dataframe['sentence'])
    X = pad_sequences(X)

    tag_list = main_dataframe['tag'].unique().tolist()
    tags_encoder.fit(tag_list)
    y = tags_encoder.transform(main_dataframe['tag'])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    input_shape = x_train.shape[1]
    vocabulary = len(tokenizer.word_index)
    output_length = tags_encoder.classes_.shape[0]

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_dim = 300

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in vector_model:
            embedding_matrix[i] = vector_model[word]

    print('Dataset preprocessing done')

    model = Sequential()
    model.add(Embedding(vocabulary+1, embedding_dim, weights=[embedding_matrix], input_length=input_shape, trainable=False))
    
    model.add(LSTM(170, activation="leaky_relu", 
                   recurrent_activation="tanh", 
                   return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(85, activation="relu", 
                   recurrent_activation="sigmoid", 
                   dropout=0.3,
                   recurrent_dropout=0.2,
                   ))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(50, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Dense(output_length, activation='softmax'))
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])

    train = model.fit(x_train, y_train, epochs=550, 
                      validation_split=0.2,
                      callbacks=[early_stopping]
                      )

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%, Test loss: {loss * 100:.2f}')

    directory = "figure"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    plt.plot(train.history['loss'])
    plt.plot(train.history['val_loss'])
    plt.title('Loss difference')
    plt.savefig('figure/loss_plot.png')
    plt.close()

    plt.plot(train.history['accuracy'])
    plt.plot(train.history['val_accuracy'])
    plt.title('Accuracy difference')
    plt.savefig('figure/acc_plot.png')
    
    os.makedirs(os.path.dirname("temp/tags_encoder.pkl"), exist_ok=True)
    pickle.dump(tags_encoder, open("temp/tags_encoder.pkl", "wb"))
    pickle.dump(tokenizer, open('temp/tokenizer.pkl', 'wb'))
    pickle.dump(input_shape, open('temp/input_shape.pkl', 'wb'))
    pickle.dump(responses, open('temp/responses.pkl', 'wb'))
    model.save('temp/model.keras')

    return model, input_shape, tokenizer, responses