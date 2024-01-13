from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, BatchNormalization

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import string
import pickle
import os
import shutil
from wordcloud import WordCloud
from PIL import Image

spacy_model = "en_core_web_sm"
dataset_path = 'data/intents.json'

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

    # if temp directory exists means the model has already trained and stored. 
    # So skip re-training and load the existing one
    if os.path.isdir('temp'):
        print('Loading the model from cache')
        return load_model_data()

    tokenizer = Tokenizer(num_words=2000)
    tags_encoder = LabelEncoder()
    nlp = load_spacy_model()
    collection_list = []
    responses = {}
    question_list = []

    print('Processing the dataset ..')

    main_dataframe = load_and_preprocess(dataset_path, nlp, collection_list, responses, question_list)

    X = get_input_feature_vectors(tokenizer, main_dataframe)
    y = get_encoded_labels(tags_encoder, main_dataframe)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = x_train.shape[1]
    vocabulary = len(tokenizer.word_index)
    output_length = tags_encoder.classes_.shape[0]
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_dim = 300

    weight_matrix = get_weight_matrix_from_fasttext(vector_model, word_index, vocab_size, embedding_dim)

    print('Done processing the dataset')

    model = prepare_ann_model(input_shape, vocabulary, output_length, embedding_dim, weight_matrix)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    train = model.fit(x_train, y_train, epochs=2,  
                      validation_split=0.2,
                      callbacks=[early_stopping]
                      )

    print('Evaluating with test dataset..')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%, Test loss: {test_loss * 100:.2f}')

    # Plotting various figures
    clean_old_figures()
    plot_loss_variation_graph(train)
    plot_accuracy_variation_graph(train)
    plot_confusion_matrix(model, tags_encoder, x_test, y_test)
    plot_train_test_loss(train, test_loss)
    plot_train_test_accuracy(train, test_accuracy)
    plot_wordcloud(question_list)

    dump_model_data(model, tokenizer, input_shape, responses, tags_encoder)

    return model, input_shape, tokenizer, responses

def prepare_ann_model(input_shape, vocabulary, output_length, embedding_dim, weight_matrix):
    model = Sequential()
    model.add(Embedding(vocabulary+1, embedding_dim, weights=[weight_matrix], 
                        input_length=input_shape, trainable=False))
    
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
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(50, activation="leaky_relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(output_length, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])
                  
    return model

def clean_old_figures():
    directory = "figure"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def get_weight_matrix_from_fasttext(vector_model, word_index, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in vector_model:
            embedding_matrix[i] = vector_model[word]
    return embedding_matrix

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

def get_encoded_labels(tags_encoder, main_dataframe):
    tag_list = main_dataframe['tag'].unique().tolist()
    tags_encoder.fit(tag_list)

    return tags_encoder.transform(main_dataframe['tag'])

def get_input_feature_vectors(tokenizer, main_dataframe):
    tokenizer.fit_on_texts(main_dataframe['sentence'])
    X = tokenizer.texts_to_sequences(main_dataframe['sentence'])

    return pad_sequences(X)

def load_and_preprocess(dataset_path, nlp, collection_list, responses, question_list):
    intents = json.loads(open(dataset_path).read())
    intent_list = intents.get('intents')
    for intent in intent_list:
        questions = intent.get('questions')
        tag = intent.get('tag')
        for question in questions:
            qs = get_lemmatized(nlp, question)
            collection_list.append([qs, tag])
            question_list.append(question) 
        responses[tag] = intent.get('answers')

    main_dataframe = pd.DataFrame(collection_list, columns=['sentence', 'tag'])

    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    main_dataframe['sentence'] = main_dataframe['sentence'].apply(lambda wrd: ''.join(wrd))

    return main_dataframe

def plot_wordcloud(question_list):

    # wine_mask = np.array(Image.open("data/restaurent_icon-whitebg.png"))
    image_mask = np.array(Image.open("data/restaurent_icon.png"))

    text = ' '.join(question_list)
    wordcloud = WordCloud(background_color="white", max_words=2000, mask=image_mask, 
                          contour_width=1, contour_color='green')
    wordcloud.generate(text)

    plt.figure(figsize=(40, 36))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('figure/wordcloud.png')
    plt.close()

def plot_train_test_accuracy(train, test_accuracy):
    # Get the average training accuracy
    avg_train_accuracy = sum(train.history['accuracy']) / len(train.history['accuracy'])
    bar_index = np.arange(2)
    plt.bar(bar_index, [avg_train_accuracy, test_accuracy])
    plt.xticks(bar_index, ['Train', 'Test'])
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.savefig('figure/train_test_accuracy_diff.png')
    plt.close()

def plot_train_test_loss(train, test_loss):
    # Get the average training loss
    avg_train_loss = sum(train.history['loss']) / len(train.history['loss'])
    bar_index = np.arange(2)
    plt.bar(bar_index, [avg_train_loss, test_loss])
    plt.xticks(bar_index, ['Train', 'Test'])
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.savefig('figure/train_test_loss_diff.png')
    plt.close()

def load_model_data():
    model = load_model('temp/model.keras')
    tokenizer = pickle.load(open('temp/tokenizer.pkl', 'rb'))
    input_shape = pickle.load(open('temp/input_shape.pkl', 'rb'))
    responses = pickle.load(open('temp/responses.pkl', 'rb'))

    return model,tokenizer,input_shape,responses

def dump_model_data(model, tokenizer, input_shape, responses, tags_encoder):
    os.makedirs(os.path.dirname("temp/tags_encoder.pkl"), exist_ok=True)
    pickle.dump(tags_encoder, open("temp/tags_encoder.pkl", "wb"))
    pickle.dump(tokenizer, open('temp/tokenizer.pkl', 'wb'))
    pickle.dump(input_shape, open('temp/input_shape.pkl', 'wb'))
    pickle.dump(responses, open('temp/responses.pkl', 'wb'))
    model.save('temp/model.keras')

def plot_loss_variation_graph(train):
    plt.plot(train.history['loss'])
    plt.plot(train.history['val_loss'])
    plt.title('Loss difference')
    plt.savefig('figure/loss_plot.png')
    plt.close()

def plot_accuracy_variation_graph(train):
    plt.plot(train.history['accuracy'])
    plt.plot(train.history['val_accuracy'])
    plt.title('Accuracy difference')
    plt.savefig('figure/acc_plot.png')
    plt.close()

def plot_confusion_matrix(model, tags_encoder, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate the confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_classes)

    plt.figure(figsize=(12, 14))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=tags_encoder.classes_, 
                yticklabels=tags_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('figure/confusion_matrix.png')
    plt.close()
