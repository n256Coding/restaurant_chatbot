import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

class ChatBot:

    def __init__(self, model = None, words = None, classes = None) -> None:
        self.model = model or load_model("chatbot_model.keras", compile=False)
        self.words = words or pickle.load(open("words.pkl", "rb"))
        self.classes = classes or pickle.load(open("classes.pkl", "rb"))

        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open("data/intents.json").read())


    # Clean up the sentences
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words


    # Converts the sentences into a bag of words
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)


    def predict_class(self, sentence):
        bow = self.bag_of_words(
            sentence
        )  # bow: Bag Of Words, feed the data into the neural network
        res = self.model.predict(np.array([bow]))[0]  # res: result. [0] as index 0
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list


    def get_response(intents_list, intents_json):
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]

        result = "Not found" 
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["answers"])
                break

        return result

# while True:
#     message = input("You: ")
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)
