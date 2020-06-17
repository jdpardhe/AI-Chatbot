import nltk
#nltk.download()
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:] #duplicates the output
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

#building model with tflearn
tensorflow.reset_default_graph() #resets the previous settings

#defines the shape of the tf graph
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#IF THE JSON FILE IS MODIFIED, DELETE THE .PICKLE FILE TO RETRAIN OR ADD x before model.load
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("\n\n*yawn (this again?? >_<)*\n\nYo wassup? Type \"quit\" to stop anytime...")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        #predicts the most probable response
        results = model.predict([bag_of_words(inp, words)])[0]
        #gives the index of the greatest value of the list of probabilities
        results_index = numpy.argmax(results)
        #gives the label of the highest probabilities as variable tag
        tag = labels[results_index]

        if results[results_index] > 0.85:
            #opens the intents JSON file to grab a response to return
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            #prints out a random response
            print("Computer boi: " + random.choice(responses))
            print()

        else:
            idk = ["not really sure what you said :/ could you ask something different?\n", "yeah...I didn't quite get that, try again plz\n"]
            print("Computer boi: " + random.choice(idk))

chat()
