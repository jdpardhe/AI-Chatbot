# AI-Chatbot
A humorous chatbot that uses TensorFlow to intelligently respond to questions. Uses Conda and other imported libraries to function, currently trains with 99.99% accuracy. The bot will only respond if it has 85% accuracy; otherwise, it will ask you to enter something else.

It has six user intents that it can identify: greeting, goodbye, age, name, shop, and hours. The JSON file is capable of expansion to any size, but any modification needs a retraining before it can be accessed.

Currently, the chatbot uses two levels of neurons, with 8 neurons per level.

Tested on Python 3.6.

To run, make sure you have the following installed:

Python 3.6 (https://www.python.org/downloads/)
Anaconda for Python 3.7 (https://www.anaconda.com/products/individual)
ntlk        (pip install ntlk)
numpy       (pip install numpy)
tflearn     (pip install tflearn)
tensorflow  (pip install tensorflow)
