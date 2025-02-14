#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# (c) Aryan Sinha, Ashutosh Mishra, Aviral Jain
import nltk
import random
import requests
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from datetime import datetime,date

def get_current_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"The current time is {current_time}."

# Function to get the current date
def get_current_date():
    current_date = date.today().strftime("%B %d, %Y")
    return f"The current date is {current_date}."

def get_random_joke():
    if not jokes:
        return f"Sorry, I'm out of jokes for now. Check back later!"
    random_joke = random.choice(jokes)
    jokes.remove(random_joke)
    return f"Sure!,{random_joke}"
    
    
jokes=[
    "Police arrested two kids yesterday, one was drinking battery acid, the other was eating fireworks. They charged one – and let the other one off.",
    "I was having dinner with Garry Kasporov (world chess champion) and there was a check tablecloth. It took him two hours to pass me the salt.",
    "My therapist says I have a preoccupation with vengeance. We’ll see about that.",
    "What falls, but never needs a bandage? The rain.",
    "I was going to tell you a joke about boxing but I forgot the punch line.",
    "Why did the egg hide? It was a little chicken.",
    "Why couldn't the sunflower ride its bike? It lost its petals.",
    "What's an egg's favorite vacation spot? New Yolk City.",
    "I wanted to buy some camo pants but couldn't find any.",
    "What kind of candy do astronauts like? Mars bars."
]

url = "http://api.openweathermap.org/data/2.5/weather"
api_key = "e42e57fc7d102ae8eba1438cca1014fc"


def get_current_weather(location):
    # Parameters for the API request
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"  # Specify the unit of measurement (metric for Celsius)
    }
    response = requests.get(url, params=params)
    weather_data = response.json()
    if response.status_code == 200:
        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        weather_message = f"The current weather in {location} is {description}. " \
                          f"The temperature is {temperature}°C and the humidity is {humidity}%."
    else:
        weather_message = "Sorry, I couldn't retrieve the weather information at the moment. " \
                          "Please try again later."

    return f"{weather_message}"

location = "London"
# Define similarity threshold
SIMILARITY_THRESHOLD = 0.6

# Define some sample responses
responses = {
    'hi': 'Hello, how can I help you?',
    'hello': 'Hi there, how can I assist you?',
    'hey': 'Hi, what can I help you with?',
    'how are you': 'I am just a computer program, so I don\'t have feelings, but thank you for asking. How can I assist you?',
    'what can you do': 'I can assist you with answering questions or providing information. Just let me know what you need help with!',
    'who are you': 'I am Denniz, a chatbot designed to help answer your questions and provide information.',
    'what is your purpose': 'My purpose is to assist you with answering questions and providing information to the best of my abilities.',
    'thank you': 'You\'re welcome!',
    'thanks': 'No problem, happy to help!',
    'bye': 'Goodbye, have a great day!',
    'see you later': 'Until next time, take care!',
    'talk to you soon': 'Looking forward to our next conversation!',
    'what is weather like': 'Go out and look out yourself lol',
    'what is your name': 'My name is denniz, how can I assist you?',
    'what can you help me with': 'I can help you with a variety of topics including [insert topics here]. What do you need help with?',
    'can you provide more information': 'Sure, what would you like to know?',
    'can you explain that in simpler terms': 'Of course, let me break it down for you. [insert explanation here]. Does that make sense?',
    'where can I find more information': 'You can find more information about that topic by [insert where to find information here].',
    'how do I ': 'Help yourself',
    'what is the meaning of [insert word or phrase here]': 'The meaning of [insert word or phrase here] is [insert meaning here].',
    'what is the difference between sun and moon': 'As an AI Language model, I cant fetch these details at this point of time',
    'can you recommend a product': 'Sure, I would recommend [insert recommendation here].',
    'how do I contact customer support': 'To contact customer support, you can [insert contact information here].',
    'what is the weather like today': 'Im sorry, but I dont have access to real-time weather information. However, you can check the weather by [insert how to check weather here].',
    'what is your favorite color': 'As a chatbot, I dont have the ability to have a favorite color. How can I assist you?',
    'what is the meaning of life': 'The meaning of life is a complex philosophical question that has been debated for centuries. What specific question or topic can I assist you with?',
    'what is your favorite food': 'As a chatbot, I dont eat food, so I dont have a favorite. What can I help you with?',
    'Can you suggest a good restaurant nearby':'Certainly! Can you please provide me with your location or a specific cuisine preference?',
    'Thank you for your help!':'Youre welcome Im here to assist you anytime.',
    'tell me a fact': 'Did you know that the honeybee is the only insect that produces food eaten by humans?',
    'what is your favorite color': 'As a chatbot, I don\'t have preferences, including favorite colors. Is there something else I can assist you with?',
    'current time': get_current_time(),
    'current date': get_current_date(),
    'Tell me a joke': get_random_joke(),
    'Current weather': get_current_weather(location),
    'default': 'Sorry, I didn\'t understand what you said. Can you please rephrase your query?'
}
  

# Define a function to preprocess text data
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Load the data
with open('data.txt') as f:
    data = f.read().splitlines()

# Preprocess the data
data = [preprocess(text) for text in data]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(data)

def get_wordnet_pos(treebank_tag):
    """Map Treebank part of speech tags to WordNet part of speech tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None
# Define a function to get the most similar query
def get_most_similar_query(query):
    query_vec = vectorizer.transform([preprocess(query)])
    similarity_scores = cosine_similarity(query_vec, vectorized_data)
    max_score = max(similarity_scores[0])
    if max_score > 0:
        index = list(similarity_scores[0]).index(max_score)
        return data[index]
    else:
        return None

def get_similarity(input_text, response):
    # Tokenize and tag part of speech of input text and response
    input_tokens = pos_tag(word_tokenize(input_text.lower()))
    response_tokens = pos_tag(word_tokenize(response.lower()))

    # Get the synset (set of synonyms) for each word in the input text and response
    input_synsets = [wn.synsets(token, get_wordnet_pos(tag))[0] for token, tag in input_tokens if len(wn.synsets(token, get_wordnet_pos(tag))) > 0]
    response_synsets = [wn.synsets(token, get_wordnet_pos(tag))[0] for token, tag in response_tokens if len(wn.synsets(token, get_wordnet_pos(tag))) > 0]

    # Get the maximum similarity score between each pair of synsets
    max_similarities = [max([synset1.path_similarity(synset2) for synset2 in response_synsets if synset1.path_similarity(synset2) is not None] or [0]) for synset1 in input_synsets]

    # Return the average of the maximum similarity scores
    return sum(max_similarities) / len(max_similarities)

# Define a function to generate a response
def generate_response(user_input):
    # Preprocess the user input
    processed_input = preprocess(user_input)

    # Calculate the similarity between the user input and the available responses
    similarities = [get_similarity(processed_input, preprocess(response)) for response in responses.keys()]

    # Find the index of the most similar query
    most_similar_index = np.argmax(similarities)

    # Check if the most similar query has a similarity score greater than the threshold
    if similarities[most_similar_index] > SIMILARITY_THRESHOLD:
        # Retrieve the response with the highest similarity score
        most_similar_query = list(responses.keys())[most_similar_index]

        # Check if the response includes a placeholder for the bot's name
        if '{name}' in most_similar_query:
            # Replace the placeholder with the bot's name
            return responses[most_similar_query].replace('{name}', 'Denniz')
        else:
            return responses[most_similar_query]
    else:
        # Return a default response if the input query does not match any of the available responses
        return "I'm sorry, Denniz doesn't understand. Can you please rephrase your query?"

# Test the chatbot
while True:
    user_input = input('You: ')
    response = generate_response(user_input)
    print('Denniz:', response)
