import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define some sample responses
responses = {
    'hi': 'Hello, how can I help you?',
    'bye': 'Goodbye, have a nice day!',
    'thanks': 'You\'re welcome!',
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
    print('Bot:', response)
