import os
import pickle
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DIRECTORY_NAME = os.getenv("DIRECTORY_NAME")

def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

# Load the models and tokenizers (prefer DIRECTORY_NAME if present)
lstm_model_path = _first_existing(os.path.join(DIRECTORY_NAME, "sentiment_model_RecNN.keras"),
                                  os.path.join(DIRECTORY_NAME, "sentiment_model_RecNN.h5"),
                                  "sentiment_model_RecNN.h5")
lstm_tokenizer_path = _first_existing(os.path.join(DIRECTORY_NAME, "sentiment_tokenizer_RecNN.pkl"),
                                      "sentiment_tokenizer_RecNN.pkl")
cnn_model_path = _first_existing(os.path.join(DIRECTORY_NAME, "sentiment_model_CNN.keras"),
                                 os.path.join(DIRECTORY_NAME, "sentiment_model_CNN_test.keras"),
                                 "sentiment_model_CNN.h5")
cnn_tokenizer_path = _first_existing(os.path.join(DIRECTORY_NAME, "sentiment_tokenizer_CNN.pkl"),
                                     os.path.join(DIRECTORY_NAME, "sentiment_tokenizer_CNN_test.pkl"),
                                     "sentiment_tokenizer_CNN.pkl")

lstm_model = load_model(lstm_model_path)
with open(lstm_tokenizer_path, "rb") as f:
    lstm_tokenizer = pickle.load(f)
cnn_model = load_model(cnn_model_path)
with open(cnn_tokenizer_path, "rb") as f:
    cnn_tokenizer = pickle.load(f)

while True:
    # Get user input
    text = input("Enter text for sentiment analysis (or 'quit' to exit): ")

    if text.lower() == "quit":
        break

    # Tokenize and pad the input text
    lstm_sequences = lstm_tokenizer.texts_to_sequences([text])
    lstm_padded_sequences = pad_sequences(lstm_sequences, padding='post', maxlen=50)
    cnn_sequences = cnn_tokenizer.texts_to_sequences([text])
    cnn_padded_sequences = pad_sequences(cnn_sequences, maxlen=359)

    # Perform sentiment analysis with LSTM model
    lstm_prediction = lstm_model.predict(lstm_padded_sequences)[0][0]
    lstm_sentiment = "Positive" if lstm_prediction >= 0.5 else "Negative"

    # Perform sentiment analysis with CNN model
    print(cnn_model.predict(cnn_padded_sequences))
    cnn_prediction = cnn_model.predict(cnn_padded_sequences)[0][0]
    cnn_sentiment = "Positive" if cnn_prediction >= 0.65 else "Negative"

    # Print the sentiment and confidence for each model
    print("RecNN Model:")
    print("Sentiment: ", lstm_sentiment)

    print("CNN Model:")
    print("Sentiment: ", cnn_sentiment)
