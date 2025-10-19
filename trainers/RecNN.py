import os
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from dask import dataframe as ddf
import pandas as pd
from pymongo import MongoClient
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import pickle
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LIMIT = int(os.getenv("LIMIT"))
DIRECTORY_NAME = os.getenv("DIRECTORY_NAME")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# Fetch reviews from the database and store them in a DataFrame
reviews = collection.find(limit=LIMIT)
start_time = time()
data = ddf.from_pandas(pd.DataFrame(list(reviews)), npartitions=10).compute()
data['positive'] = data['positive']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['positive'], train_size=0.3, random_state=1)

# Tokenize the text data and create a vocabulary
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
max_len = 50

# Model creation
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Print model summary
print(model.summary())

# Convert text data to sequences and pad them to a fixed length
cv = CountVectorizer(max_features=1000, binary=False)
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# Fit the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=1)

end_time = time()

# Print test loss and accuracy
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Plot the model's accuracy over epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot the model's loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

seconds_elapsed = end_time - start_time
print('Amount of time needed to complete: ' + str(seconds_elapsed) )

# Save the model for later use
os.makedirs(DIRECTORY_NAME, exist_ok=True)
model.save(os.path.join(DIRECTORY_NAME, "sentiment_model_RecNN.keras"))
# Save the tokenizer for later use
with open(os.path.join(DIRECTORY_NAME, "sentiment_tokenizer_RecNN.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
