import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from dask.distributed import Client
import joblib
from dask_ml.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LIMIT = int(os.getenv("LIMIT"))
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
data = pd.DataFrame(list(reviews))

# Extract the text reviews and corresponding labels from the DataFrame
texts = data['review']
labels = data['positive']

# Initialize Dask client
client = Client(processes=False)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, train_size=0.5, random_state=42)

# Vectorize text data using CountVectorizer
cv = CountVectorizer(max_features=1000, binary=False)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Train Logistic Regression model using Dask
with joblib.parallel_backend('dask'):
    LR = LogisticRegression(solver='lbfgs')
    LR.fit(X_train_cv.toarray(), y_train)

# Predict labels for the test set
y_test_pred = LR.predict(X_test_cv.toarray())
y_pred_proba = LR.predict_proba(X_test_cv.toarray())[::, 1]

end_time = time()

# Compute confusion matrix
confused_matrix = pd.DataFrame(confusion_matrix(y_test_pred, y_test), index=['Prediction: Negative', 'Prediction: Positive'], columns=['Answered: Negative', 'Answered: Positive'])

# Compute learning curve on vectorized features using scikit-learn estimator
train_sizes_arr, train_scores, test_scores = learning_curve(
    estimator=SkLogisticRegression(solver='lbfgs', max_iter=1000),
    X=X_train_cv,
    y=y_train,
    cv=10,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=1,
)


# Print confusion matrix and final accuracy
print(confused_matrix)
print("Final Accuracy: %s" % accuracy_score(y_test, y_test_pred))

# Plot ROC curve
plt.figure(figsize=(10, 5))
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Linear Regression AUC=" + str(auc))
plt.legend(loc=4)
plt.show()

# Calculate elapsed time
seconds_elapsed = end_time - start_time
print('Amount of time needed to complete: ' + str(seconds_elapsed))

# Perform live tests
liveTest = ['It was a really bad experience, never coming back', 'We had a nice stay at this Hotel, definitely coming back']
liveTest_cv = cv.transform(liveTest)
print(LR.predict(liveTest_cv.toarray()))
