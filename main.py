import requests 
import pandas as pd
import numpy as np

df = pd.read_csv('IMDB Dataset.csv')

df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': -1})

apostrophe = {"isn't": 'is not', "don't": 'do not', "can't": 'can not', "cannot": 'can not'}
negation = {'not good': 'notgood', 'not bad': 'notbad', 'not great': 'notgreat'}

df['review'] = df['review'].replace(apostrophe, regex=True)
df['review'] = df['review'].replace(negation, regex=True)

target = df['sentiment']

features = ['good', 'bad','amazing', 'okay', 'terrible', 'solid', 'poor', 'decent', 
            'great', 'notgood', 'notbad', 'notgreat', 'not', 'awful', 'amazing', 'new', 'favorite', 
            'worst', 'plot', 'masterpiece', 'after', 'like', 'average', 'could', 'gripping', 'too', 'but'
            'wonderful', 'how', 'most', 'one', 'garbage', 'fan', 'big', 'just', 'enjoyed', 'liked', 'laughed',
            'boring', 'tried', 'tries', 'joke', 'maybe', 'horrible', 'best', 'well', 'outstanding', 'bored']

feature_vector = pd.DataFrame()

for feature in features:
    feature_vector[feature] = df['review'].str.count(feature)

np.random.seed(42) 
mask = np.random.rand(len(df)) < 0.7
train_data = feature_vector[mask]
test_data = feature_vector[~mask]
train_labels = target[mask]
test_labels = target[~mask]


X_train = train_data.values
y_train = train_labels.values
X_test = test_data.values
y_test = test_labels.values

num_features = X_train.shape[1]
weights = np.zeros(num_features)
bias = 0.0
learning_rate = 0.002
epochs = 200


#sgd
for epoch in range(epochs):
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = y_train[i]
        prediction = np.dot(weights, xi) + bias
        if yi * prediction < 1:
            gradient_w = -yi * xi
            gradient_b = -yi
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b
        else:
            gradient_w = 0
            gradient_b = 0

train_predictions = np.dot(X_train, weights) + bias
test_predictions = np.dot(X_test, weights) + bias

train_accuracy = np.mean(np.sign(train_predictions) == y_train)
test_accuracy = np.mean(np.sign(test_predictions) == y_test)

print("Final Train Accuracy:", train_accuracy)
print("Final Test Accuracy:", test_accuracy)
