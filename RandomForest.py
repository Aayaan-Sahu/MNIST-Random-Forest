import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten X_train
m_X_train = []
for i in X_train:
    temp = []
    for j in i:
        for k in j:
            temp.append(k)
    m_X_train.append(temp)
m_X_train = np.asarray(m_X_train)

# Flatten X_test
m_X_test = []
for i in X_test:
    temp = []
    for j in i:
        for k in j:
            temp.append(k)
    m_X_test.append(temp)
m_X_test = np.asarray(m_X_test)

# Plot the first image
plt.imshow(m_X_train[0].reshape((28, 28)))
plt.show()

# Model
classifier = RandomForestClassifier(n_estimators=1000)
classifier.fit(m_X_train, y_train)

def predict_sample(index, data, classifier):
    return classifier.predict(data[index].reshape(1, -1))[0]

# Get a prediction
predict_sample(50223, m_X_test, classifier)