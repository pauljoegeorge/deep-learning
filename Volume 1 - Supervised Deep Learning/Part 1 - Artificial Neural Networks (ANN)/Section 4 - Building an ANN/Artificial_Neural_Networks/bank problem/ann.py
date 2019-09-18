# -*- coding: utf-8 -*-

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

"""# Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""# Importing the dataset"""
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #creditscore ...estimated salary
y = dataset.iloc[:, 13].values

"""# Encoding categorical data"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #country
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #gender
onehotencoder = OneHotEncoder(categorical_features = [1]) #dummy variables for categories - France, Germany and Spain 001 )
X = onehotencoder.fit_transform(X).toarray()
# now there are 3 dummy variables, remove one dummy variable to avoid dummy variable trap ( so removing index 1 )
X = X[:, 1:]

"""# Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


"""# Feature Scaling"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
# Making ANN
    - kernel_initializer = dummy weight to hidden layer
    - 6: no of nodes for hidden layer
"""
#Import Keras
import keras
from keras.models import Sequential # To initialize ANN
from keras.layers import Dense  # To create layers

#Initializing ANN
classifier = Sequential()
# Adding first input layer and hidden layer
classifier.add(Dense(6, activation="relu", kernel_initializer='uniform'))
# Adding the second hidden layer
classifier.add(Dense(6, activation="relu", kernel_initializer='uniform'))
# Adding output layer
classifier.add(Dense(1, activation="sigmoid", kernel_initializer='uniform'))  #activation softmax  will be used ,if more than 2 categories.

# compile ANN''
   # - §adam§ -> optimized weight using backpropogation -> Stochastic Gradient Descent
   # if output layer has more than 2 output, categorical_crossentropy 
   # accuracy is the expected difference in errror rate
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])


"""
Fitting ANN to the Training set
  - batch_size: no of iterations after which weight has to be updated
"""
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test) #y_pred => probability to leave the bank
y_pred = (y_pred > 0.5) # true if y_pred > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)