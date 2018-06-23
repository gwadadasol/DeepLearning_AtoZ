#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:04:25 2018

@author: philippeanselme-vatin
"""

#Artificial Neural Network

#Installing Theano

#Installing Tensorflow

#Installing Keras


# Part 1 : Data preparation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Remove useless cloumn, here the first 3 columns
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # set the scaler transformation
X_test = sc.transform(X_test)


# Part 2 : Build the neuronal network

#Import Keras modules
import keras
from keras.models import Sequential # Neuron network
from keras.layers import Dense  # Init the weight of neuron network
from keras.layers import Dropout # use to reduce the overfiting

# Initialization of the neuron network
classifier = Sequential()

# add a neuron hidden layer
classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform", input_dim=11)) # when initialize the first hidden layer, need
                                                                # to specify the first layer ( here 11 column in the data)

classifier.add(Dropout(rate=0.1)) # break some link between the neurons


# add a second hidden  layer
classifier.add(Dense(units=6, activation="relu",
                     kernel_initializer="uniform")) # no need to specify the first layer

classifier.add(Dropout(rate=0.1))

# add the exit layer
# add a second hidden  layer
classifier.add(Dense(units=1, activation="sigmoid",
                     kernel_initializer="uniform"))

# Compile the neural network
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the neural network
classifier.fit(x=X_train, y=y_train,batch_size=10, epochs=100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# example d'pplication: prediction pour un client donne
exampleTest = np.array([[0.0,0.0,601.0,
                         0.0,
                              40.0,
               3.0,
               60000.0,
               2.0,
               1.0,
               1.0,
               50000.0
               ]])



exampleTest = sc.transform(exampleTest)


exampleResult = classifier.predict(exampleTest)
exampleResult = (exampleResult > 0.5)

# Partie 3 - k-fld cross validation
# Import modules
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  cross_val_score

# classifier building function
def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(units=6, activation="relu",
                         kernel_initializer="uniform", input_dim=11))  # when initialize the first hidden layer, need
    classifier.add(Dense(units=6, activation="relu",
                         kernel_initializer="uniform"))  # no need to specify the first layer

    classifier.add(Dense(units=1, activation="sigmoid",
                         kernel_initializer="uniform"))

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

# k-fold cross validation
classifier = KerasClassifier(build_classifier, batch_size=10, epochs=100 )
precisions = cross_val_score(classifier, X=X_train, y=y_train, cv=10)

moyenne = precisions.mean()
ecart_type = precisions.std()


# Partie 4 - change some paramters to optimize them

# Import modules
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import  GridSearchCV

# classifier building function
def build_classifier(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units=6, activation="relu",
                         kernel_initializer="uniform", input_dim=11))  # when initialize the first hidden layer, need

    classifier.add(Dense(units=10, activation="relu",
                         kernel_initializer="uniform"))  # no need to specify the first layer

    classifier.add(Dense(units=1, activation="sigmoid",
                         kernel_initializer="uniform"))

    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

# k-fold cross validation
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100 )

# list of the parmaters qwe want to try
# use exactly the same name as the attributes in the classifier ( creator and compile  functions)
#paramters = {"batch_size": [25, 32],
#             "epochs": [100, 500],
#             "optimizer": ["adam", "rmsprop"]}


paramters = {"batch_size": [32],
             "epochs": [500],
             "optimizer": ["adam"]}

# create the object with all the params
grid_search = GridSearchCV( estimator=classifier,
                            param_grid=paramters,
                            scoring="accuracy",
                            cv=10)

# run the evaluation
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_precision = grid_search.best_score_


