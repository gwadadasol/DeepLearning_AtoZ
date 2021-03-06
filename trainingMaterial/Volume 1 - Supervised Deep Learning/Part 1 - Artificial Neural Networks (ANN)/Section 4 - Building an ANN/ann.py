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


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 : Build the neuronal network

#Import Keras modules
import keras
from keras.models import Sequential # Neuron network
from keras.layers import Dense  # Init the weight of neuron network

# Initialization of the neuron network
classifier = Sequential()

# add a neuron layer
classifier.add(Dense(units=6, activation="relu", 
                     kernel_initializer="uniform", input_dim=11))

