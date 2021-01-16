# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:35:13 2020

@author: Aydin
"""
########## This is a test algorithm using piano only ##########
# The first chapters of the music21 documentations is well-enough to understand all the code
# Available here http://web.mit.edu/music21/doc/index.html


#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# Library for understanding music
import music21 as mu

# Useful functions to manipulate and adapt music data to our models
import utils as u

# For listing down the file names
import os


# To decide if a sample will be in training or testing
import random

############### IMPORTANT - Specify the path to the input music file ####################
file_input = '/Users/aydinabiar/Desktop/MALIS Project/mozart_samples/mid/lacrimosa_original.mid'



# Read and store the chords of the file
chords, initial_stream = u.read_midi(file_input)

# Create dataset 
X, y = u.make_data(chords)

# Create training and testing data. use parameter random_state = int to fix one data for multiple calls
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Fitting a multi classification model using SVM with different kernels :
#  Linear, Radial Basis, Polynomial and Sigmoid
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)

# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = 100 * linear.score(X_test, y_test)
accuracy_poly = 100 * poly.score(X_test, y_test)
accuracy_rbf = 100 * rbf.score(X_test, y_test)
accuracy_sig = 100 * sig.score(X_test, y_test)
print("Accuracy Linear Kernel:", accuracy_lin, "%")
print("Accuracy Polynomial Kernel:", accuracy_poly, "%")
print("Accuracy Radial Basis Kernel:", accuracy_rbf, "%")
print("Accuracy Sigmoid Kernel:", accuracy_sig, "%")

# Plot bar chart of accuracy
kernels = ['linear', 'polynomial d°3', 'radial \u03B3 = 1', 'sigmoid']
accuracies = [accuracy_lin, accuracy_poly, accuracy_rbf, accuracy_sig]
plt.bar(kernels, accuracies)
plt.ylabel('Accuracy (in %)')
plt.ylim(0,100)
plt.yticks(np.arange(0,101,10))
plt.xlabel('Kernel')
plt.title('Accuracy for different kernels')
plt.grid()
plt.show()