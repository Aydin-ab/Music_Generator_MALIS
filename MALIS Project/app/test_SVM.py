# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:35:13 2020

@author: Aydin
"""

#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

# Library for understanding music
import music21 as mu

# For listing down the file names
import os

# To decide if a sample will be in training or testing
import random

############### IMPORTANT - Specify the path to the input music file ####################
file_input = '/Users/aydinabiar/Desktop/MALIS Project/mozart_samples/mid/lacrimosa_original.mid'

############### IMPORTANT - Specify the output path where you want the resulting generated midi file ####################
output_path = '/Users/aydinabiar/Desktop/MALIS Project/results/mozart_predicted_mediocre_10notes.mid'


########## This is a test algorithm using piano only ##########
# The first chapters of the music21 documentations is well-enough to understand all the code
# Available here http://web.mit.edu/music21/doc/index.html

# Defining function to read MIDI files. 
def read_midi(file):
    
    print("Loading Music File:",file)
    
    chords=[]
    chords_to_parse = None
    
    # Parsing a midi file
    midi = mu.converter.parse(file)

    # Grouping based on different instruments
    s2 = mu.instrument.partitionByInstrument(midi)

    # Looping over all the instruments
    for part in s2.parts:
    
        chords_to_parse = part.recurse() 
    
        # Finding whether a particular element is note or a chord
        for element in chords_to_parse:
            
            # Note
            if isinstance(element, mu.note.Note):
                chords.append(element)
            
            # Chord
            elif isinstance(element, mu.chord.Chord):
                #chord_notes = element.notes # List of the notes in the chord
                chords.append(element)
            
    return chords, midi

chords, initial_stream = read_midi(file_input)
#mediocre_stream = mu.stream.Stream()
#for chord in chords :
#    mediocre_stream.append(chord)
#mediocre_stream.show()
#initial_stream.show()

# Return the number of notes in a chord
def get_number_of_notes(chord) :
    

    # It is a "chord" of only one note, which is simply... a note, return just 1 note
    if isinstance(chord, mu.note.Note) : 
        return 1

    # It is a chord, which is a list of Notes, return he number of notes
    elif isinstance(chord, mu.chord.Chord) : 
        # Getting the number of notes
        number_of_notes = len(chord)

        # To simplify, we only consider chords of 5 notes or less, 
        # so we say large chords are just chord with 5 notes...
        if number_of_notes >= 5 :
            return 5
        else :
            return number_of_notes

    else :
        print('ERROR IN CONVERSION')
        return 


# Convert a list of note (chord) to a list of ranks
def convert_chord_to_ranks(chord) :

    ranks = []

    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # If it's a chord of multiple notes
    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    elif isinstance(chord, mu.chord.Chord) :
        chord_tuple = chord.notes[:5]
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 5*[0]

    else :
        print('ERROR IN CONVERSION')
        return ranks

    if len(chord_tuple) > 5 :
        return convert_chord_to_ranks(chord[:5])

    else :
        # Iterate through the notes in the chord
        for chord in chord_tuple :
            octave = chord.octave             
            class_note = chord.pitch.pitchClass
            rank = class_note + 12*(octave-1)
            ranks.append(rank)

        # Converting the NULL NOTES to rank = 0
        number_of_NULL_NOTES = 5 - len(chord_tuple)
        for _ in range(number_of_NULL_NOTES) :
            ranks.append(0)

        return ranks

# Make training and testing data from the chords
# Input data is 9 consecutive chords so a vector of 9*38 = 342 bits
# Output data is the predicted 10th chord so a vector of 38 bits
def make_data(chords) :

    number_of_samples = len(chords) - 9

    X = []
    y = []


    # Iterate through each sample
    for i in range(number_of_samples) :

        # Sample is 9 consecutive chords
        sample_chords = chords[i : i + 9]
        
        # Input vector x whose elements will be the 5*9 = 45 ranks representing the sample above
        sample_ranks = []

        # Converting each chord in the sample into the 5 ranks of its notes and then append them to the input vector x
        for chord in sample_chords :
            # Convert
            ranks = convert_chord_to_ranks(chord)
            # Append
            for rank in ranks :
                sample_ranks.append(rank)

        # 10th consecutive chord of the sample
        next_chord = chords[i + 9]

        # Ouput y which is the predicted number of notes in the 10th chord
        number_of_notes_predicted = get_number_of_notes(next_chord)

        # Adding the sample to the dataset
        X.append(sample_ranks)
        y.append(number_of_notes_predicted)

    return np.array(X), np.array(y)


# Create dataset 
X, y = make_data(chords)

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