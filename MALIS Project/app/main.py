# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:35:13 2020

@author: Aydin
"""
# Library for understanding music
import music21 as mu

# Search for an external software to display or play music. MuseScore 3 is good.
# Will annoy you with questions at the beginning...
# To go through them fast press Enter ==> Enter ==> n + enter ==> n + Enter ==> n + Enter ==> Enter 
# Comment this if you don't want to display or listen to music 
#mu.configure.run()

# For listing down the file names
import os

# Array Processing
import numpy as np

# To decide if a sample will be in training or testing
import random

############### IMPORTANT - Specify the path to the input music file ####################
file_input = '/Users/aydinabiar/Desktop/MALIS Project/mozart_samples/mid/lacrimosa_original.mid'

############### IMPORTANT - Specify the output path where you want the resulting generated midi file ####################
output_path = '/Users/aydinabiar/Desktop/MALIS Project/results/mozart_predicted_mediocre_10notes.mid'

############### IMPORTANT - Specify the weights of each error ####################
M_1 = 0 # Error on the number of notes
M_2 = 1 # Error on the classes of the notes
M_3 = 1 # ERror on the octaves of the notes


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
                chord_notes = element.notes # List of the notes in the chord
                chords.append(chord_notes)
            
    return chords, midi

chords, initial_stream = read_midi(file_input)
#mediocre_stream = mu.stream.Stream()
#for chord in chords :
#    mediocre_stream.append(chord)
#mediocre_stream.show()
#initial_stream.show()

# Convert a list of note to a list of 38 bits
def convert_chord_to_bits(chord) :
    
    bits = []

    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # It is a chord, which is a list of Notes
    elif isinstance(chord, tuple) : 
        chord_tuple = chord
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 38*[0]

    else :
        print('ERROR IN CONVERSION')
        return bits

    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    if len(chord_tuple) > 5 :
        return convert_chord_to_bits(chord[:5])

    else :
        # Converting the number of notes in the chord in bits
        number_of_notes = len(chord_tuple)
        bits_number_of_notes_string = f'{number_of_notes:03b}' # For Python 3.6 or higher
        for bit_string in bits_number_of_notes_string :
            bit_int =  int(bit_string)
            bits.append(bit_int)

        # Iterate through the notes in the chord
        for chord in chord_tuple :
            # Converting the octave of the note in bits
            octave = chord.octave 
            bits_octave_string = f'{octave:03b}'
            for bit_string in bits_octave_string :
                bit_int =  int(bit_string)
                bits.append(bit_int)

            # Converting the class of the note in bits
            class_note = chord.pitch.pitchClass
            bits_class_string = f'{class_note:04b}'
            for bit_string in bits_class_string :
                bit_int =  int(bit_string)
                bits.append(bit_int)

        # Converting the NULL NOTES to bits = 0
        number_of_NULL_NOTES = 35 - 7 * number_of_notes
        for i in range(number_of_NULL_NOTES) :
            bits.append(0)

        return bits


# Make training and testing data from the chords
# Input data is 9 consecutive chords so a vector of 9*38 = 342 bits
# Output data is the predicted 10th chord so a vector of 38 bits
def make_samples(chords) :

    number_of_samples = len(chords) - 9

    x_train = []
    y_train = []
    
    x_test = []
    y_test = []

    # Iterate through each sample
    for i in range(number_of_samples) :

        # Sample is 9 consecutive chords
        sample_chords = chords[i : i + 9]
        
        # Input vector x whose elements will be the 342 bits representing the sample above
        sample_bits = []

        # Converting each chord in the sample into 38 bits then append it to the input vector x
        for chord in sample_chords :
            # Convert
            bits = convert_chord_to_bits(chord)
            # Append
            for bit in bits :
                sample_bits.append(bit)

        # 10th consecutive chord of the sample
        next_chord = chords[i + 9]
        # Ouput vector y whose elements are the 38 bits representing the 10th consecutive chord of the sample above
        bits_next_chord = convert_chord_to_bits(next_chord)

        # Deciding randomly if the sample + output will be in the training or testing data. We pick p so that ~80% are in training
        p = random.random() 
        if p <0.80 :
            x_train.append(sample_bits)
            y_train.append(bits_next_chord)

        else :
            x_test.append(sample_bits)
            y_test.append(bits_next_chord)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


# Create samples 
x_train, y_train, x_test, y_test = make_samples(chords)

# To manipulate Dataframe
import pandas as pd
# To plot figures
import matplotlib.pyplot as plt
# For the Neural Network
from sklearn.model_selection import train_test_split
from scipy.stats import logistic
# Multi-Layer Perceptron
from NeuralNetwork import MLP

# For saving and loading previous NeuralNetwork. Keep persistent data and not do the whole training/testing again
from pathlib import Path
import pickle

# To differientate Neural Network with different cost parameters
pickle_name = str(M_1) + "_" + str(M_2) + "_" + str(M_3)
# Path to save and load the Neural Network
pickle_path = Path("/Users/aydinabiar/Desktop/MALIS Project/app/pickles/"+ pickle_name + '.pk')
if pickle_path.is_file(): # File exists
    # Load Neural Network back to memory 
    with open(pickle_path, 'rb') as fi:
        NN1 = pickle.load(fi)

# If it's the first time we make that Neural Network
else :
    # Neural Network (NN) parameters
    epochs=50
    learning_rate=0.01
    # Print advancement True or False, every k steps
    verbose=True
    print_every_k=10

    # Initialization of the NN.
    # 342 is the dimension of the input vector x. 
    # 38 is the dimension of the output vector y.
    # Rest is the dimension of the hidden layers
    NN1 = MLP([342, 10, 10, 10, 38])

    print('TRAINING')
    # Weights of the errors in the loss function
    NN1.M_1 = M_1 # Error weight on the number of notes
    NN1.M_2 = M_2 # Error weight on the classes
    NN1.M_3 = M_3 # Error weight on the octaves
    # Training
    NN1.training(x_train,y_train,learning_rate,epochs,verbose,print_every_k)
    # Compute the training loss and accuracy after having completed the training
    y_hat=NN1.forward(x_train)
    print('final : loss = %.3e , accuracy = %.2f %%'%(NN1.loss(y_hat,y_train),100*NN1.accuracy(y_hat,y_train)))

    # Test
    print('\nTEST')
    y_hat=NN1.forward(x_test)
    print('loss = %.3e , accuracy = %.2f %%\n'%(NN1.loss(y_hat,y_test),100*NN1.accuracy(y_hat,y_test)))

    # Open and save the Neural Netork in a pickle file
    with open(pickle_path, 'wb') as fi:
        # dump your data into the file
        pickle.dump(NN1, fi)

### To plot the loss and accuracies over time

#plt.plot(list(range(epochs)),NN1.losses,c='r',marker='o',ls='--')
#plt.title("Training Loss")
#plt.xlabel("epochs")
#plt.ylabel("loss value")
#plt.show()

#plt.plot(list(range(epochs)),NN1.accuracies,c='g',marker='o',ls='--');
#plt.title("Training accuracy")
#plt.xlabel("epochs")
#plt.ylabel("accuracy")
#plt.show()



# Smooth the output vector of the Neural Network by giving only 0 or 1 values on its elements (0.999 => 1 or 0.001 ==> 0 etc)
def clean(dirty_bits) :
    clean_bits = []
    for dirty_bit in dirty_bits[0] :
        if dirty_bit > 0.5 :
            clean_bits.append(1)
        else :
            clean_bits.append(0)
    return np.array(clean_bits)


# Convert a vector of 38 bits to a chord (= a list of mu.note.Note instances)
def read_chord_bits(chord_bits) :
    
    chord = []
    INDEX_OF_NOTES = [3,10,17,24,31] # Beginning index of the notes bits in the chord
    NULL_NOTE = [0,0,0,0,0,0,0] # Bits of no note (for example, the fifth note of a chord which has only 3 notes)
    
    # Iterating on each notes' bits
    for i in INDEX_OF_NOTES :

        if list(chord_bits[i : i+7]) != NULL_NOTE : 
            
            # Read the octave bits to translate it into an octave
            octave_bits = ''
            for bit in chord_bits[i : i+3] :
                octave_bits += str(bit)
            octave = int(octave_bits, 2)%8

            # Read the class bits to translate it into a class
            class_note_bits = ''
            for bit in chord_bits[i+3 : i+7] :
                class_note_bits += str(bit)
            class_note = int(class_note_bits)%13

            # Create a note and append it to the chord. If it's a rest, do nothing.
            if octave != 0 and class_note != 0 : # If we can define a note
                pitch = mu.pitch.Pitch(pitchClass = class_note, octave = octave)
                note = mu.note.Note()
                note.pitch = pitch
                chord.append(note)

    
    # Need to translate a "chord" of one note into a mu.note.Note instance or if there are multiple notes, into a mu.chord.Chord instance
    if len(chord) == 1 : # "chord" of one note
        return chord[0] # Return a mu.note.Note instance
    elif len(chord) > 1 : # chord of multiple notes
        return mu.chord.Chord(chord) # Return a mu.chord.Chord instance
    else : # If no notes, it's a rest
        rest = mu.note.Rest() 
        return rest # Return a mu.note.Rest instance


# Generate a certain number of chord from a list of 9 initial consecutive chords
# We use the Neural Network before
def generate_music_stream(start_chords, number_of_notes) :
    
    # Stream of the predicted notes
    stream = mu.stream.Stream()

    # Initialise with the 9 input consecutive chords 
    last_chords = start_chords
    # Generate a note one by one
    for i in range(number_of_notes) :
        # Computing the bits of the input vector of the Neural Network
        last_chords_bits_list = [[]]
        for chord in last_chords :
            bits = convert_chord_to_bits(chord)
            for bit in bits :
                last_chords_bits_list[0].append(bit)
        last_chord_bits = np.array(last_chords_bits_list)

        # Forward the input bits into the Neural Network
        next_chord_bits_dirty = NN1.forward(last_chord_bits)

        # Clean the output vector to get only 1 and 0
        next_chord_bits = clean(next_chord_bits_dirty)

        # Convert the bits into a chord
        next_chord = read_chord_bits(next_chord_bits)

        # Append into the stream
        stream.append(next_chord)

        # Next input of the Neural Network will be the 8 last consecutive chords of the previous input + the generated note
        next_chords = last_chords[ 1:len(last_chords)]
        next_chords.append(next_chord)
        last_chords = next_chords
    
    return stream



# Last 9 consecutive chords of the piece of music
last_chords = chords[len(chords)-9 :]

# Generated chords
predicted_future_stream = generate_music_stream(last_chords, 100)

# Load the prediction
predicted_future_stream.write('midi', fp='/Users/aydinabiar/Desktop/MALIS Project/results/'+pickle_name+'.mid')

# Show the prediction in MuseScore 3 for example
predicted_future_stream.show()

# Append the generated chords to the initial piece of music
final_stream = initial_stream
for chord in predicted_future_stream :
    final_stream.append(chord)

# Load the final music (initial + prediction)
#final_stream.write('midi', fp=output_path)

    
