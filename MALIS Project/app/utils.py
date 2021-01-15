
# Library for understanding music
import music21 as mu

# To decide if a sample will be in training or testing (in make_sample function)
import random

# Array Processing
import numpy as np



# This file is for storing useful functions and methods to manage, convert or compute chords and whatever related to music
def read_midi(file):
    """ Read music files in format .mid and returns the piano part only aswell as the whole complete music

    Parameters
    ----------
    file : string, path to the .mid file
        Input

    Output
    ----------
    chords : list of the successive chords played by the piano.
    midi : mu.Stream, whole music conversion. Not used in this project (was for comparison purpose).
    """
    
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




# Convert a list of note to a list of 38 bits
def convert_chord_to_bits(chord) :
    """ Read music files in format .mid and returns the piano part only aswell as the whole complete music

    Parameters
    ----------
    chord : list, the chord which is simply list of the notes (of type mu.Note) composing it. 

    Output
    ----------
    bits : list of the bits representing the chord.
    midi : mu.Stream, whole music conversion. Not used in this project (was for comparison purpose).
    """
    bits = []

    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # It is a chord, which is a list of Notes
    elif isinstance(chord, mu.chord.Chord) : 
        chord_tuple = chord.notes[:5]
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 38*[0]

    else :
        print('ERROR IN CONVERSION')
        return bits

    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    if len(chord_tuple) > 5 :
        return convert_chord_to_bits(chord.notes[:5])

    else :
        # Converting the number of notes in the chord in bits
        number_of_notes = len(chord_tuple)
        bits_number_of_notes_string = f'{number_of_notes:03b}' # For Python 3.6 or higher
        for bit_string in bits_number_of_notes_string :
            bit_int =  int(bit_string)
            bits.append(bit_int)

        # Iterate through the notes in the chord
        for note in chord_tuple :
            # Converting the octave of the note in bits
            octave = note.octave 
            bits_octave_string = f'{octave:03b}'
            for bit_string in bits_octave_string :
                bit_int =  int(bit_string)
                bits.append(bit_int)

            # Converting the class of the note in bits
            class_note = note.pitch.pitchClass
            bits_class_string = f'{class_note:04b}'
            for bit_string in bits_class_string :
                bit_int =  int(bit_string)
                bits.append(bit_int)

        # Converting the NULL NOTES to bits = 0
        number_of_NULL_NOTES = 35 - 7 * number_of_notes
        for _ in range(number_of_NULL_NOTES) :
            bits.append(0)

        return bits


def convert_chord_to_bitsNumberNotes(chord) :
    """ Convert a chord (= list of note) to a list of 5 bits corresponding to its number of notes.

    Parameters
    ----------
    chord : list mu.Note[], the chord which is simply list of the notes (of type mu.Note) composing it. 


    Output
    ----------
    bits : list of 5 bits representing the number of note in the chord. The bit equal to 1 tells us the class. 
            [0,0,0,0,0] is 0 note, [1,0,0,0,0] is 1 note, [0,1,0,0,0] is 2 notes, [0,0,1,0,0] is 3 notes, [0,0,0,1,0] is 4 notes, [0,0,0,0,1] is 5 notes.
    """
    
    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # It is a chord, which is a list of Notes
    elif isinstance(chord, mu.chord.Chord) : 
        chord_tuple = chord.notes[:5]
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 5*[0]

    else :
        print('ERROR IN CONVERSION')
        return []

    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    if len(chord_tuple) > 5 :
        return convert_chord_to_bits(chord.notes[:5])

    else :
        # Converting the number of notes in the chord in bits
        number_of_notes = len(chord_tuple)
        bits = [0,0,0,0,0]
        bits[number_of_notes - 1] = 1
        return bits



def convert_chord_to_bitsNotes(chord) :
    """ Convert a list of note (= chord) to a list of 20 bits corresponding to its notes (each 4 bits is for a note)
        Explanation : Notes are among 12 possible note (C, C#, D, etc) which can be coded in 4 bits.

    Parameters
    ----------
    chord : list mu.Note[], the chord which is simply list of the notes (of type mu.Note) composing it. 


    Output
    ----------
    bits : list of 20 bits representing the different notes in the chord. The bits are the juxtaposed binary (on 4 bits) of the class of the notes.
            Class are determined by the chromatic scale https://viva.pressbooks.pub/openmusictheory/chapter/pitch-and-pitch-class/ .
            [0,0,0,0] is a null note, [0,0,0,1] is 1st note which is C, [0,0,1,0] is 2nd note which is C#, [0,0,1,1] is 3rd note which is D, etc.
    """


    bits = []

    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # It is a chord, which is a list of Notes
    elif isinstance(chord, mu.chord.Chord) : 
        chord_tuple = chord.notes[:5]
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 20*[0]

    else :
        print('ERROR IN CONVERSION')
        return []

    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    if len(chord_tuple) > 5 :
        return convert_chord_to_bits(chord.notes[:5])

    else :

        # Iterate through the notes in the chord
        for note in chord_tuple :

            # Converting the class of the note in bits
            class_note = note.pitch.pitchClass # Return the class number, the rank between 1 and 12
            bits_class_string = f'{class_note:04b}' # Convert to a binary number of 4 bits in a String
            for bit_string in bits_class_string : # for every bit of the string
                bit_int =  int(bit_string) # Convert to int type ( 0 or 1 )
                bits.append(bit_int) # append to our result
        
        # Converting the NULL NOTES to bits = 0 so that the length of the vectors are consistently equal to 20
        number_of_notes = len(chord_tuple)
        number_of_NULL_NOTES = 20 - 4 * number_of_notes
        for _ in range(number_of_NULL_NOTES) :
            bits.append(0)

        return bits



def convert_chord_to_bitsOctaves(chord) :
    """ Convert a list of note (= chord) to a list of 15 bits corresponding to its octaves (each 3 bits is for an octave)
        Explanation : Notes are among 7 possible octaves (1st, 2nd etc) which can be coded in 3 bits.
        
    Parameters
    ----------
    chord : list mu.Note[], the chord which is simply list of the notes (of type mu.Note) composing it. 


    Output
    ----------
    bits : list of 15 bits representing the different octaves in the chord. The bits are the juxtaposed binary (on 3 bits) of the octaves of the notes.
            Octaves are arbitrarly chosen to be the 7 possible octaves in a piano because we believe the range of notes by a piano is sufficient to touch most musics.
            [0,0,0] is a null octave, [0,0,1] is 1st octave of the piano (the most squeaky), [0,1,0] is 2nd octave, [0,1,1] is 3rd octave, etc.
    """


    bits = []

    # It is a "chord" of only one note, which is simply... a note
    if isinstance(chord, mu.note.Note) : 
        chord_tuple = (chord, )

    # It is a chord, which is a list of Notes
    elif isinstance(chord, mu.chord.Chord) : 
        chord_tuple = chord.notes[:5]
    
    # It is a rest, which is an empty list
    elif isinstance(chord, mu.note.Rest):
        return 15*[0]

    else :
        print('ERROR IN CONVERSION')
        return []

    # To simplify, we only consider chords of 5 notes or less, so we cut chords who are too large
    if len(chord_tuple) > 5 :
        return convert_chord_to_bits(chord.notes[:5])

    else :

        # Iterate through the notes in the chord
        for note in chord_tuple :

            # Converting the octave of the note in bits
            octave = note.octave 
            bits_octave_string = f'{octave:03b}'
            for bit_string in bits_octave_string :
                bit_int =  int(bit_string)
                bits.append(bit_int)
        
        # Converting the NULL NOTES to bits = 0 so that the length of the vectors are consistently equal to 20
        number_of_notes = len(chord_tuple)
        number_of_NULL_NOTES = 15 - 3 * number_of_notes
        for _ in range(number_of_NULL_NOTES) :
            bits.append(0)

        return bits





def convert_chord_to_ranks(chord) :
    """ Convert a list of note (= chord) to a list of 5 integers corresponding to the ranks of its notes (each number is the rank of a note)
        Explanation : Notes are among 88 possible keys (= ranks) in a piano keyboard.
        This will create our input vectors of all our Neural Networks
        
    Parameters
    ----------
    chord : list mu.Note[], the chord which is simply list of the notes (of type mu.Note) composing it. 


    Output
    ----------
    ranks : list int[] of 5 integers representing the different ranks in the chord.
            Ranks are determined by the order in the keyboard of a common piano, which has 88 keys (so a rank is between 1 and 88 and 0 for a null note)
            0 is a null note, 1 is 1st note on a piano keyboard (the squeakiest), 2 is 2nd note on a piano keyboard, 3 is 3rd note on a piano keyboard, etc.
    """


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
# Input data is 9 consecutive chords so a vector of 9*5 = 45 ranks number between 1 and 88 (the 88 notes of the piano)
# Output data is the predicted property of the 10th chord so either a vector of 5 bits (Number of notes), 20 bits (Notes) or 15 bits (Octaves)
def make_samples(chords, feature) :

    number_of_samples = len(chords) - 9

    x_train = []
    y_train = []
    
    x_test = []
    y_test = []


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

        ## Compuation of the output
        # Depend on what we want to predict : Numver of Notes, Notes, or Octaves ?
        if feature == "NN" :
            # Ouput vector y whose elements are the 5 bits representing the number of notes in the 10th consecutive chord of the sample above
            bits_next_chord = convert_chord_to_bitsNumberNotes(next_chord)
        elif feature == "N" :
            # Ouput vector y whose elements are the 20 bits representing the notes in the 10th consecutive chord of the sample above
            bits_next_chord = convert_chord_to_bitsNotes(next_chord)
        elif feature == "O" :
            # Ouput vector y whose elements are the 15 bits representing the octaves in the 10th consecutive chord of the sample above
            bits_next_chord = convert_chord_to_bitsOctaves(next_chord)
        # Deciding randomly if the sample + output will be in the training or testing data. We pick p so that ~80% are in training
        p = random.random() 
        if p <0.80 :
            x_train.append(sample_ranks)
            y_train.append(bits_next_chord)

        else :
            x_test.append(sample_ranks)
            y_test.append(bits_next_chord)


    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def get_list_layers(feature) :
    layer_NN = [45,25,5] # Layer to predict the number of notes
    layer_N = [45,28,20] # Layer to predict the notes
    layer_O = [45,30,15] # Layer to predict the octaves

    if  feature == "NN" :
        return layer_NN
    elif feature == "N" :
        return layer_N
    elif feature == "O" :
        return layer_O
    else :
        print("ERROR : Invalid feature")
        return []


def get_number_of_notes(chord) :
    """ Returns the number of notes in a chord
        This is for the SVM that predicts the number of note in a chord.
        
    Parameters
    ----------
    chord : list mu.Note[], the chord which is simply list of the notes (of type mu.Note) composing it. 


    Output
    ----------
    number_of_notes : int , the number of notes in a chord (between 0 and 5).
            If a chord has more than 5 notes, we truncate it to 5 of its notes. This case is very rare anyway (<1% for Mozart's Lacrimosa music we used).
            0 is a null note.
    """
    

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


### For the SVM only
# Make training and testing data base from the chords
# Input data is 9 consecutive chords so a vector of 9*38 = 342 bits
# Output data is the predicted feature (Numver of Note, Notes or Octaves) of the 10th chord so a vector of 5,20 or 15 bits
def make_data(chords) :
    """ Create a data set that the SVM can use from the raw list of chords in the .mid file
        
    Parameters
    ----------
    chords : list , list of every chords (type mu.Chord, mu.Note or mu.Rest) that compose the music input at the beginning. In this form, the SVM can't deal with them


    Output
    ----------
    X : np.ndarray , Array of size N x D of input vectors of the SVM, with N the number of samples and D = 5 (ranks) the dimension of an input vector.
            Ranks are determined by the order in the keyboard of a common piano, which has 88 keys (so a rank is between 1 and 88 and 0 for a null note)
            0 is a null note, 1 is 1st note on a piano keyboard (the squeakiest), 2 is 2nd note on a piano keyboard, 3 is 3rd note on a piano keyboard, etc until 88.

    y : np.ndarray , Array of size N x D of output vectors of the SVM (multi classification), with N the number of samples and D = 1 the number of notes in a chord) the dimension of an output vector.
    """


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





def output_to_chord(next_chord_NN, next_chord_N, next_chord_O) :
    """ Create a chord from the predicted output of the 3 Neural Networks (Number of Notes, Notes and Octaves)
        Returns a type mu.note.Note or mu.chord.Chord or mu.note.Rest that are readable by music software such as MuseScore 3.
        
    Parameters
    ----------
    next_chord_NN : float[] , of length 5. Vector output of the Neural Network predicting the number of notes.
    next_chord_N : float[] , of length 20. Vector output of the Neural Network predicting the notes. (5x4 bits)
    next_chord_O : float[] , of length 15. Vector output of the Neural Network predicting the octaves. (5x3 bits)


    Output
    ----------
    chord : mu.note.Note or mu.chord.Chord or mu.note.Rest. Next chord to be appended to the Stream and be read by MuseScore 3
    """
    
    ##### STEP 1 : From the predicted notes and octaves, we only keep those who are not representing a null_note
    # Here for notes
    notes_not_null = []
    for i in [0, 4, 8, 12, 16]: # Index of each note's starting bit
        note_bits = next_chord_N[0][i : i+4] # Whole bits of a single note
        # Clean the bits. Turn the 0.999 to 1 and the 0.001 to 0
        note_bits_clean = []
        for bit in note_bits :
            if bit > 0.5 :
                note_bits_clean.append(1)
            else :
                note_bits_clean.append(0)
        # Compute the class of the note by returning from a binary form to a decimal form
        note_class = 8 * note_bits_clean[0] + 4 * note_bits_clean[1] + 2 * note_bits_clean[2] + 1 * note_bits_clean[3]
        # If the class is not between 0 and 12, just return the modulo
        note_class = note_class % 13
        if note_class != 0 :
            notes_not_null.append(note_class)

    # Here for octaves
    octave_not_null = []
    for i in [0, 3, 6, 9, 12]: # Index of each octave's starting bit
        octave_bits = next_chord_O[0][i : i+3] # Whole bits of a single octave
        # Clean the bits. Turn the 0.999 to 1 and the 0.001 to 0
        octave_bits_clean = []
        for bit in octave_bits :
            if bit > 0.5 :
                octave_bits_clean.append(1)
            else :
                octave_bits_clean.append(0)
        # Compute the octave of the note by returning from a binary form to a decimal form
        octave = 4 * octave_bits_clean[0] + 2 * octave_bits_clean[1] + 1 * octave_bits_clean[2]
        # If the classe is not between 0 and 12, just return the modulo
        octave = octave % 8
        if octave != 0 :
            octave_not_null.append(note_class)

    ##### STEP 2 : Assemble de the prediction to create a chord
    number_of_notes_predicted = np.argmax(next_chord_NN) + 1 
    # Returns the number of notes we can create from our predictions
    N = min([number_of_notes_predicted, len(notes_not_null) , len(octave_not_null)])
    if N == 0 : # No notes, it's a rest of type mu.note.Rest
        rest = mu.note.Rest() 
        return rest # Return a mu.note.Rest instance
    
    else :
        chord = []
        for i in range(N) : # For each note possible to construct
            note_predicted = notes_not_null[i]
            octave_predicted = octave_not_null[i]

            # Using music21 library to create a note
            pitch = mu.pitch.Pitch(pitchClass = note_predicted, octave = octave_predicted)
            note = mu.note.Note()
            note.pitch = pitch
            chord.append(note)
        
        if N == 1 :
            return chord[0] # Return a mu.note.Note instance
        else :
            return mu.chord.Chord(chord) # Return a mu.chord.Chord instance


# Need Path class to create file for the Neural Network
from pathlib import Path
# Need pickle to save and load files
import pickle
# For listing down the file names (for the pickle)
import os

# Generate a certain number of chord from a list of 9 initial consecutive chords
# We use the Neural Network before
def generate_music_stream(start_chords, number_of_notes) :
    """ Create a mu.stream.Stream that contains a chosen number of predicted chords from 9 following inputs chords
    
    
    Parameters
    ----------
    start_chords : list , list of the 9 starting chords (either of type mu.chord.Chord or mu.note.Note)
    number_of_notes : int , Number of chords to predict


    Output
    ----------
    stream : mu.stream.Stream containing the predicted chords (only them)
    """


    ##### STEP 1 : Load the 3 Neural Networks

    # Path to Neural Network that predict the Number of Notes
    pickle_path_NN = Path("/Users/aydinabiar/Desktop/MALIS Project/app/pickles/"+ "45_25_5_" + '.pk')

    if pickle_path_NN.is_file(): # If File exists
        # Load Neural Network back to memory 
        with open(pickle_path_NN, 'rb') as fi:
            NN1_NN = pickle.load(fi)
    
    # Path to Neural Network that predict the  Notes
    pickle_path_N = Path("/Users/aydinabiar/Desktop/MALIS Project/app/pickles/"+ "45_28_20_" + '.pk')

    if pickle_path_N.is_file(): # If File exists
        # Load Neural Network back to memory 
        with open(pickle_path_N, 'rb') as fi:
            NN1_N = pickle.load(fi)

    # Path to Neural Network that predict the Octaves
    pickle_path_O= Path("/Users/aydinabiar/Desktop/MALIS Project/app/pickles/"+ "45_30_15_" + '.pk')

    if pickle_path_O.is_file(): # If File exists
        # Load Neural Network back to memory 
        with open(pickle_path_O, 'rb') as fi:
            NN1_O = pickle.load(fi)

    ##### STEP 2 : Initialization of the mu.stream.Stream that will store our chords. Stream is a type that is readable by music software MuseScore 3
    # Stream of the predicted notes
    stream = mu.stream.Stream()

    # Initialise with the 9 input consecutive chords 
    last_chords = start_chords

    ##### STEP 3 : Generate a note one by one 
    for _ in range(number_of_notes) :
    # Create the input of our 3 Neural Networks in the shape of a nd array of size 1 x 45        
        last_chords_ranks_list = [[]]
        for chord in last_chords :
            ranks = convert_chord_to_ranks(chord)
            for rank in ranks :
                last_chords_ranks_list[0].append(rank)
        last_chord_ranks = np.array(last_chords_ranks_list) #ndarray of size 1x45 which will be the input of our Neural Networks


        # Forward the input bits into the 3 Neural Network
        next_chord_NN = NN1_NN.forward(last_chord_ranks)
        next_chord_N = NN1_N.forward(last_chord_ranks)
        next_chord_O = NN1_O.forward(last_chord_ranks)

        # Convert the bits into a chord
        next_chord = output_to_chord(next_chord_NN, next_chord_N, next_chord_O)

        # Append into the stream
        stream.append(next_chord)

        # Next input of the Neural Network will be the 8 last consecutive chords of the previous input + the generated note
        next_chords = last_chords[ 1:len(last_chords)]
        next_chords.append(next_chord)
        last_chords = next_chords
    
    return stream
