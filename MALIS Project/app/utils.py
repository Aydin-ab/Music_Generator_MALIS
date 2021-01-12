
# Library for understanding music
import music21 as mu

# To decide if a sample will be in training or testing (in make_sample function)
import random

# Array Processing
import numpy as np


# This file is for storing useful functions and methods to manage, convert or compute chords and whatever related to music
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




# Convert a list of note to a list of 38 bits
def convert_chord_to_bits(chord) :
    
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


## For predicting the notes in a chord
# Convert a list of note (= chord) to a list of 20 bits corresponding to its notes (each 4 bits is a note)
# Explanation : Notes are among 12 possible note (C, C#, D, etc) which can be coded in 4 bits
## For predicting the number of notes
# Convert a list of note (= chord) to a list of 5 bits corresponding to its number of notes
def convert_chord_to_bitsNotes(chord) :
    
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
        for i in range(number_of_NULL_NOTES) :
            bits.append(0)

        return bits


## For predicting the number of notes
# Convert a list of note (= chord) to a list of 5 bits corresponding to its number of notes
def convert_chord_to_bitsNumberNotes(chord) :
    
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
        for i in range(number_of_NULL_NOTES) :
            ranks.append(0)

        return ranks



# Make training and testing data from the chords
# Input data is 9 consecutive chords so a vector of 9*5 = 45 ranks number between 1 and 88 (the 88 notes of the piano)
# Output data is the predicted property of the 10th chord so either a vector of 5 bits (Number of notes), 20 bits (Notes) or 15 bits (Octaves)
def make_samples(chords, prediction) :

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
        if prediction == "NN" :
            # Ouput vector y whose elements are the 5 bits representing the number of notes in the 10th consecutive chord of the sample above
            bits_next_chord = convert_chord_to_bitsNumberNotes(next_chord)
        elif prediction == "N" :
            # Ouput vector y whose elements are the 20 bits representing the notes in the 10th consecutive chord of the sample above
            bits_next_chord = convert_chord_to_bitsNotes(next_chord)

        # Deciding randomly if the sample + output will be in the training or testing data. We pick p so that ~80% are in training
        p = random.random() 
        if p <0.80 :
            x_train.append(sample_ranks)
            y_train.append(bits_next_chord)

        else :
            x_test.append(sample_ranks)
            y_test.append(bits_next_chord)


    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)



'''
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
        # Computing the ranks of the input vector of the Neural Network
        last_chords_ranks_list = [[]]
        for chord in last_chords :
            ranks = convert_chord_to_ranks(chord)
            for rank in ranks :
                last_chords_ranks_list[0].append(rank)
        last_chord_ranks = np.array(last_chords_ranks_list)

        # Forward the input bits into the Neural Network
        next_chord_bits_dirty = NN1.forward(last_chord_ranks)

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
'''